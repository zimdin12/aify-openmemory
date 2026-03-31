"""
Enhanced memory utilities for intelligent memory management.
Provides hybrid search, smart memory addition, and relationship analysis.
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any


def _strip_json_fences(text):
    """Strip markdown code fences and think blocks from JSON output (qwen3.5+ models)."""
    text = text.strip()
    # Strip <think>...</think> blocks (safety net if think:false is ignored)
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    if text.startswith('```'):
        first_nl = text.find('\n')
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith('```'):
            text = text.rstrip()[:-3].rstrip()
    return text
from dataclasses import dataclass

from service.database import SessionLocal
from service.database.models import Memory, MemoryState
from service.memory.client import get_memory_client


@dataclass
class MemorySearchResult:
    """Enhanced search result combining vector, graph, and temporal data"""
    id: str
    content: str
    score: float
    source: str  # 'vector', 'graph', 'temporal'
    metadata: Dict[str, Any]
    relationships: List[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class MemoryAdditionResult:
    """Result of smart memory addition"""
    added_memories: List[Dict[str, Any]]
    updated_memories: List[Dict[str, Any]]
    skipped_facts: List[str]
    related_memories: List[Dict[str, Any]]
    status: str  # 'new', 'updated', 'merged', 'duplicate'
    summary: str


class EnhancedMemoryManager:
    """Enhanced memory manager with hybrid search and smart addition"""

    # Per-session conversation context for LLM review.
    # Key: "user_id:client_name", Value: list of {"role": ..., "content": ...}
    # Maintained automatically — no caller action needed.
    _session_contexts: Dict[str, List[Dict[str, str]]] = {}
    _MAX_CONTEXT_MESSAGES = 10  # 5 turns (user + assistant each)

    def __init__(self):
        self.memory_client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize memory client safely"""
        try:
            self.memory_client = get_memory_client()
        except Exception as e:
            logging.warning(f"Failed to initialize memory client: {e}")
            self.memory_client = None

    def _is_context_enabled(self) -> bool:
        """Check if session context retention is enabled via env var."""
        import os
        return os.environ.get('SESSION_CONTEXT', 'true').lower() not in ('false', '0', 'no', 'off')

    def _get_session_context(self, user_id: str, client_name: str = "") -> List[Dict[str, str]]:
        """Get recent conversation context for this session."""
        if not self._is_context_enabled():
            return []
        key = f"{user_id}:{client_name}"
        return self._session_contexts.get(key, [])[-self._MAX_CONTEXT_MESSAGES:]

    def _append_session_context(self, user_id: str, client_name: str = "",
                                 user_msg: str = "", assistant_msg: str = ""):
        """Append a conversation turn to session context (auto-truncates)."""
        if not self._is_context_enabled():
            return
        key = f"{user_id}:{client_name}"
        if key not in self._session_contexts:
            self._session_contexts[key] = []
        if user_msg:
            self._session_contexts[key].append({"role": "user", "content": user_msg[:500]})
        if assistant_msg:
            self._session_contexts[key].append({"role": "assistant", "content": assistant_msg[:500]})
        # Keep only last N messages
        self._session_contexts[key] = self._session_contexts[key][-self._MAX_CONTEXT_MESSAGES:]

    def hybrid_search(self, query: str, user_id: str, limit: int = 10) -> List[MemorySearchResult]:
        """
        Perform hybrid search across vector, graph, and temporal dimensions
        """
        if not self.memory_client:
            return []

        results = []

        # 1. Vector similarity search
        try:
            vector_results = self._vector_search(query, user_id, limit)
            results.extend(vector_results)
        except Exception as e:
            logging.warning(f"Vector search failed: {e}")

        # 2. Graph relationship search
        try:
            graph_results = self._graph_search(query, user_id, limit)
            results.extend(graph_results)
        except Exception as e:
            logging.warning(f"Graph search failed: {e}")

        # 3. Temporal context search
        try:
            temporal_results = self._temporal_search(query, user_id, limit)
            results.extend(temporal_results)
        except Exception as e:
            logging.warning(f"Temporal search failed: {e}")

        # 4. Deduplicate and rank results
        return self._deduplicate_and_rank(results, limit)

    def _vector_search(self, query: str, user_id: str, limit: int) -> List[MemorySearchResult]:
        """Vector similarity search in Qdrant"""
        results = []

        try:
            filters = {"user_id": user_id}
            embeddings = self.memory_client.embedding_model.embed(query, "search")

            hits = self.memory_client.vector_store.search(
                query=query,
                vectors=embeddings,
                limit=limit,
                filters=filters,
            )

            for hit in hits:
                results.append(MemorySearchResult(
                    id=hit.id,
                    content=hit.payload.get("data", ""),
                    score=hit.score,
                    source="vector",
                    metadata=hit.payload,
                    created_at=hit.payload.get("created_at"),
                    updated_at=hit.payload.get("updated_at")
                ))
        except Exception as e:
            logging.warning(f"Vector search error: {e}")

        return results

    def _graph_search(self, query: str, user_id: str, limit: int) -> List[MemorySearchResult]:
        """Graph relationship search in Neo4j"""
        results = []

        try:
            if hasattr(self.memory_client, 'graph') and self.memory_client.graph:
                # Extract entities from query
                entities = self._extract_entities_from_query(query)

                # Search for related entities and relationships
                for entity in entities:
                    related = self._find_related_entities(entity, user_id)
                    for rel in related:
                        results.append(MemorySearchResult(
                            id=rel.get('id', str(uuid.uuid4())),
                            content=rel.get('content', ''),
                            score=rel.get('relevance', 0.7),
                            source="graph",
                            metadata=rel,
                            relationships=rel.get('relationships', [])
                        ))
        except Exception as e:
            logging.warning(f"Graph search error: {e}")

        return results

    def _temporal_search(self, query: str, user_id: str, limit: int) -> List[MemorySearchResult]:
        """Temporal context search based on recency and patterns"""
        results = []

        try:
            from service.database.utils import get_user_and_app
            db = SessionLocal()
            try:
                # Get user object to get the UUID
                try:
                    user, _ = get_user_and_app(db, user_id=user_id, app_id="temporal_search")
                    actual_user_id = user.id
                except Exception:
                    # If we can't get user, skip temporal search
                    return results

                # Get recent memories using the actual UUID
                current_time = datetime.utcnow()
                recent_memories = db.query(Memory).filter(
                    Memory.user_id == actual_user_id,
                    Memory.state == MemoryState.active,
                    Memory.created_at >= current_time - timedelta(days=30)
                ).order_by(Memory.created_at.desc()).limit(limit).all()

                # Score based on recency and relevance
                for memory in recent_memories:
                    age_days = (current_time - memory.created_at).days
                    recency_score = max(0.3, 1.0 - (age_days / 30))  # Decay over 30 days

                    # Simple relevance check
                    relevance = self._calculate_text_relevance(query, memory.content)
                    combined_score = (recency_score * 0.3) + (relevance * 0.7)

                    if combined_score > 0.4:  # Threshold for inclusion
                        results.append(MemorySearchResult(
                            id=str(memory.id),
                            content=memory.content,
                            score=combined_score,
                            source="temporal",
                            metadata={"age_days": age_days, "recency_score": recency_score},
                            created_at=memory.created_at,
                            updated_at=memory.updated_at
                        ))
            finally:
                db.close()
        except Exception as e:
            logging.warning(f"Temporal search error: {e}")

        return results

    def smart_add_memory(self, text: str, user_id: str, metadata: Dict = None,
                          client_name: str = "") -> MemoryAdditionResult:
        """
        Intelligently add memory by checking existing memories and extracting only new facts.
        Includes LLM review step to fix orphaned facts and drop noise.
        """
        if not self.memory_client:
            return MemoryAdditionResult([], [], [], [], "error", "Memory system unavailable")

        # 1. Search for existing related memories
        existing_memories = self.hybrid_search(text, user_id, limit=20)

        # 2. Extract facts from new text
        #    Short inputs (under 100 chars, single sentence) bypass extraction —
        #    they are already a single fact and shouldn't be filtered by min length.
        #    LLM noise check filters greetings/filler in any language.
        stripped = text.strip()
        if len(stripped) < 100 and len(stripped) >= 10 and '\n' not in stripped:
            if self._llm_is_noise(stripped):
                new_facts = []
            else:
                new_facts = [stripped]
        else:
            new_facts = self._extract_facts(text)
        # 2b. Inject topic context into orphaned facts (fast heuristic)
        new_facts = self._inject_context(new_facts, text)
        # 2c. LLM review: fix missing context, drop noise, merge fragments
        if new_facts:
            new_facts = self._llm_review_facts(
                candidate_facts=new_facts,
                source_text=text,
                user_id=user_id,
                client_name=client_name,
            )

        # 3. Compare with existing facts
        existing_facts = []
        for memory in existing_memories:
            if memory.score > 0.7:  # High similarity threshold
                facts = self._extract_facts(memory.content)
                existing_facts.extend(facts)

        # 4. Identify truly new facts
        novel_facts = self._find_novel_facts(new_facts, existing_facts, user_id=user_id)

        # 5. Add only new information
        #    - Store each fact individually in vector store (infer=False, graph=False)
        #    - Do ONE batched graph extraction with the full original text
        if novel_facts:
            all_results = []
            for fact in novel_facts:
                try:
                    # infer=False: skip LLM re-extraction
                    # graph=False: skip graph extraction per-fact (we batch it below)
                    response = self.memory_client.add(
                        [{"role": "user", "content": fact}],
                        user_id=user_id,
                        metadata=metadata or {},
                        infer=False,
                        graph=False,
                    )
                    all_results.extend(response.get('results', []))
                except Exception as e:
                    logging.warning(f"Failed to add fact '{fact[:50]}': {e}")

            try:
                if hasattr(self.memory_client, 'enable_graph') and self.memory_client.enable_graph:
                    import threading
                    ctx = self._get_session_context(user_id, client_name) or None
                    t = threading.Thread(
                        target=self._background_graph_extract,
                        args=(self.memory_client, text, user_id, ctx),
                        daemon=True,
                    )
                    t.start()
            except Exception as e:
                logging.warning(f"Graph extraction thread start failed: {e}")

            response = {'results': all_results}

            status = "new" if not existing_memories else "updated"
            summary = f"Added {len(novel_facts)} new facts. Found {len(existing_facts)} existing related facts."

            return MemoryAdditionResult(
                added_memories=response.get('results', []) if response else [],
                updated_memories=[],
                skipped_facts=[f for f in new_facts if f not in novel_facts],
                related_memories=[m.__dict__ for m in existing_memories[:5]],
                status=status,
                summary=summary
            )
        else:
            return MemoryAdditionResult(
                added_memories=[],
                updated_memories=[],
                skipped_facts=new_facts,
                related_memories=[m.__dict__ for m in existing_memories[:5]],
                status="duplicate",
                summary=f"No new facts found. All {len(new_facts)} facts already exist in memory."
            )

    @staticmethod
    def _background_graph_extract(mem_client, full_text: str, uid: str,
                                   context: List[Dict[str, str]] = None):
        """Chunked graph extraction in background thread."""
        try:
            max_chunk = 2000
            if len(full_text) <= max_chunk:
                chunks = [full_text]
            else:
                paragraphs = re.split(r'\n\n+', full_text)
                chunks = []
                current = ""
                for para in paragraphs:
                    if len(current) + len(para) > max_chunk and current:
                        chunks.append(current.strip())
                        current = para
                    else:
                        current = current + "\n\n" + para if current else para
                if current.strip():
                    chunks.append(current.strip())

            total_added = []
            for i, chunk in enumerate(chunks):
                try:
                    filters = {"user_id": uid}
                    # Pass session context to graph extraction via transient filter key
                    if context:
                        filters["_session_context"] = context
                    result = mem_client._add_to_graph(
                        [{"role": "user", "content": chunk}],
                        filters,
                    )
                    added = result.get('added_entities', [])
                    total_added.extend(added)
                    logging.info(f"Graph chunk {i+1}/{len(chunks)}: {len(added)} entities added")
                except Exception as e:
                    logging.warning(f"Graph chunk {i+1}/{len(chunks)} failed: {e}")
            logging.info(f"Graph extraction completed: {len(chunks)} chunks, {len(total_added)} total entities")
        except Exception as e:
            logging.warning(f"Graph extraction failed (non-fatal): {e}")

    def _smart_add_reviewed_fact(self, fact: str, user_id: str, metadata: Dict = None) -> MemoryAdditionResult:
        """Store a single already-reviewed fact (skip regex extraction and LLM review).

        Used by comprehensive_memory_handle where facts are already reviewed.
        Still does vector dedup and graph extraction.
        """
        if not self.memory_client:
            return MemoryAdditionResult([], [], [], [], "error", "Memory system unavailable")

        # Dedup check
        novel = self._find_novel_facts([fact], [], user_id=user_id)
        if not novel:
            return MemoryAdditionResult([], [], [fact], [], "duplicate",
                                         f"Fact already exists in memory.")

        # Store
        try:
            response = self.memory_client.add(
                [{"role": "user", "content": fact}],
                user_id=user_id,
                metadata=metadata or {},
                infer=False,
                graph=False,
            )
            results = response.get('results', [])
        except Exception as e:
            logging.warning(f"Failed to add reviewed fact '{fact[:50]}': {e}")
            return MemoryAdditionResult([], [], [], [], "error", str(e))

        return MemoryAdditionResult(
            added_memories=results,
            updated_memories=[],
            skipped_facts=[],
            related_memories=[],
            status="new",
            summary=f"Stored 1 fact."
        )

    def comprehensive_memory_handle(self, user_message: str, llm_response: str,
                                     user_id: str, client_name: str = "") -> Dict[str, Any]:
        """
        Conversation memory processing: extract facts via regex, then LLM-review
        for quality (fix missing context, drop noise, merge fragments).

        Session context is maintained automatically — each call appends the
        conversation turn, and the LLM review sees recent history.
        """
        result = {
            "extracted_memories": [],
            "related_context": [],
            "memory_updates": [],
            "insights": [],
            "status": "processed"
        }

        try:
            # 1. Extract candidate facts via regex (fast, ~1ms)
            candidate_facts = self._extract_memorable_content(user_message, llm_response)

            if not candidate_facts:
                result["status"] = "no_memorable_content"
                # Still append context so LLM has history for next call
                self._append_session_context(user_id, client_name, user_message, llm_response)
                return result

            # 2. LLM review: fix context, drop noise, merge fragments
            #    Uses server-side session context automatically
            reviewed_facts = self._llm_review_facts(
                candidate_facts=candidate_facts,
                user_id=user_id,
                client_name=client_name,
                user_message=user_message,
                llm_response=llm_response,
            )

            result["extracted_memories"] = reviewed_facts

            # 3. Append this turn to session context (for next call's LLM review)
            self._append_session_context(user_id, client_name, user_message, llm_response)

            # 4. Search for related existing memories
            related_memories = []
            search_queries = list(set(reviewed_facts[:5]))
            for content in search_queries:
                related = self.hybrid_search(content, user_id, limit=5)
                related_memories.extend(related)

            # 5. Smart addition of reviewed facts (skip LLM review — already done above)
            for content in reviewed_facts:
                addition_result = self._smart_add_reviewed_fact(content, user_id)
                result["memory_updates"].append({
                    "content": content,
                    "result": addition_result.__dict__
                })

            # 6. Background graph extraction from the full conversation turn
            #    Combines user + assistant text for richer entity extraction
            full_turn_text = f"{user_message}\n{llm_response}"
            try:
                if hasattr(self.memory_client, 'enable_graph') and self.memory_client.enable_graph:
                    import threading
                    ctx = self._get_session_context(user_id, client_name) or None
                    t = threading.Thread(
                        target=self._background_graph_extract,
                        args=(self.memory_client, full_turn_text, user_id, ctx),
                        daemon=True,
                    )
                    t.start()
            except Exception as e:
                logging.warning(f"Graph extraction thread start failed: {e}")

            # 7. Provide contextual insights
            result["related_context"] = [m.__dict__ for m in related_memories[:10]]
            result["insights"] = self._generate_memory_insights(related_memories, reviewed_facts)

            # 7. Identify patterns
            result["patterns"] = self._identify_conversation_patterns(user_message, llm_response, related_memories)

        except Exception as e:
            logging.error(f"Comprehensive memory handle error: {e}")
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def _llm_is_noise(self, text: str) -> bool:
        """Use LLM to check if short text is noise (greetings, filler, etc). Language-agnostic."""
        from service.memory.llm import llm_chat
        from service.memory.model_configs import detect_model_family as _detect_model_family, get_llm_options as _get_llm_options
        model = os.environ.get('LLM_MODEL', 'qwen3.5-9b')
        options = _get_llm_options(_detect_model_family(model))
        try:
            content = llm_chat(
                messages=[
                    {'role': 'system', 'content': 'You classify text for a memory system. Respond with JSON only.'},
                    {'role': 'user', 'content': (
                        'Does this text state a FACT about someone or something? '
                        'KEEP only if it declares a preference, skill, decision, project detail, or personal info. '
                        'DROP if it is: a greeting, small talk, weather, daily routine, '
                        'a question, a request/command (even if it mentions technologies like "search Python" '
                        'or "show me Rust"), or an acknowledgment ("ok", "thanks", "got it"). '
                        'The key test: does it TELL something new, or does it ASK/REQUEST something?\n\n'
                        f'Text: "{text}"\n\nRespond: {{"keep": true/false}}'
                    )},
                ],
                model=model, json_mode=True, options=options, timeout=10,
            )
            result = json.loads(content)
            is_noise = not result.get('keep', True)
            if is_noise:
                logging.info(f"LLM noise filter: dropped '{text[:60]}'")
            return is_noise
        except Exception as e:
            logging.warning(f"LLM noise check failed: {e}")
        return False  # Default: not noise (don't lose data)

    def _llm_review_facts(self, candidate_facts: List[str],
                           source_text: str = "",
                           user_id: str = "", client_name: str = "",
                           user_message: str = "", llm_response: str = "") -> List[str]:
        """Use LLM to review and fix regex-extracted facts.

        Works for both add_memories (source_text) and conversation_memory (user_message + llm_response).
        Uses server-side session context automatically — no caller action needed.

        Controlled by LLM_FACT_REVIEW env var (default: "true"). Set to "false" to disable.

        Falls back to original candidates if LLM call fails.
        """
        import os

        # Check if LLM review is enabled
        if os.environ.get('LLM_FACT_REVIEW', 'true').lower() in ('false', '0', 'no', 'off'):
            logging.info("LLM fact review disabled (LLM_FACT_REVIEW=false)")
            return candidate_facts

        model = os.environ.get('LLM_MODEL', 'qwen3.5-9b')

        # Model-specific sampling options
        from service.memory.llm import llm_chat
        from service.memory.model_configs import detect_model_family as _detect_model_family, get_llm_options as _get_llm_options
        family = _detect_model_family(model)
        options = _get_llm_options(family)

        # Build conversation context from server-side session history
        session_context = self._get_session_context(user_id, client_name)

        # Assemble the review prompt
        # Part 1: Recent conversation context (from session history)
        context_block = ""
        if session_context:
            context_lines = []
            for turn in session_context[-6:]:  # Last 3 turns
                role = turn.get("role", "user")
                content = turn.get("content", "")[:500]
                context_lines.append(f"[{role}]: {content}")
            context_block = "RECENT CONVERSATION:\n" + "\n".join(context_lines) + "\n\n"

        # Part 2: Source material (either conversation turn or raw text)
        if user_message or llm_response:
            source_block = f"CURRENT TURN:\n[user]: {user_message[:800]}\n[assistant]: {llm_response[:800]}"
        elif source_text:
            source_block = f"SOURCE TEXT:\n{source_text[:1500]}"
        else:
            source_block = ""

        # Part 3: Candidate facts
        facts_list = "\n".join(f"{i+1}. {f}" for i, f in enumerate(candidate_facts))

        prompt = f"""{context_block}{source_block}

EXTRACTED FACTS:
{facts_list}

REVIEW TASK:
Review the extracted facts above. Fix or improve them:
1. Each fact MUST include its subject (person, project, entity name) — add it if missing
2. DROP facts that are noise — greetings, small talk, weather chat, daily routine without specifics, instructions to the AI, meta-commentary ("here is what I found"). A fact is ONLY worth keeping if it contains specific information about a person, project, preference, skill, or decision that would be useful to recall later.
3. MERGE fragments that describe the same thing into one clear fact
4. SPLIT compound facts that contain multiple distinct pieces of information
5. Keep facts concise but self-contained (someone reading just the fact should understand it)

Return ONLY a JSON object: {{"facts": ["fact1", "fact2", ...]}}
Return empty list if nothing is worth remembering: {{"facts": []}}"""

        try:
            system_prompt = (
                "You are a strict fact reviewer for a memory system. "
                "You receive conversation context and extracted facts. "
                "Your job: ensure every fact is self-contained and includes its subject "
                "(person, project, entity name). Use the conversation context to identify "
                "what is being discussed and add missing subjects. "
                "AGGRESSIVELY drop noise: greetings, small talk, weather, daily routine, "
                "filler, instructions. Only keep facts with specific memorable information. "
                "Return only JSON."
            )

            content = llm_chat(
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                model=model, json_mode=True, options=options, timeout=30,
            )

            parsed = json.loads(_strip_json_fences(content))
            reviewed = parsed.get('facts', [])

            if not isinstance(reviewed, list) or len(reviewed) == 0:
                logging.info("LLM review returned empty/invalid, using regex facts")
                return candidate_facts

            # Filter: must be strings, min length 20 chars
            reviewed = [f.strip() for f in reviewed if isinstance(f, str) and len(f.strip()) >= 20]

            if len(reviewed) == 0:
                logging.info("LLM review filtered to 0 facts, using regex facts")
                return candidate_facts

            logging.info(f"LLM fact review: {len(candidate_facts)} candidates → {len(reviewed)} reviewed facts")
            return reviewed

        except Exception as e:
            logging.warning(f"LLM fact review error ({e}), using regex facts as fallback")
            return candidate_facts

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential entities from search query"""
        import re as _re
        # Strip punctuation, lowercase, split
        cleaned = _re.sub(r'[^\w\s]', '', query.lower())
        words = cleaned.split()
        # Filter out common/stop words, keep potential entities
        stopwords = frozenset([
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'she', 'use', 'way', 'what', 'does', 'this', 'that', 'with',
            'from', 'have', 'been', 'will', 'would', 'could', 'should', 'there',
            'their', 'them', 'they', 'then', 'than', 'when', 'where', 'which',
            'about', 'into', 'some', 'also', 'just', 'more', 'very', 'much',
            'work', 'used', 'main', 'exist',
        ])
        entities = [w for w in words if len(w) > 2 and w not in stopwords]
        return entities[:5]

    def _find_related_entities(self, entity: str, user_id: str) -> List[Dict]:
        """Find entities related to given entity in Neo4j graph"""
        results = []
        try:
            if not (hasattr(self.memory_client, 'graph') and self.memory_client.graph):
                return results
            graph = self.memory_client.graph
            if not hasattr(graph, 'graph'):
                return results
            driver = graph.graph.driver if hasattr(graph.graph, 'driver') else None
            if not driver:
                # Try _driver attribute (neo4j graph store)
                driver = getattr(graph.graph, '_driver', None)
            if not driver:
                return results

            query = """
            MATCH (n)-[r]->(m)
            WHERE (toLower(n.name) CONTAINS toLower($entity)
                   OR toLower(m.name) CONTAINS toLower($entity))
            AND n.name <> m.name
            RETURN n.name AS source, type(r) AS relationship, m.name AS target
            LIMIT 5
            """
            with driver.session() as session:
                records = session.run(query, entity=entity)
                for record in records:
                    src = record['source'].replace('_', ' ')
                    rel = record['relationship'].replace('_', ' ')
                    tgt = record['target'].replace('_', ' ')
                    content = f"{src} {rel} {tgt}"
                    results.append({
                        'content': content,
                        'relevance': 0.65,
                        'relationships': [
                            {'source': record['source'], 'rel': record['relationship'], 'target': record['target']}
                        ]
                    })
        except Exception as e:
            logging.warning(f"Graph entity search failed for '{entity}': {e}")
        return results

    def _calculate_text_relevance(self, query: str, text: str) -> float:
        """Simple text relevance calculation"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())

        if not query_words:
            return 0.0

        intersection = query_words.intersection(text_words)
        return len(intersection) / len(query_words)

    # File extensions and version-like patterns that should NOT be treated as sentence endings
    _EXTENSION_RE = re.compile(
        r'\.'
        r'(?:php|js|jsx|ts|tsx|py|json|yaml|yml|toml|xml|html|css|scss|md|txt|sh|bat|'
        r'env|lock|sql|csv|log|conf|cfg|ini|vue|svelte|rb|rs|go|java|kt|swift|c|cpp|h|'
        r'hpp|cs|fs|ex|exs|erl|hs|ml|r|jl|dart|lua|pl|pm|ipynb)'
        r'(?=[\s,;:)\]}\-/"\']|$)',
        re.IGNORECASE
    )
    _VERSION_RE = re.compile(r'(?:v?\d+)\.\d+(?:\.\d+)*(?:\+|-\w+)*')
    _PATH_RE = re.compile(r'[\w/\\]+\.[\w]+')
    _PLACEHOLDER = '\x00DOT\x00'

    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text, protecting file extensions and versions."""
        # 1. Protect file extensions, version numbers, and paths from sentence splitting
        protected = text
        replacements = []

        # Protect version numbers first (e.g., "v3.3", "PHP 8.2+", "1.17.0")
        for m in self._VERSION_RE.finditer(protected):
            replacements.append(m.group())
        for r in replacements:
            protected = protected.replace(r, r.replace('.', self._PLACEHOLDER))

        # Protect file extensions (e.g., ".php", ".js")
        protected = self._EXTENSION_RE.sub(
            lambda m: m.group().replace('.', self._PLACEHOLDER), protected
        )

        # 2. Split on multiple boundary types:
        #    - Newlines (handles bullets, headers, structured docs)
        #    - Sentence endings (. ! ?) followed by space
        #    - Bullet markers (* -)
        # First split on newlines
        lines = re.split(r'\n+', protected)
        # Then split each line on sentence boundaries
        raw_segments = []
        for line in lines:
            segments = re.split(r'(?<=[.!?])\s+', line)
            raw_segments.extend(segments)

        # 3. Restore protected dots and validate
        facts = []
        for s in raw_segments:
            restored = s.replace(self._PLACEHOLDER, '.').strip()
            # Strip trailing punctuation for cleaner facts
            restored = restored.rstrip('.!? ')
            # Strip leading bullet markers and numbering
            restored = re.sub(r'^[\*\-\u2022]+\s*', '', restored).strip()
            restored = re.sub(r'^\d+[\.\)]\s*', '', restored).strip()
            if not restored or len(restored) < 35:
                continue
            # Reject fragments starting with orphaned extensions/lowercase
            if re.match(r'^[a-z]{1,4}[)\]\s,]', restored):
                continue
            # Reject section headers and labels (e.g., "Pre-Production (Months 3-4):")
            if restored.endswith(':') and len(restored) < 80:
                continue
            # Reject pure labels/titles (short, no verb-like content)
            if len(restored) < 60 and not any(c in restored for c in '.,$'):
                words = restored.split()
                # If most words are capitalized, it's a header
                cap_words = sum(1 for w in words if w[0].isupper()) if words else 0
                if cap_words >= len(words) * 0.7:
                    continue
            facts.append(restored)

        return facts

    # Pattern to find capitalized multi-word phrases (project/person names)
    # Matches: "Echoes of the Fallen", "Dead Cells", "Voxel Plugin Legacy"
    _TOPIC_RE = re.compile(
        r'\b([A-Z][a-z]+'
        r'(?:\s+(?:of|the|and|in|on|for|to|with|a|an)\b)*'
        r'(?:\s+[A-Z][a-z]+'
        r'(?:\s+(?:of|the|and|in|on|for|to|with|a|an)\b)*)*)'
    )
    # Starters that indicate orphaned facts (no clear subject)
    _ORPHAN_STARTERS_RE = re.compile(
        r'^(?:Total|The|A|An|It|This|That|Each|Every|All|'
        r'Primary|Secondary|Core|Main|Basic|Full|'
        r'Budget|Phase|Risk|Strategy|Implementation|'
        r'\d)',
        re.IGNORECASE
    )

    # Prepositions that should not appear at the end of a topic name
    _TRAILING_PREP_RE = re.compile(
        r'(?:\s+(?:of|the|and|in|on|for|to|with|a|an))+$', re.IGNORECASE
    )
    # Technology/tool compound names that should NOT be used as document topics
    _TECH_NAME_BLOCKLIST = frozenset([
        'docker compose', 'docker desktop', 'docker swarm', 'docker engine',
        'visual studio', 'vs code', 'android studio', 'xcode',
        'unreal engine', 'unity engine', 'godot engine', 'game maker',
        'laravel forge', 'ruby rails',
        'google cloud', 'amazon web', 'microsoft azure',
        'open source', 'pull request', 'merge request',
        'node modules', 'next js', 'nuxt js',
    ])

    def _inject_context(self, facts: List[str], full_text: str) -> List[str]:
        """Prepend topic context to orphaned facts that lack a clear subject.

        Detects the main topic (e.g., project name) from the text header,
        then enriches facts that don't mention it or any proper noun.
        Zero LLM cost — pure heuristic.

        Only activates for longer texts (>200 chars) where facts are likely
        to be orphaned. Short texts (personal info, preferences) are already
        self-contained and don't need context injection.
        """
        # Short texts don't need context injection — they're self-contained.
        # Multi-sentence personal info (under ~300 chars) is still self-contained.
        # Context injection is for long documents where facts lose their parent topic.
        if len(full_text) < 300:
            return facts

        # Step 1: detect topic from the opening of the text only.
        # A document topic appears near the start: "Echoes of the Fallen is a..."
        # Scanning further picks up list items (e.g., "Dead Cells" in a game list).
        # Use the shorter of: first sentence/line or first 100 chars.
        first_break = len(full_text)
        for sep in ['\n', '. ', '! ', '? ']:
            pos = full_text.find(sep)
            if 0 < pos < first_break:
                first_break = pos
        header = full_text[:min(first_break + 1, 80)]
        topic = None
        for match in self._TOPIC_RE.finditer(header):
            candidate = match.group(1).strip()
            # Strip trailing prepositions (e.g., "Docker on Windows with" → "Docker on Windows")
            candidate = self._TRAILING_PREP_RE.sub('', candidate).strip()
            words = candidate.split()
            # Must be 2+ words, not a common sentence opener, not a tech name
            if len(words) >= 2 and not candidate.startswith(('The ', 'A ', 'An ', 'This ', 'That ')):
                if candidate.lower() not in self._TECH_NAME_BLOCKLIST:
                    topic = candidate
                    break

        if not topic:
            return facts

        topic_lower = topic.lower()

        enriched = []
        for fact in facts:
            fact_lower = fact.lower()
            # Already contains the topic?
            if topic_lower in fact_lower:
                enriched.append(fact)
                continue
            # Contains a multi-word proper noun (e.g., "Unreal Engine", "Dark Souls")?
            # These facts already have their own context — no need to inject.
            # This is generic: works for any user, not just hardcoded names.
            has_proper_name = any(
                len(m.group(1).split()) >= 2
                for m in self._TOPIC_RE.finditer(fact)
            )
            if has_proper_name:
                enriched.append(fact)
                continue
            # Fact lacks context — prepend topic
            enriched.append(f"{topic}: {fact}")
        return enriched

    def _find_novel_facts(self, new_facts: List[str], existing_facts: List[str], user_id: str = None) -> List[str]:
        """Identify facts that are truly new using Qdrant vector similarity"""
        if not self.memory_client:
            return new_facts

        novel = []
        for new_fact in new_facts:
            try:
                # Check directly against Qdrant for semantic duplicates
                emb = self.memory_client.embedding_model.embed(new_fact, "search")
                filters = {"user_id": user_id} if user_id else {}
                hits = self.memory_client.vector_store.search(
                    query=new_fact, vectors=emb, limit=1, filters=filters
                )
                best_score = hits[0].score if hits else 0
                if best_score >= 0.85:
                    logging.info(f"Dedup (enhanced): skip '{new_fact[:60]}' (score={best_score:.3f})")
                    continue
            except Exception as e:
                logging.warning(f"Vector dedup failed, keeping fact: {e}")
            novel.append(new_fact)
        return novel

    def _extract_memorable_content(self, user_message: str, llm_response: str) -> List[str]:
        """Extract content worth remembering from conversation.

        User messages: split on sentences, keep everything (even short preferences).
        LLM responses: use _extract_facts (stricter), skip filler/meta lines.
        """
        memorable = []

        # Extract from user message — use lightweight splitting (no min length)
        # User preferences are often short: "I prefer dark mode", "I use vim"
        for line in re.split(r'[.\n]+', user_message):
            line = line.strip()
            if len(line) >= 8:  # Very low threshold — only skip tiny fragments like "ok" "yes"
                memorable.append(line)

        # Extract from LLM response — stricter, filter filler
        llm_facts = self._extract_facts(llm_response)
        _FILLER_STARTS = (
            'sure', 'okay', 'alright', 'great', 'yes', 'no', 'i can', "i'll",
            'let me', 'here is', 'here are', 'as you', 'based on', 'in summary',
            'to summarize', 'in conclusion', 'as mentioned', 'as i said',
            'feel free', 'don\'t hesitate', 'hope this', 'glad to',
        )
        for fact in llm_facts:
            fact_lower = fact.lower().strip()
            if fact_lower.startswith(_FILLER_STARTS):
                continue
            if len(fact) < 40:
                continue
            memorable.append(fact)

        return memorable

    def _generate_memory_insights(self, related_memories: List[MemorySearchResult], new_content: List[str]) -> List[str]:
        """Generate insights from memory patterns"""
        insights = []

        if len(related_memories) > 5:
            insights.append(f"Found {len(related_memories)} related memories - this appears to be an ongoing topic of interest")

        # Analyze sources
        sources = [m.source for m in related_memories]
        if sources.count('temporal') > sources.count('vector'):
            insights.append("Recent conversations show continued interest in this topic")

        if sources.count('graph') > 0:
            insights.append("This topic has established relationships with other concepts in memory")

        return insights

    def _identify_conversation_patterns(self, user_message: str, llm_response: str, related_memories: List[MemorySearchResult]) -> List[str]:
        """Identify patterns in conversation flow"""
        patterns = []

        if len(related_memories) > 0:
            patterns.append("Building on previous conversations")

        if any(word in user_message.lower() for word in ['remember', 'recall', 'before', 'previous']):
            patterns.append("User referencing past information")

        if any(word in llm_response.lower() for word in ['as we discussed', 'previously', 'remember']):
            patterns.append("LLM using memory context")

        return patterns

    def _deduplicate_and_rank(self, results: List[MemorySearchResult], limit: int) -> List[MemorySearchResult]:
        """Remove duplicates, interleave sources, and rank by relevance."""
        # Deduplication by ID and content
        seen_ids = set()
        seen_content = set()
        unique_results = []

        for result in results:
            content_key = result.content.lower().strip()[:60]
            if result.id not in seen_ids and content_key not in seen_content:
                seen_ids.add(result.id)
                seen_content.add(content_key)
                unique_results.append(result)

        # Sort by score within each source
        vector_results = sorted([r for r in unique_results if r.source == 'vector'],
                                key=lambda x: x.score, reverse=True)
        graph_results = sorted([r for r in unique_results if r.source == 'graph'],
                               key=lambda x: x.score, reverse=True)
        temporal_results = sorted([r for r in unique_results if r.source == 'temporal'],
                                  key=lambda x: x.score, reverse=True)

        # Interleave: prioritize vector (most informative), then graph, then temporal
        # Take up to 60% vector, 30% graph, 10% temporal
        max_vector = max(1, int(limit * 0.6))
        max_graph = max(1, int(limit * 0.3))
        max_temporal = max(1, int(limit * 0.1))

        merged = []
        merged.extend(vector_results[:max_vector])
        merged.extend(graph_results[:max_graph])
        merged.extend(temporal_results[:max_temporal])

        # If we haven't hit the limit, fill with remaining results
        remaining = [r for r in unique_results if r not in merged]
        remaining.sort(key=lambda x: x.score, reverse=True)
        merged.extend(remaining[:limit - len(merged)])

        # Final sort by score
        merged.sort(key=lambda x: x.score, reverse=True)

        return merged[:limit]


# Global instance
enhanced_memory_manager = EnhancedMemoryManager()
