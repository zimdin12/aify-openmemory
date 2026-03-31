"""
JSON-based graph extraction (replaces unreliable tool calling).

Uses a single LLM call with JSON mode to extract both entities and
relationships. Supports both Ollama and OpenAI-compatible APIs (llama.cpp, etc.)
via LLM_API_URL env var.

Supports model detection: adjusts prompts and sampling params per model family.
"""

import json
import os
import re
import logging
from mem0.memory.graph_memory import MemoryGraph

from service.memory.llm import llm_chat, _strip_json_fences
from service.memory.model_configs import detect_model_family, get_llm_options

# Backwards-compatible aliases used by other files (enhanced_memory.py, benchmarks)
_detect_model_family = detect_model_family
_get_llm_options = get_llm_options

logger = logging.getLogger(__name__)

# Save originals for potential restore
_original_add = MemoryGraph.add
_original_retrieve_nodes = MemoryGraph._retrieve_nodes_from_data
_original_establish_relations = MemoryGraph._establish_nodes_relations_from_data

# --- Filters (code-level safety nets, not LLM guidance) ---

# Entity names that are never meaningful graph nodes
_ENTITY_BLOCKLIST = frozenset([
    # File extensions (often extracted as entities from code discussions)
    'php', 'js', 'jsx', 'ts', 'tsx', 'py', 'json', 'yaml', 'yml', 'css', 'html',
    'xml', 'md', 'txt', 'sh', 'bat', 'sql', 'csv', 'log', 'conf', 'cfg', 'ini',
    # Generic filler words
    'utilities', 'default', 'record', 'entity', 'unknown', 'other', 'none', 'null',
    'true', 'false', 'yes', 'no', 'n/a', 'etc',
    # Config/infra noise
    'metadata', 'project', 'timezone', 'port', 'api', 'url', 'host',
])

# Number/amount pattern — these should be facts in vector store, not graph nodes
_NUMBER_PATTERN = re.compile(
    r'^[\d$€£¥,.%]+$'                    # pure numbers: 60, $5000, 75%
    r'|^\d+[\w_]*$'                       # number-prefixed: 60fps, 4k, 10gbe
    r'|^\d+[_-]\w+'                       # number-dash-word: 4-player, 18-months
    r'|^\d+:\d+'                          # ratios: 2:3
    r'|\d+_(months?|years?|days?|hours?|minutes?|seconds?|euros?|dollars?|fps|gb|mb|tb|gbe)$'
)

# Entity types to drop — only truly useless categories
_BLOCKED_ENTITY_TYPES = frozenset([
    'concept',      # too vague — if it's not specific enough to type, skip it
    'metric',       # numbers belong in vector store, not graph
])

# Types that can be relationship sources (hub nodes that connect to everything else)
# Includes game_element because qwen3.5 often types game projects as game_element
_HUB_TYPES = frozenset(['person', 'project', 'game_element'])

# Map for flipping relationship direction when swapping source/target
_RELATION_FLIP = {
    'preferred_by': 'prefers', 'used_by': 'uses', 'owned_by': 'owns',
    'developed_by': 'develops', 'created_by': 'creates', 'built_by': 'builds',
    'played_by': 'plays', 'spoken_by': 'speaks', 'practiced_by': 'practices',
    'has_user': 'uses', 'has_developer': 'develops', 'has_member': 'member_of',
    'employs': 'works_at', 'teaches': 'mentored_by', 'mentors': 'mentored_by',
    'contains': 'part_of', 'includes': 'part_of',
}

def _flip_relation(relation: str) -> str:
    """Flip a relationship for hub-source normalization."""
    if relation in _RELATION_FLIP:
        return _RELATION_FLIP[relation]
    if relation.endswith('_by'):
        base = relation[:-3]
        return base + 'es' if base.endswith(('s', 'sh', 'ch')) else base + 's'
    return 'has'

# --- Prompt ---

GRAPH_EXTRACTION_PROMPT = """Extract entities and relationships from text. Return ONLY a JSON object.

Output format:
{"entities": [{"name": "...", "type": "..."}, ...], "relationships": [{"source": "...", "relation": "...", "target": "..."}, ...]}

PURPOSE: Build a knowledge graph for navigating a person's world — their projects, tools, people, interests. Extract only entities worth revisiting later. Fewer high-quality entities is better than many questionable ones.

ENTITY TYPES — use whatever type best describes the entity. Common types include:
person, project, technology, framework, tool, hardware, organization, place, skill, preference, hobby, game_element, database, service, protocol, genre

You are free to use other types if they fit better. Just be specific — avoid vague types like "concept" or "thing".

ENTITY QUALITY RULES:
- Names should be 1-4 words, lowercase
- Each entity should be something a person would want to look up or navigate to later
- IMPORTANT: When text uses "I", "me", "my" — create a "USER_ID" entity (type: person) and use it as the relationship source
- If context mentions a specific project by name, connect related features/tech to THAT project, not to the person

DO NOT create entities from:
- Numbers, amounts, budgets, durations, percentages, ratios (store these as facts, not graph nodes)
- Development phases or milestones (phase 1, alpha, sprint 3)
- Quality levels or settings (full detail, low, medium, high, ultra)
- URLs, paths, hostnames, env variables, config values
- Generic words (utilities, default, metadata, mode)
- Abstract descriptions of actions (e.g. "plain text", "output consistency", "number of calls")

RELATIONSHIP RULES:
Only "person", "project", and "game_element" should be relationship SOURCES. Everything else is a target.
- person -> develops/builds -> project
- person -> uses/prefers -> tool/technology/framework
- person -> has_preference -> preference
- person -> has_skill/learning -> skill/technology
- person -> has_hobby -> hobby
- person -> owns -> hardware
- person -> knows -> person
- person -> works_at -> organization
- person -> lives_in -> place
- project -> built_with -> technology/framework
- project -> features/uses -> game_element/technology/tool
- When a tool/technology connects to another tool/technology, route through the project instead
  BAD: "voxel_plugin -> leverages -> nanite"  GOOD: "my_project -> uses -> nanite"
- "built_with" and "features" should come FROM the project, never from a person
  BAD: "steven -> built_with -> godot"  GOOD: "echoes_of_the_fallen -> built_with -> godot"
  A person "develops" or "builds" a project. A project is "built_with" a technology.

When text says "the game" or "the project", use the actual project name from context instead.

EXAMPLE 1:
Text: "Steven is a PHP developer building Echoes of the Fallen, a voxel roguelike. He prefers dark mode and uses vim. Alex works at Google. The game uses C++ and dual contouring. Steven also enjoys cooking."
Output:
{"entities": [
  {"name": "steven", "type": "person"},
  {"name": "echoes of the fallen", "type": "project"},
  {"name": "c++", "type": "technology"},
  {"name": "dual contouring", "type": "technology"},
  {"name": "dark mode", "type": "preference"},
  {"name": "vim", "type": "tool"},
  {"name": "alex", "type": "person"},
  {"name": "google", "type": "organization"},
  {"name": "php development", "type": "skill"},
  {"name": "cooking", "type": "hobby"}
], "relationships": [
  {"source": "steven", "relation": "develops", "target": "echoes of the fallen"},
  {"source": "steven", "relation": "has_skill", "target": "php development"},
  {"source": "steven", "relation": "has_preference", "target": "dark mode"},
  {"source": "steven", "relation": "uses", "target": "vim"},
  {"source": "steven", "relation": "knows", "target": "alex"},
  {"source": "steven", "relation": "has_hobby", "target": "cooking"},
  {"source": "alex", "relation": "works_at", "target": "google"},
  {"source": "echoes of the fallen", "relation": "built_with", "target": "c++"},
  {"source": "echoes of the fallen", "relation": "features", "target": "dual contouring"}
]}

EXAMPLE 2 (first person — note I/me replaced with user name):
Text: "I replaced the tool-calling pipeline with JSON mode. I use Neovim and prefer dark themes."
Output:
{"entities": [
  {"name": "USER_ID", "type": "person"},
  {"name": "json mode", "type": "technology"},
  {"name": "neovim", "type": "tool"},
  {"name": "dark themes", "type": "preference"}
], "relationships": [
  {"source": "USER_ID", "relation": "uses", "target": "json mode"},
  {"source": "USER_ID", "relation": "uses", "target": "neovim"},
  {"source": "USER_ID", "relation": "has_preference", "target": "dark themes"}
]}"""


def _json_extract_graph(self, data, filters, context=None):
    """
    Extract entities and relationships using JSON mode (single LLM call).
    Returns (entity_type_map, relationships).

    Args:
        context: Optional list of recent conversation turns [{role, content}, ...]
                 to help the LLM understand multi-topic text.
    """
    user_id = filters.get('user_id', 'user')
    prompt = GRAPH_EXTRACTION_PROMPT.replace("USER_ID", user_id)

    try:
        model = os.environ.get('LLM_MODEL', 'qwen3.5-9b')
        family = _detect_model_family(model)
        options = _get_llm_options(family)

        # Universal extraction emphasis — helps all models produce complete graphs
        prompt += (
            "\n\nIMPORTANT: You MUST generate relationships for every entity. "
            "Every entity should connect to at least one person or project via a relationship. "
            "If an entity has no relationship, remove it from entities too."
            "\nPreserve NEGATIONS: does_not_use, stopped_using, rejected, never_used (NOT just 'uses')."
            "\nPreserve TEMPORAL state: formerly_worked_at vs currently_works_at, previously_used vs uses."
            f"\nALWAYS include a \"{user_id}\" (type: person) entity. If text says \"the project\" or "
            f"\"my project\", create \"{user_id}\" as source with relation \"uses\" to all technologies mentioned."
            "\nNEVER use 'built_with' or 'features' with a person as source. "
            "Only a project can be 'built_with' something. A person 'develops' or 'uses' things."
        )

        # Build user message with optional conversation context
        user_content = data
        if context:
            context_lines = []
            for turn in context[-6:]:  # last 3 turns max
                role = turn.get('role', 'user')
                content = turn.get('content', '')[:400]
                context_lines.append(f"[{role}]: {content}")
            if context_lines:
                user_content = (
                    "CONVERSATION CONTEXT (for understanding topics being discussed):\n"
                    + "\n".join(context_lines)
                    + "\n\nTEXT TO EXTRACT FROM:\n"
                    + data
                )

        logger.info(f"Graph extraction using {model} (family={family})")
        content = llm_chat(
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': user_content},
            ],
            model=model,
            json_mode=True,
            options=options,
            timeout=120,
        )
        if not content:
            logger.warning("Graph JSON extraction returned empty content")
            return {}, []

        parsed = json.loads(_strip_json_fences(content))

        # --- I/me/my → user_id normalization ---
        _SELF_REFS = frozenset(['i', 'me', 'my', 'myself', 'user', 'the_user', 'the_subject', 'subject'])
        user_id_key = user_id.lower().replace(' ', '_')

        def _normalize_self_ref(name_key):
            """Replace first-person references with the actual user_id."""
            if name_key in _SELF_REFS:
                return user_id_key
            return name_key

        # --- Build entity_type_map with safety filters ---
        entity_type_map = {}
        raw_entities = parsed.get('entities', [])
        for ent in raw_entities:
            if isinstance(ent, str):
                name = ent.strip()
                etype = 'concept'
            elif isinstance(ent, dict):
                name = ent.get('name', '').strip()
                etype = ent.get('type', 'concept').strip()
            else:
                continue
            if not name:
                continue

            # Normalize
            name_key = name.lower().replace(' ', '_')
            name_key = _normalize_self_ref(name_key)
            etype_key = etype.lower().replace(' ', '_')
            # If self-ref was resolved, force type to person
            if name_key == user_id_key:
                etype_key = 'person'

            # Hard filters (structural, not domain-specific)
            if name_key in _ENTITY_BLOCKLIST:
                continue
            if len(name_key) < 2:
                continue
            if len(name_key) > 60:
                logger.info(f"Skipping overly long entity: {name_key[:40]}...")
                continue
            if any(c in name_key for c in ('http://', 'https://', '=', '://', '.internal')):
                logger.info(f"Skipping URL/config entity: {name_key}")
                continue
            if _NUMBER_PATTERN.match(name_key):
                logger.info(f"Skipping number/amount entity: {name_key}")
                continue
            if etype_key in _BLOCKED_ENTITY_TYPES:
                logger.info(f"Skipping entity with blocked type: {name_key} ({etype_key})")
                continue

            # Accept whatever type the LLM generated
            entity_type_map[name_key] = etype_key

        # --- Build relationships with structural validation ---
        relationships = []
        raw_rels = parsed.get('relationships', [])
        for rel in raw_rels:
            if not isinstance(rel, dict):
                continue
            source = _normalize_self_ref(str(rel.get('source', '')).strip().lower().replace(' ', '_'))
            target = _normalize_self_ref(str(rel.get('target', '')).strip().lower().replace(' ', '_'))
            relation = str(rel.get('relation', '')).strip().lower().replace(' ', '_')

            if not source or not target or not relation:
                continue
            # Self-reference checks
            if source == target:
                continue
            if len(source) > 2 and len(target) > 2 and (source in target or target in source):
                logger.info(f"Skipping fuzzy self-ref: {source} -> {target}")
                continue
            # Hard filters on entity names
            if source in _ENTITY_BLOCKLIST or target in _ENTITY_BLOCKLIST:
                continue
            if _NUMBER_PATTERN.match(source) or _NUMBER_PATTERN.match(target):
                logger.info(f"Skipping relationship with number entity: {source} -> {target}")
                continue

            # Auto-add user_id as person if used in relationship (LLM often omits from entities)
            if source not in entity_type_map:
                if source == user_id_key:
                    entity_type_map[source] = 'person'
                else:
                    logger.info(f"Skipping relationship with unknown source: {source}")
                    continue
            if target not in entity_type_map:
                if target == user_id_key:
                    entity_type_map[target] = 'person'
                else:
                    logger.info(f"Skipping relationship with unknown target: {target}")
                    continue

            # Prefer hub types (person, project) as relationship sources
            source_type = entity_type_map.get(source, '')
            target_type = entity_type_map.get(target, '')
            if source_type not in _HUB_TYPES:
                if target_type in _HUB_TYPES:
                    # Flip: tool --preferred_by--> person => person --uses--> tool
                    source, target = target, source
                    source_type, target_type = target_type, source_type
                    relation = _flip_relation(relation)
                    logger.info(f"Flipped non-hub source: {target} -> {source} --{relation}--> {target}")
                else:
                    logger.info(f"Skipping relationship with non-hub source: {source} ({source_type}) -> {target}")
                    continue

            # Fix misattributed built_with/features: person can't "build with" a tech
            if source_type == 'person' and relation in ('built_with', 'features'):
                relation = 'uses'
                logger.info(f"Fixed relation: {source} --built_with--> {target} => --uses-->")

            relationships.append({
                'source': source,
                'relationship': relation,
                'destination': target,
            })

        logger.info(f"Graph JSON extraction: {len(entity_type_map)} entities, {len(relationships)} relationships")
        return entity_type_map, relationships

    except json.JSONDecodeError as e:
        logger.warning(f"Graph JSON parse error: {e}")
        return {}, []
    except Exception as e:
        logger.warning(f"Graph JSON extraction error: {e}")
        return {}, []


def _patched_add(self, data, filters):
    """
    Replacement for MemoryGraph.add() that uses JSON extraction.
    Skips the unreliable tool-calling-based pipeline entirely.
    """
    # Extract context from filters (transient field, not persisted)
    context = filters.pop('_session_context', None)

    # Step 1: Extract entities and relationships in one JSON call
    entity_type_map, to_be_added = _json_extract_graph(self, data, filters, context=context)

    if not entity_type_map and not to_be_added:
        logger.info("Graph extraction returned nothing, skipping graph update")
        return {"deleted_entities": [], "added_entities": []}

    # Step 2: Search for existing similar nodes (uses embeddings, no LLM)
    search_output = self._search_graph_db(
        node_list=list(entity_type_map.keys()), filters=filters
    )

    # Step 3: Skip deletion step (unreliable with small LLMs, and risky)
    # The old pipeline used an LLM call to decide what to delete — too error-prone.
    # We only add/update, never delete from graph via extraction.

    # Step 4: Add entities and relationships to Neo4j
    added_entities = self._add_entities(to_be_added, filters, entity_type_map)

    return {"deleted_entities": [], "added_entities": added_entities}


def _patched_retrieve_nodes(self, data, filters):
    """Stub — extraction happens in _patched_add via JSON."""
    return {}


def _patched_establish_relations(self, data, filters, entity_type_map):
    """Stub — extraction happens in _patched_add via JSON."""
    return []


def apply_patch():
    """Apply JSON-based graph extraction patch for qwen3:4b."""
    MemoryGraph.add = _patched_add
    MemoryGraph._retrieve_nodes_from_data = _patched_retrieve_nodes
    MemoryGraph._establish_nodes_relations_from_data = _patched_establish_relations
    print("✓ Applied qwen3:4b entity and relationship extraction fixes for graph extraction")


def remove_patch():
    """Remove the patch and restore original behavior."""
    MemoryGraph.add = _original_add
    MemoryGraph._retrieve_nodes_from_data = _original_retrieve_nodes
    MemoryGraph._establish_nodes_relations_from_data = _original_establish_relations
    print("✓ Removed qwen3:4b graph extraction patch")
