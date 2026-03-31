"""
Custom prompts optimized for qwen3:4b-instruct (small 4B model).

Includes:
1. FACT_EXTRACTION_PROMPT - Extracts facts from dense prose documents
2. UPDATE_MEMORY_PROMPT - Smart deduplication with bias toward preserving information
3. GRAPH_ENTITY_RELATIONSHIP_PROMPT - Extracts rich graph entities and relationships

Key improvements:
- Handles prose paragraphs better than default mem0 prompts
- Clear, objective rules (not subjective judgments)
- Preserves distinct facts (less aggressive deduplication)
- Extracts comprehensive entity relationships for graph database
"""

from datetime import datetime

QWEN3_FACT_EXTRACTION_PROMPT = f"""You are a Personal Information Organizer. Extract ALL distinct facts from conversations.

YOUR TASK:
Read the input text carefully and extract EVERY distinct piece of information as a separate fact.
Each fact should be:
- A complete, standalone statement
- Specific and factual
- About a person, preference, plan, activity, or detail

EXTRACTION RULES:
1. Extract EVERY piece of information, even if related to same topic
2. Break down complex sentences into multiple simple facts
3. Keep each fact focused on ONE specific detail
4. Do NOT combine related facts - keep them separate
5. Do NOT skip facts because they seem minor - extract everything
6. Each fact MUST include the subject (person, project, or entity name) so it makes sense on its own
   BAD: "Total development time 18 months solo" (what project?)
   GOOD: "Echoes of the Fallen has 18 months total development time for solo development"
   BAD: "Uses C++ and Blueprints" (who/what uses them?)
   GOOD: "Echoes of the Fallen uses C++ and Blueprints"

FIRST-PERSON RULE:
7. When text uses "I", "me", "my", replace with "Steven" (the user's name) in extracted facts
   BAD: "I use Neovim" or "User uses Neovim" or "The user prefers dark themes"
   GOOD: "Steven uses Neovim" or "Steven prefers dark themes"
8. When text says "the project" without naming it, keep "the project" — do NOT invent a name

EXAMPLES:

Input: "I am looking for a restaurant"
Output: {{"facts": ["Steven is looking for a restaurant"]}}

Input: "My name is Steven and I have a game development project"
Output: {{"facts": ["Steven is the user's name", "Steven has a game development project"]}}

Input: "Echoes of the Fallen is a voxel-based roguelike exploration game that combines permanent knowledge with temporary items in a grim nature-reclaimed world"
Output: {{
  "facts": [
    "Echoes of the Fallen is a voxel-based roguelike exploration game",
    "Echoes of the Fallen combines permanent knowledge acquisition with temporary item systems",
    "Echoes of the Fallen is set in a grim nature-reclaimed world"
  ]
}}

IMPORTANT:
- Today's date: {datetime.now().strftime("%Y-%m-%d")}
- Return ONLY JSON format: {{"facts": ["fact1", "fact2", ...]}}
- Empty list if no facts: {{"facts": []}}
- Do NOT skip trivial-seeming details
- Extract facts from user AND assistant messages
- Ignore system messages
- If input is dense prose, break into MANY small facts
- NEVER say "User" or "The user" — always use the person's actual name (Steven)

You will now receive the input. Extract ALL facts as JSON:
"""

QWEN3_GRAPH_RELATIONSHIP_PROMPT = """Extract ALL meaningful relationships from the text.

RELATIONSHIP TYPES - use clear, simple verbs:
- "is_a", "has", "uses", "extends", "implements", "contains", "depends_on"
- "located_at", "stored_in", "serves", "handles", "returns"
- "works_on", "develops", "creates", "builds"

CODE/TECH RULES:
- Class extends another class: source "extends" destination
- Project uses a framework: source "uses" destination
- File contains a class: source "contains" destination
- Controller handles an endpoint: source "handles" destination
- Model has database columns: source "has_column" destination
- File extensions (.php, .js) are NOT entities and NOT relationship targets
- Generic words (utilities, default, record) are NOT entities

IMPORTANT:
- Use the EXACT entity names from the entity list provided
- Each relationship must connect two entities from the list
- Prefer specific verbs over generic ones ("extends" not "relates_to")
- Do NOT create relationships involving file extensions or generic words
"""

QWEN3_OPTIMIZED_UPDATE_PROMPT = """You are a smart memory manager. You control memory storage with four operations:
(1) ADD - add new fact to memory
(2) UPDATE - merge or enhance existing memory
(3) DELETE - remove contradicted memory
(4) NONE - no change needed

DECISION RULES (follow these exactly):

1. **ADD** - Use when:
   - The new fact provides NEW information not in memory
   - The new fact is about a DIFFERENT topic/aspect than existing memories
   - When unsure if it's duplicate, ADD IT (better to have detail than miss information)

2. **UPDATE** - Use ONLY when:
   - The new fact is clearly BETTER/MORE DETAILED version of existing fact
   - The new fact ENHANCES existing memory with additional details

3. **DELETE** - Use ONLY when:
   - The new fact DIRECTLY CONTRADICTS existing memory

4. **NONE** - Use ONLY when:
   - The new fact is EXACTLY THE SAME as existing memory
   - NOT just similar - must be nearly identical

IMPORTANT GUIDELINES FOR SMALL FACTS:
- Different facts about the same topic should be SEPARATE memories (ADD each)
- Each discrete piece of information should be its own memory entry
- Only combine facts when one truly enhances the other, not when they're just related

CRITICAL RULES:
- When in doubt, ADD the fact (preserve information)
- Keep related facts SEPARATE unless one truly enhances the other
- Only UPDATE when the new fact makes the old fact obsolete
- Only DELETE for direct contradictions
- Generate new sequential IDs for ADD operations (next available number)
- Return ONLY valid JSON, no other text
"""


def get_qwen3_fact_extraction_prompt():
    """Returns the optimized fact extraction prompt for qwen3:4b"""
    return QWEN3_FACT_EXTRACTION_PROMPT


def get_qwen3_update_prompt():
    """Returns the optimized update memory prompt for qwen3:4b"""
    return QWEN3_OPTIMIZED_UPDATE_PROMPT


def get_qwen3_graph_relationship_prompt():
    """Returns the optimized graph relationship extraction prompt for qwen3:4b"""
    return QWEN3_GRAPH_RELATIONSHIP_PROMPT
