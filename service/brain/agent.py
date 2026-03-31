"""
Memory Brain Agent — autonomous LLM agent for natural-language memory operations.

v2 architecture: single endpoint, single model. The agent determines intent
(search, store, delete, update) from the natural language request and uses
whatever tools are needed.

Uses the same LLM_MODEL as the extraction pipeline (configurable via env var).
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from service.brain.prompts import get_system_prompt
from service.brain.tools import TOOLS, TOOL_DEFINITIONS, _log_audit

logger = logging.getLogger(__name__)


def _strip_json_fences(text: str) -> str:
    """Strip markdown code fences and think blocks from JSON output."""
    import re
    text = text.strip()
    # Strip <think>...</think> blocks (qwen3.5 thinking mode)
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    if text.startswith('```'):
        first_nl = text.find('\n')
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith('```'):
            text = text.rstrip()[:-3].rstrip()
    return text


@dataclass
class BrainResult:
    """Result of a brain agent invocation."""
    answer: str
    steps: int
    tools_called: List[str] = field(default_factory=list)
    user_id: str = ""
    success: bool = True
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


class MemoryBrainAgent:
    """
    LLM agent that sits between callers and the three databases.
    Accepts natural language. Decides which tools to call. Returns synthesized answer.

    Single model, single endpoint. The agent determines intent from the request.
    """

    def __init__(self):
        # Use the same model as extraction pipeline — one model for everything
        self.model = os.environ.get("LLM_MODEL", "qwen3.5-9b")
        self.max_steps = int(os.environ.get("BRAIN_MAX_STEPS", "12"))

    def run(self, request: str, user_id: str) -> BrainResult:
        """
        Run the brain agent loop.

        The agent has full read/write access. It determines from the request
        whether to search, store, delete, or update.

        Args:
            request: Natural language request from the caller.
            user_id: User whose memories to operate on.

        Returns:
            BrainResult with the answer, steps taken, and tools called.
        """
        start = time.time()
        system_prompt = get_system_prompt(user_id)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request},
        ]

        tools_called = []

        for step in range(1, self.max_steps + 1):
            # Call LLM
            try:
                response_text = self._call_llm(messages)
            except Exception as e:
                logger.error(f"Brain step {step}: LLM call failed: {e}")
                return BrainResult(
                    answer=f"Error: LLM call failed: {e}",
                    steps=step,
                    tools_called=tools_called,
                    user_id=user_id,
                    success=False,
                    error=str(e),
                    elapsed_seconds=time.time() - start,
                )

            # Parse JSON response
            action = self._parse_response(response_text, messages, step)
            if action is None:
                return BrainResult(
                    answer="Error: Failed to parse LLM response as JSON after retries.",
                    steps=step,
                    tools_called=tools_called,
                    user_id=user_id,
                    success=False,
                    error="JSON parse failure",
                    elapsed_seconds=time.time() - start,
                )

            # Check if final
            if action.get("final"):
                answer = action.get("answer", "No answer provided.")
                _log_audit("brain_run", user_id,
                           {"request": request[:500], "steps": step, "tools": tools_called},
                           {"answer": answer[:500]})
                return BrainResult(
                    answer=answer,
                    steps=step,
                    tools_called=tools_called,
                    user_id=user_id,
                    success=True,
                    elapsed_seconds=time.time() - start,
                )

            # Execute tool
            tool_name = action.get("action")
            tool_args = action.get("args") or {}

            if not tool_name or tool_name not in TOOLS:
                error_msg = f"Unknown tool: {tool_name}. Available: {', '.join(TOOLS.keys())}"
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": json.dumps({"error": error_msg})})
                continue

            tools_called.append(tool_name)

            # Execute the tool
            try:
                tool_fn = TOOLS[tool_name]
                result = tool_fn(**tool_args)
                result_str = json.dumps(result, default=str)
            except Exception as e:
                logger.warning(f"Brain step {step}: tool {tool_name} failed: {e}")
                result_str = json.dumps({"error": str(e)})

            # Append to conversation
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content": result_str})

        # Reached max steps
        _log_audit("brain_run", user_id,
                   {"request": request[:500], "steps": self.max_steps, "tools": tools_called},
                   {"answer": "Reached step limit"})
        return BrainResult(
            answer="Reached maximum steps without completing. Partial results may be in the tool calls above.",
            steps=self.max_steps,
            tools_called=tools_called,
            user_id=user_id,
            success=False,
            error="max_steps_reached",
            elapsed_seconds=time.time() - start,
        )

    def _call_llm(self, messages: List[Dict]) -> str:
        """Call LLM chat API with JSON mode (supports Ollama and OpenAI-compatible APIs)."""
        from service.memory.llm import llm_chat

        return llm_chat(
            messages=messages,
            model=self.model,
            json_mode=True,
            options={
                "temperature": 0.1,
                "top_p": 0.8,
                "num_predict": 1024,
            },
            timeout=120,
        )

    def _parse_response(self, response_text: str, messages: List[Dict], step: int) -> Optional[Dict]:
        """
        Parse JSON response from the brain LLM.
        Retries up to 2 times if JSON parsing fails.
        """
        max_retries = 2
        current_text = response_text

        for attempt in range(max_retries + 1):
            try:
                cleaned = _strip_json_fences(current_text)
                parsed = json.loads(cleaned)

                if not isinstance(parsed, dict):
                    raise ValueError("Response is not a JSON object")
                if "final" not in parsed and "action" not in parsed:
                    raise ValueError("Missing 'final' or 'action' field")

                return parsed

            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries:
                    logger.warning(f"Brain step {step}, attempt {attempt+1}: JSON parse failed: {e}")
                    messages.append({"role": "assistant", "content": current_text})
                    messages.append({"role": "user", "content": json.dumps({
                        "error": f"Invalid JSON: {str(e)}. You MUST respond with a single valid JSON object matching the schema: {{\"thinking\": \"...\", \"action\": \"tool_name\", \"args\": {{}}, \"final\": false}} or {{\"thinking\": \"...\", \"final\": true, \"answer\": \"...\"}}"
                    })})
                    try:
                        current_text = self._call_llm(messages)
                    except Exception:
                        return None
                else:
                    logger.error(f"Brain step {step}: JSON parse failed after {max_retries} retries")
                    return None

        return None


# Module-level singleton
brain_agent = MemoryBrainAgent()
