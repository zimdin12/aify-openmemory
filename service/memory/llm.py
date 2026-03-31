"""
Pluggable LLM backend — resolves inference URLs from external config or container manager.

Three modes (set via LLM_MODE env var):
- external: LLM_API_URL points to an external service (default)
- container: orchestrator manages an LLM sub-container, URL resolved at runtime
- openai: standard OpenAI API with real API key

Same pattern for embeddings (EMBEDDER_MODE).
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


def _strip_json_fences(text: str) -> str:
    """Strip markdown code fences and think blocks from JSON output."""
    text = text.strip()
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    if text.startswith('```'):
        first_nl = text.find('\n')
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith('```'):
            text = text.rstrip()[:-3].rstrip()
    return text


class LLMBackend:
    """Pluggable LLM backend with container-aware URL resolution."""

    def __init__(self, container_manager=None):
        self.container_manager = container_manager
        self.llm_mode = os.environ.get('LLM_MODE', 'external')
        self.embedder_mode = os.environ.get('EMBEDDER_MODE', 'external')

    def get_llm_url(self) -> str:
        """Resolve LLM API URL based on mode."""
        if self.llm_mode == 'container':
            name = os.environ.get('LLM_CONTAINER_NAME', 'inference-llm')
            if self.container_manager:
                url = self.container_manager.resolve_url(name)
                if url:
                    return url
            raise RuntimeError(f"LLM container '{name}' not available")

        url = os.environ.get('LLM_API_URL', os.environ.get('OLLAMA_BASE_URL', ''))
        if not url:
            raise RuntimeError("LLM_API_URL must be set (or use LLM_MODE=container)")
        return url

    def get_embedder_url(self) -> str:
        """Resolve embedder API URL based on mode."""
        if self.embedder_mode == 'container':
            name = os.environ.get('EMBEDDER_CONTAINER_NAME', 'inference-embed')
            if self.container_manager:
                url = self.container_manager.resolve_url(name)
                if url:
                    return url
            raise RuntimeError(f"Embedder container '{name}' not available")

        url = os.environ.get('EMBEDDER_API_URL', os.environ.get('OLLAMA_BASE_URL', ''))
        if not url:
            raise RuntimeError("EMBEDDER_API_URL must be set")
        return url

    def chat(self, messages: List[Dict], model: str = None,
             json_mode: bool = True, options: Dict = None,
             timeout: int = 120) -> str:
        """
        Call LLM chat API. Supports both OpenAI-compatible and Ollama native APIs.

        Args:
            messages: List of {role, content} dicts.
            model: Model name (default from LLM_MODEL env var).
            json_mode: Request JSON output format.
            options: Sampling options (temperature, top_p, etc.).
            timeout: Request timeout in seconds.

        Returns:
            Content string from the LLM response.
        """
        if model is None:
            model = os.environ.get('LLM_MODEL', 'qwen3.5:4b')

        api_url = self.get_llm_url()
        use_openai = os.environ.get('LLM_PROVIDER', 'openai') != 'ollama'

        if use_openai:
            return self._chat_openai(api_url, messages, model, json_mode, options, timeout)
        else:
            return self._chat_ollama(api_url, messages, model, json_mode, options, timeout)

    def _chat_openai(self, api_url: str, messages: List[Dict], model: str,
                     json_mode: bool, options: Dict, timeout: int) -> str:
        """OpenAI-compatible API call (llama.cpp, vLLM, SGLang, Ollama /v1, etc.)."""
        url = api_url.rstrip('/') + '/v1/chat/completions'
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if options:
            if 'temperature' in options:
                payload['temperature'] = options['temperature']
            if 'top_p' in options:
                payload['top_p'] = options['top_p']
            if 'num_predict' in options:
                payload['max_tokens'] = options['num_predict']

        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM API error: HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        return data['choices'][0]['message']['content']

    def _chat_ollama(self, api_url: str, messages: List[Dict], model: str,
                     json_mode: bool, options: Dict, timeout: int) -> str:
        """Ollama native API call."""
        url = api_url.rstrip('/') + '/api/chat'
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
        }
        if json_mode:
            payload["format"] = "json"
        if options:
            payload["options"] = options

        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama API error: HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        msg = data.get('message', {})
        if isinstance(msg, str):
            return msg
        elif isinstance(msg, dict):
            return msg.get('content', '')
        return ''


# Module-level singleton (initialized lazily with container manager)
_backend: Optional[LLMBackend] = None


def get_llm_backend() -> LLMBackend:
    """Get the global LLM backend instance."""
    global _backend
    if _backend is None:
        _backend = LLMBackend()
    return _backend


def init_llm_backend(container_manager=None):
    """Initialize the LLM backend with an optional container manager."""
    global _backend
    _backend = LLMBackend(container_manager=container_manager)
    return _backend


def llm_chat(messages, model=None, json_mode=True, options=None, timeout=120):
    """Module-level convenience function (backwards compatible)."""
    return get_llm_backend().chat(messages, model, json_mode, options, timeout)
