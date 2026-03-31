"""
Per-model-family LLM configurations for the extraction pipeline.

To add a new model:
1. Add an entry to MODEL_CONFIGS with the family name as key
2. Set LLM_MODEL env var to the Ollama model name
3. Optionally set LLM_FAMILY env var if auto-detection doesn't match

Only include options you want to OVERRIDE — anything omitted uses the
model's own Ollama default. This is intentional: different models have
different optimal defaults, and we shouldn't force params unnecessarily.

The 'family_patterns' list controls auto-detection from model names.
"""

import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-family configs
# ---------------------------------------------------------------------------
# Only set options that differ from Ollama defaults for this model.
# Common options: temperature, top_p, top_k, presence_penalty, num_predict
#
# Notes on specific params:
# - temperature: lower = more deterministic JSON output, higher = more creative
# - presence_penalty: helps prevent repetition in long outputs
# - top_k: limits token sampling pool
# - num_predict: max output tokens (set per-call, not here)

MODEL_CONFIGS = {
    "qwen3": {
        "family_patterns": ["qwen3"],  # matches qwen3:4b, qwen3:8b, etc.
        "options": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
        },
        "notes": "Reliable for extraction. temp=0.7 works fine, no cross-contamination.",
    },

    "qwen3.5:4b": {
        "family_patterns": ["qwen3.5:4b", "qwen35:4b"],
        "options": {
            "temperature": 0.15,          # Lower temp than 9b — best for 4b multi-person attribution
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 0.5,      # pp=0.5 sweet spot for 4b (jake->devops). pp=1.5 too aggressive.
        },
        "notes": "Full 11-test: t=0.15/pp=0.5 score=135.5 (45f/25r, jake->devops YES). t=0.3 close second.",
    },

    "qwen3.5": {
        "family_patterns": ["qwen3.5", "qwen35"],
        "options": {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
        },
        "notes": "9b default. Full 11-test: t=0.1/pp=1.5 score=134.0 (45f/36r). Original config was best all along.",
    },

    "ministral": {
        "family_patterns": ["ministral"],
        "options": {
            "temperature": 0.05,  # Very low temp = more rels extracted
            "top_p": 0.8,
            "top_k": 40,         # k=40 better than k=20 (score 176.0 vs 158.5)
        },
        "notes": "Full 11-test: t=0.05/k=40 score=176.0 (44f/46r, jake->devops YES). Best overall model.",
    },

    "granite": {
        "family_patterns": ["granite"],
        "options": {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 20,
        },
        "notes": "IBM. Good graph extraction, weak fact splitting. Apache 2.0.",
    },

    "gemma": {
        "family_patterns": ["gemma"],
        "options": {
            # Gemma3 defaults: temp=1.0, top_k=64, top_p=0 (disabled)
            # For extraction we want lower temp for reliable JSON
            "temperature": 0.3,
            "top_k": 64,  # Gemma's default, higher than most models
        },
        "notes": "Google Gemini-based. 140 langs, 128K ctx. Strong reasoning.",
    },

    "phi": {
        "family_patterns": ["phi"],
        "options": {
            # phi4-mini-reasoning: math/logic focused, thinking model
            "temperature": 0.3,
            "top_p": 0.8,
        },
        "notes": "Microsoft. Math/reasoning focused. 128K ctx. May overthink extraction.",
    },

    "lfm": {
        "family_patterns": ["lfm"],
        "options": {
            # Official default: temp=0.05, top_k=50
            "temperature": 0.05,
            "top_k": 50,
        },
        "notes": "Liquid AI. 1.2B only, 32K ctx. Official temp=0.05 (very deterministic).",
    },

    "exaone": {
        "family_patterns": ["exaone"],
        "options": {
            # Official default: temp=0.6
            "temperature": 0.3,  # Lower than default for JSON extraction
        },
        "notes": "LG AI Research. Reasoning model. 2.4B smallest. 32K ctx.",
    },

    "deepscaler": {
        "family_patterns": ["deepscaler"],
        "options": {
            # Based on DeepSeek-R1-Distilled-Qwen-1.5B, reasoning model
            "temperature": 0.3,
            "top_p": 0.8,
        },
        "notes": "Agentica. 1.5B math reasoning. 128K ctx. Very small for extraction.",
    },

    "llama": {
        "family_patterns": ["llama"],
        "options": {
            "temperature": 0.3,
            "top_p": 0.8,
        },
        "notes": "Meta. Untested for extraction.",
    },
}

# Fallback for completely unknown models — conservative params for JSON extraction
DEFAULT_OPTIONS = {
    "temperature": 0.3,
    "top_p": 0.8,
    "top_k": 20,
}


# ---------------------------------------------------------------------------
# Detection + lookup
# ---------------------------------------------------------------------------

def detect_model_family(model_name: str) -> str:
    """Detect model family from model name.

    Priority:
    1. LLM_FAMILY env var (explicit override)
    2. Auto-detect by matching model name against family_patterns
    3. 'unknown' fallback
    """
    env_family = os.environ.get('LLM_FAMILY', '').strip().lower()
    if env_family:
        if env_family not in MODEL_CONFIGS:
            logger.warning(
                f"LLM_FAMILY='{env_family}' not in MODEL_CONFIGS — using default options. "
                f"Known families: {', '.join(MODEL_CONFIGS.keys())}"
            )
        return env_family

    name = model_name.lower()
    # Check patterns in order — more specific patterns should come first in config
    # We sort by pattern length descending so "qwen3.5" matches before "qwen3"
    for family, cfg in sorted(
        MODEL_CONFIGS.items(),
        key=lambda x: max(len(p) for p in x[1]["family_patterns"]),
        reverse=True,
    ):
        for pattern in cfg["family_patterns"]:
            if pattern in name:
                return family

    return 'unknown'


def get_llm_options(family: str) -> dict:
    """Get LLM sampling options for a model family.

    Returns only explicitly configured options — anything missing is left
    to the model's Ollama default (not overridden).
    """
    cfg = MODEL_CONFIGS.get(family, {})
    return dict(cfg.get("options", DEFAULT_OPTIONS))


def get_model_info() -> dict:
    """Get current model info for logging/debugging."""
    model = os.environ.get('LLM_MODEL', 'unknown')
    family = detect_model_family(model)
    options = get_llm_options(family)
    return {
        "model": model,
        "family": family,
        "options": options,
        "known_families": list(MODEL_CONFIGS.keys()),
    }
