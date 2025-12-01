"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# Provider selection: "openrouter" (default) or "deepseek"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv(
    "DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions"
)
DEFAULT_DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner")
DEFAULT_DEEPSEEK_CHAIRMAN = os.getenv("DEEPSEEK_CHAIRMAN_MODEL", DEFAULT_DEEPSEEK_MODEL)

# Council members - list of model identifiers
if LLM_PROVIDER == "deepseek":
    COUNCIL_MODELS = [DEFAULT_DEEPSEEK_MODEL]
    CHAIRMAN_MODEL = DEFAULT_DEEPSEEK_CHAIRMAN
else:
    COUNCIL_MODELS = [
        "openai/gpt-5.1",
        "google/gemini-3-pro-preview",
        "anthropic/claude-sonnet-4.5",
        "x-ai/grok-4",
    ]
    CHAIRMAN_MODEL = "google/gemini-3-pro-preview"

# Model used for generating conversation titles
TITLE_MODEL = os.getenv("TITLE_MODEL", CHAIRMAN_MODEL if LLM_PROVIDER == "deepseek" else "google/gemini-2.5-flash")

# Data directory for conversation storage
DATA_DIR = "data/conversations"
