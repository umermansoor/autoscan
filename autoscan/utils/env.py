from __future__ import annotations
import os

MODEL_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def get_env_var_for_model(model_name: str) -> str | None:
    """Return the required environment variable for a provider if any."""
    provider = model_name.split("/", 1)[0].lower()
    return MODEL_ENV_VARS.get(provider)


def ensure_env_for_model(model_name: str) -> None:
    """Raise ValueError if the appropriate environment variable is missing."""
    env_var = get_env_var_for_model(model_name)
    if env_var and not os.environ.get(env_var):
        raise ValueError(f"{env_var} environment variable is not set.")

