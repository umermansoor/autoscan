import pytest
from unittest.mock import AsyncMock, patch

import autoscan.cli as cli
from autoscan.utils.env import get_env_var_for_model


def test_get_env_var_for_model():
    assert get_env_var_for_model('openai/gpt-4o') == 'OPENAI_API_KEY'
    assert get_env_var_for_model('anthropic/claude') == 'ANTHROPIC_API_KEY'
    assert get_env_var_for_model('gemini/gemini-pro') == 'GEMINI_API_KEY'
    assert get_env_var_for_model('unknown/model') is None
