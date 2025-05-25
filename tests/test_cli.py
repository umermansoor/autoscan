import pytest
from unittest.mock import AsyncMock, patch

import autoscan.cli as cli
from autoscan.utils.env import get_env_var_for_model

@pytest.mark.asyncio
async def test_process_file_passes_accuracy():
    called = {}

    async def fake_autoscan(pdf_path, model_name="openai/gpt-4o", accuracy="medium", user_instructions=None, save_llm_calls=False, temp_dir=None, cleanup_temp=True):
        called['accuracy'] = accuracy
        called['model'] = model_name

    with patch('autoscan.cli.autoscan', new=fake_autoscan):
        await cli._process_file('sample.pdf', 'openai/gpt-4o', 'high')

    assert called.get('accuracy') == 'high'
    assert called.get('model') == 'openai/gpt-4o'


def test_get_env_var_for_model():
    assert get_env_var_for_model('openai/gpt-4o') == 'OPENAI_API_KEY'
    assert get_env_var_for_model('anthropic/claude') == 'ANTHROPIC_API_KEY'
    assert get_env_var_for_model('gemini/gemini-pro') == 'GEMINI_API_KEY'
    assert get_env_var_for_model('unknown/model') is None
