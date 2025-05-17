import pytest
from unittest.mock import AsyncMock, patch

import autoscan.cli as cli

@pytest.mark.asyncio
async def test_process_file_passes_contextual_flag():
    called = {}

    async def fake_autoscan(pdf_path, contextual_conversion=False):
        called['flag'] = contextual_conversion

    with patch('autoscan.cli.autoscan', new=fake_autoscan):
        await cli._process_file('sample.pdf', True)

    assert called.get('flag') is True
