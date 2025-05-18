import pytest
from unittest.mock import AsyncMock, patch

import autoscan.cli as cli

@pytest.mark.asyncio
async def test_process_file_passes_accuracy():
    called = {}

    async def fake_autoscan(pdf_path, accuracy="medium", debug=False):
        called['accuracy'] = accuracy
        called['debug'] = debug

    with patch('autoscan.cli.autoscan', new=fake_autoscan):
        await cli._process_file('sample.pdf', 'high')

    assert called.get('accuracy') == 'high'
    assert called.get('debug') is False
