import pytest
from aioresponses import aioresponses
from pathlib import Path
from autoscan.common import get_or_download_file

@pytest.mark.asyncio
async def test_valid_http_url(tmp_path):
    url = "http://example.com/file.txt"
    content = b"sample content"
    mock_path = tmp_path / "file.txt"
    
    # Mock HTTP GET request
    with aioresponses() as mock_response:
        mock_response.get(url, status=200, body=content)
        
        # Call the function and validate the result
        result = await get_or_download_file(url, str(tmp_path))
        
        assert result == str(mock_path)
        assert mock_path.read_bytes() == content

@pytest.mark.asyncio
async def test_invalid_http_url():
    url = "http://example.com/file.txt"
    
    # Mock HTTP GET request
    with aioresponses() as mock_response:
        mock_response.get(url, status=404)
        
        # Call the function and validate it returns None
        result = await get_or_download_file(url, "/some/dir")
        
        assert result is None

@pytest.mark.asyncio
async def test_existing_local_file(tmp_path):
    local_file = tmp_path / "existing_file.txt"
    local_file.write_text("file content")
    
    # Call the function with a valid local file path
    result = await get_or_download_file(str(local_file), str(tmp_path))
    
    assert result == str(local_file)

@pytest.mark.asyncio
async def test_nonexistent_local_file(tmp_path):
    nonexistent_file = tmp_path / "nonexistent_file.txt"
    
    # Call the function with a nonexistent local file path
    result = await get_or_download_file(str(nonexistent_file), str(tmp_path))
    
    assert result is None

@pytest.mark.asyncio
async def test_malformed_url():
    url = "malformed://example.com/file.txt"
    
    # Call the function with a malformed URL
    result = await get_or_download_file(url, "/some/dir")
    
    assert result is None
