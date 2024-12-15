import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from autoscan.autoscan import autoscan
from autoscan.errors import PDFFileNotFoundError, PDFPageToImageConversionError
from autoscan.types import AutoScanOutput

@pytest.mark.asyncio
async def test_autoscan_successful(tmp_path):
    # Mock inputs
    pdf_path = "sample.pdf"
    model_name = "gpt-4o"
    output_dir = tmp_path / "output"
    temp_dir = tmp_path / "temp"

    # Mock dependencies
    with patch("autoscan.autoscan.get_or_download_file", new=AsyncMock(return_value=str(tmp_path / "sample.pdf"))) as mock_download, \
         patch("autoscan.autoscan.pdf_to_images", new=MagicMock(return_value=["image1.png", "image2.png"])) as mock_pdf_to_images, \
         patch("autoscan.autoscan._process_images_async", new=AsyncMock(return_value=(["Markdown Page 1", "Markdown Page 2"], 100, 200, 0.5))) as mock_process, \
         patch("autoscan.autoscan.write_text_to_file", new=AsyncMock(return_value=str(output_dir / "sample.md"))) as mock_write, \
         patch("autoscan.autoscan.LlmModel") as MockModel, \
         patch("os.makedirs") as mock_makedirs:

        # Mock the LlmModel
        MockModel.return_value.completion = AsyncMock()

        # Run the function
        result = await autoscan(pdf_path, model_name, output_dir=str(output_dir), temp_dir=str(temp_dir))

        # Assertions
        assert isinstance(result, AutoScanOutput)
        assert result.completion_time > 0
        assert "Markdown Page 1" in result.markdown
        assert "Markdown Page 2" in result.markdown
        assert result.input_tokens == 100
        assert result.output_tokens == 200
        assert result.markdown_file == str(output_dir / "sample.md")

        # Ensure mocks were called
        mock_download.assert_called_once_with(pdf_path, str(temp_dir))
        mock_pdf_to_images.assert_called_once_with(str(tmp_path / "sample.pdf"), str(temp_dir))
        mock_process.assert_called_once()
        mock_write.assert_called_once()
        mock_makedirs.assert_called_with(str(output_dir), exist_ok=True)

@pytest.mark.asyncio
async def test_autoscan_pdf_not_found():
    # Mock inputs
    pdf_path = "nonexistent.pdf"

    # Mock dependencies
    with patch("autoscan.autoscan.get_or_download_file", new=AsyncMock(return_value=None)):
        # Run the function and verify it raises the correct exception
        with pytest.raises(PDFFileNotFoundError):
            await autoscan(pdf_path)

@pytest.mark.asyncio
async def test_autoscan_no_images_generated(tmp_path):
    # Mock inputs
    pdf_path = "sample.pdf"
    temp_dir = tmp_path / "temp"

    # Mock dependencies
    with patch("autoscan.autoscan.get_or_download_file", new=AsyncMock(return_value=str(temp_dir / "sample.pdf"))), \
         patch("autoscan.autoscan.pdf_to_images", new=MagicMock(return_value=[])), \
         patch("os.makedirs"):

        # Run the function and verify it raises a RuntimeError
        with pytest.raises(PDFPageToImageConversionError, match="Failed to convert PDF pages to images."):
            await autoscan(pdf_path, temp_dir=str(temp_dir))

@pytest.mark.asyncio
async def test_autoscan_temp_dir_cleanup(tmp_path):
    # Mock inputs
    pdf_path = "sample.pdf"
    temp_dir = tmp_path / "temp"

    # Mock dependencies
    with patch("autoscan.autoscan.get_or_download_file", new=AsyncMock(return_value=str(temp_dir / "sample.pdf"))), \
         patch("autoscan.autoscan.pdf_to_images", new=MagicMock(return_value=["image1.png", "image2.png"])), \
         patch("autoscan.autoscan._process_images_async", new=AsyncMock(return_value=(["Markdown Page 1"], 50, 50, 0.1))), \
         patch("autoscan.autoscan.write_text_to_file", new=AsyncMock(return_value="sample.pdf")) as mock_write, \
         patch("autoscan.autoscan._cleanup_temp_files") as mock_cleanup, \
         patch("os.makedirs"):

        # Run the function
        await autoscan(pdf_path, temp_dir=str(temp_dir), cleanup_temp=True)

        # Ensure the cleanup function was called
        mock_cleanup.assert_called_once()
