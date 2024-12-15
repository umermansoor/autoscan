import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from autoscan.autoscan import autoscan
from autoscan.errors import PDFFileNotFoundError, PDFPageToImageConversionError, MarkdownFileWriteError, LLMProcessingError
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


@pytest.mark.asyncio
async def test_autoscan_no_cleanup(tmp_path):
    # This test ensures that if cleanup_temp=False, _cleanup_temp_files is not called
    pdf_path = "sample.pdf"
    temp_dir = tmp_path / "temp"

    with patch("autoscan.autoscan.get_or_download_file", new=AsyncMock(return_value=str(temp_dir / "sample.pdf"))), \
         patch("autoscan.autoscan.pdf_to_images", return_value=["image1.png"]), \
         patch("autoscan.autoscan._process_images_async", return_value=(["Markdown content"], 10, 20, 0.2)), \
         patch("autoscan.autoscan.write_text_to_file", new=AsyncMock(return_value="output.md")), \
         patch("autoscan.autoscan._cleanup_temp_files") as mock_cleanup, \
         patch("os.makedirs"):
        await autoscan(pdf_path, temp_dir=str(temp_dir), cleanup_temp=False)
        mock_cleanup.assert_not_called()


@pytest.mark.asyncio
async def test_autoscan_model_failure(tmp_path):
    # Test scenario where the model completion raises an Exception
    pdf_path = "sample.pdf"
    temp_dir = tmp_path / "temp"

    async def mock_completion(*args, **kwargs):
        raise RuntimeError("Model failure")

    with patch("autoscan.autoscan.get_or_download_file", return_value=str(temp_dir / "sample.pdf")), \
         patch("autoscan.autoscan.pdf_to_images", return_value=["image1.png"]), \
         patch("autoscan.autoscan.LlmModel") as MockModel, \
         patch("os.makedirs"):
        model_instance = MockModel.return_value
        model_instance.completion = mock_completion

        # We expect the _process_images_async to raise a LLMProcessingError
        with pytest.raises(LLMProcessingError, match="Model failure"):
            await autoscan(pdf_path, temp_dir=str(temp_dir))


@pytest.mark.asyncio
async def test_autoscan_markdown_write_failure(tmp_path):
    # Test scenario where write_text_to_file returns None, causing MarkdownFileWriteError
    pdf_path = "sample.pdf"
    temp_dir = tmp_path / "temp"

    with patch("autoscan.autoscan.get_or_download_file", return_value=str(temp_dir / "sample.pdf")), \
         patch("autoscan.autoscan.pdf_to_images", return_value=["image1.png"]), \
         patch("autoscan.autoscan._process_images_async", return_value=(["Some content"], 10, 20, 0.2)), \
         patch("autoscan.autoscan.write_text_to_file", return_value=None), \
         patch("os.makedirs"):
        with pytest.raises(MarkdownFileWriteError):
            await autoscan(pdf_path, temp_dir=str(temp_dir))


@pytest.mark.asyncio
async def test_autoscan_with_custom_concurrency(tmp_path):
    # Test that passing concurrency parameter works
    pdf_path = "sample.pdf"
    temp_dir = tmp_path / "temp"
    output_dir = tmp_path / "output"
    
    with patch("autoscan.autoscan.get_or_download_file", return_value=str(temp_dir / "sample.pdf")), \
         patch("autoscan.autoscan.pdf_to_images", return_value=["image1.png", "image2.png", "image3.png"]) as mock_pdf_to_images, \
         patch("autoscan.autoscan._process_images_async", return_value=(["Page1", "Page2", "Page3"], 30, 60, 0.3)) as mock_process, \
         patch("autoscan.autoscan.write_text_to_file", return_value=str(output_dir / "sample.md")), \
         patch("os.makedirs"):
        # Run with concurrency=2
        result = await autoscan(pdf_path, temp_dir=str(temp_dir), output_dir=str(output_dir), concurrency=2)
        assert len(result.markdown.split("\n\n")) == 3
        # Check that _process_images_async was called with concurrency=2
        mock_process.assert_called_with(["image1.png", "image2.png", "image3.png"], 
                                        mock_process.call_args[0][1],  # model (mocked)
                                        True,  # transcribe_images default
                                        concurrency=2)
