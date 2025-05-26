import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from autoscan.autoscan import autoscan
import autoscan.autoscan as autoscan_module
from autoscan.errors import PDFFileNotFoundError, PDFPageToImageConversionError, MarkdownFileWriteError, LLMProcessingError
from autoscan.types import AutoScanOutput, ModelResult


@pytest.mark.asyncio
async def test_autoscan_successful(tmp_path):
    # Mock inputs
    pdf_path = "sample.pdf"
    model_name = "openai/gpt-4o"
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
async def test_autoscan_auto_created_temp_dir_cleanup(tmp_path):
    # This test ensures that when we auto-create temp directory, cleanup occurs
    pdf_path = "sample.pdf"

    with patch("autoscan.autoscan.get_or_download_file", new=AsyncMock(return_value=str(tmp_path / "sample.pdf"))), \
         patch("autoscan.autoscan.pdf_to_images", new=MagicMock(return_value=["image1.png", "image2.png"])), \
         patch("autoscan.autoscan._process_images_async", new=AsyncMock(return_value=(["Markdown Page 1"], 50, 50, 0.1))), \
         patch("autoscan.autoscan.write_text_to_file", new=AsyncMock(return_value="sample.pdf")) as mock_write, \
         patch("autoscan.autoscan._cleanup_temp_files") as mock_cleanup, \
         patch("autoscan.autoscan.LlmModel") as MockModel, \
         patch("os.makedirs"):
        # Mock the LlmModel
        MockModel.return_value.completion = AsyncMock()

        # Run the function without specifying temp_dir (so it auto-creates one)
        await autoscan(pdf_path)
        # Ensure the cleanup function was called since we auto-created temp dir
        mock_cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_autoscan_user_provided_temp_dir_no_cleanup(tmp_path):
    # This test ensures that when user provides temp_dir, _cleanup_temp_files is not called
    pdf_path = "sample.pdf"
    temp_dir = tmp_path / "temp"

    with patch("autoscan.autoscan.get_or_download_file", new=AsyncMock(return_value=str(temp_dir / "sample.pdf"))), \
         patch("autoscan.autoscan.pdf_to_images", return_value=["image1.png"]), \
         patch("autoscan.autoscan._process_images_async", return_value=(["Markdown content"], 10, 20, 0.2)), \
         patch("autoscan.autoscan.write_text_to_file", new=AsyncMock(return_value="output.md")), \
         patch("autoscan.autoscan._cleanup_temp_files") as mock_cleanup, \
         patch("autoscan.autoscan.LlmModel") as MockModel, \
         patch("os.makedirs"):
        # Mock the LlmModel
        MockModel.return_value.completion = AsyncMock()
        
        # Run with user-provided temp_dir - should not cleanup
        await autoscan(pdf_path, temp_dir=str(temp_dir))
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
        model_instance.image_to_markdown_completion = mock_completion

        # We expect the _process_images_async to raise a LLMProcessingError
        with pytest.raises(LLMProcessingError, match="Error processing image"):
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
         patch("autoscan.autoscan.LlmModel") as MockModel, \
         patch("os.makedirs"):
        # Mock the LlmModel
        MockModel.return_value.completion = AsyncMock()
        
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
         patch("autoscan.autoscan.LlmModel") as MockModel, \
         patch("os.makedirs"):
        # Mock the LlmModel
        MockModel.return_value.completion = AsyncMock()
        
        # Run with concurrency=2 and low accuracy to ensure sequential=False
        result = await autoscan(pdf_path, temp_dir=str(temp_dir), output_dir=str(output_dir), concurrency=2, accuracy="low")
        assert result is not None
        # Check that _process_images_async was called with concurrency=2
        mock_process.assert_called_with(
            ["image1.png", "image2.png", "image3.png"],
            mock_process.call_args[0][1],
            concurrency=2,
            sequential=False,
            user_instructions=None,
        )

@pytest.mark.asyncio
async def test_process_images_async_sequential():
    images = ["p1.png", "p2.png", "p3.png"]
    calls = []

    async def fake_image_to_markdown(image_path, previous_page_markdown=None, user_instructions=None, page_number=None):
        calls.append(previous_page_markdown)
        return ModelResult(
            content=f"md_{image_path}", prompt_tokens=1, completion_tokens=1, cost=0.0
        )

    model = MagicMock()
    model.image_to_markdown.side_effect = fake_image_to_markdown

    result = await autoscan_module._process_images_async(
        images, model, concurrency=2, sequential=True, user_instructions=None
    )

    assert calls == [None, "md_p1.png", "md_p2.png"]
    assert result[0] == ["md_p1.png", "md_p2.png", "md_p3.png"]

@pytest.mark.asyncio
async def test_temp_dir_cleanup_logic(tmp_path):
    """Test that cleanup works correctly for both auto-created and user-provided temp dirs"""
    pdf_path = "sample.pdf"
    
    # Test 1: Auto-created temp dir should trigger cleanup
    with patch("autoscan.autoscan.get_or_download_file", new=AsyncMock(return_value=str(tmp_path / "sample.pdf"))), \
         patch("autoscan.autoscan.pdf_to_images", return_value=["image1.png", "image2.png"]), \
         patch("autoscan.autoscan._process_images_async", return_value=(["Markdown content"], 10, 20, 0.2)), \
         patch("autoscan.autoscan.write_text_to_file", new=AsyncMock(return_value="output.md")), \
         patch("autoscan.autoscan._cleanup_temp_files") as mock_cleanup_auto, \
         patch("autoscan.autoscan.LlmModel") as MockModel, \
         patch("tempfile.TemporaryDirectory") as mock_temp_dir, \
         patch("os.makedirs"):
        
        # Mock the TemporaryDirectory
        temp_dir_instance = MagicMock()
        temp_dir_instance.name = "/tmp/auto_created"
        temp_dir_instance.cleanup = MagicMock()
        mock_temp_dir.return_value = temp_dir_instance
        
        MockModel.return_value.completion = AsyncMock()
        
        # Run without providing temp_dir (should auto-create and cleanup)
        await autoscan(pdf_path)
        
        # Should cleanup images and temp directory
        mock_cleanup_auto.assert_called_once()
        temp_dir_instance.cleanup.assert_called_once()
    
    # Test 2: User-provided temp dir should NOT trigger cleanup
    user_temp_dir = tmp_path / "user_temp"
    with patch("autoscan.autoscan.get_or_download_file", new=AsyncMock(return_value=str(tmp_path / "sample.pdf"))), \
         patch("autoscan.autoscan.pdf_to_images", return_value=["image1.png", "image2.png"]), \
         patch("autoscan.autoscan._process_images_async", return_value=(["Markdown content"], 10, 20, 0.2)), \
         patch("autoscan.autoscan.write_text_to_file", new=AsyncMock(return_value="output.md")), \
         patch("autoscan.autoscan._cleanup_temp_files") as mock_cleanup_user, \
         patch("autoscan.autoscan.LlmModel") as MockModel, \
         patch("os.makedirs"):
        
        MockModel.return_value.completion = AsyncMock()
        
        # Run with user-provided temp_dir (should NOT cleanup)
        await autoscan(pdf_path, temp_dir=str(user_temp_dir))
        
        # Should NOT cleanup images since user provided the directory
        mock_cleanup_user.assert_not_called()

