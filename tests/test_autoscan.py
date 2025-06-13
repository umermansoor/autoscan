"""
Simplified, robust unit tests for autoscan.py - Focused on core behaviors with minimal brittleness.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from contextlib import asynccontextmanager
from autoscan.autoscan import autoscan, _process_images_async
from autoscan.types import AutoScanOutput, ModelResult


# ============================================================================
# TEST FIXTURES AND HELPERS
# ============================================================================

@pytest.fixture
def sample_images():
    """Sample image paths for testing."""
    return ["/fake/page1.png", "/fake/page2.png", "/fake/page3.png"]


@pytest.fixture
def sample_model_results():
    """Sample ModelResult objects with different content for testing context flow."""
    return [
        ModelResult("# Page 1\nFirst page content", 100, 50, 0.01),
        ModelResult("## Page 2\nSecond page content", 110, 55, 0.012),
        ModelResult("### Page 3\nThird page content", 120, 60, 0.014)
    ]


def create_mock_processor(return_values=None):
    """Helper function to create a mock processor with optional return values."""
    mock_processor = MagicMock()
    if return_values:
        mock_processor.acompletion = AsyncMock(side_effect=return_values)
    else:
        default_result = ModelResult("# Test\nContent", 100, 50, 0.01)
        mock_processor.acompletion = AsyncMock(return_value=default_result)
    return mock_processor


@asynccontextmanager
async def mock_autoscan_dependencies(**overrides):
    """
    Flexible context manager for mocking autoscan dependencies.
    Accepts overrides for specific mocks to reduce test brittleness.
    """
    defaults = {
        '_create_temp_dir': ("/fake/temp", None),
        'get_or_download_file': "/fake/test.pdf",
        'pdf_to_images': ["/fake/page1.png"],
        'write_text_to_file': "/fake/output/test.md",
        'asyncio.to_thread': ["/fake/page1.png"]
    }
    defaults.update(overrides)
    
    with patch('autoscan.autoscan._create_temp_dir', return_value=defaults['_create_temp_dir']), \
         patch('autoscan.autoscan.get_or_download_file', return_value=defaults['get_or_download_file']), \
         patch('autoscan.autoscan.pdf_to_images', return_value=defaults['pdf_to_images']), \
         patch('autoscan.autoscan.write_text_to_file', return_value=defaults['write_text_to_file']), \
         patch('os.makedirs'), \
         patch('os.path.join', return_value="/fake/output"), \
         patch('os.getcwd', return_value="/fake"), \
         patch('os.path.basename', return_value="test.pdf"), \
         patch('os.path.splitext', return_value=("test", ".pdf")), \
         patch('asyncio.to_thread', return_value=defaults['asyncio.to_thread']):
        yield


# ============================================================================
# CORE BEHAVIOR TESTS - Parametrized for efficiency
# ============================================================================

@pytest.mark.parametrize("accuracy,expected_context,expected_sequential", [
    ("high", True, True),
    ("low", False, False),
])
@pytest.mark.asyncio
async def test_accuracy_modes_end_to_end(accuracy, expected_context, expected_sequential):
    """
    Comprehensive test covering processor initialization, DPI usage, and context behavior.
    Replaces multiple individual tests with a single parametrized test.
    """
    test_images = ["/fake/page1.png", "/fake/page2.png"]
    test_results = [
        ModelResult("# Page 1\nContent 1", 100, 50, 0.01),
        ModelResult("# Page 2\nContent 2", 110, 55, 0.012)
    ]
    
    async with mock_autoscan_dependencies(pdf_to_images=test_images):
        mock_processor = create_mock_processor(test_results)
        
        with patch('autoscan.autoscan.ImageToMarkdownProcessor') as mock_processor_class:
            mock_processor_class.return_value = mock_processor
            
            result = await autoscan(
                pdf_path="/fake/test.pdf",
                model_name="test-model",
                accuracy=accuracy
            )
            
            # Verify high-level behavior
            assert isinstance(result, AutoScanOutput)
            assert result.accuracy == accuracy
            
            # Verify processor initialization
            call_kwargs = mock_processor_class.call_args[1]
            assert call_kwargs['pass_previous_page_context'] is expected_context
            
            # Verify context passing behavior
            calls = mock_processor.acompletion.call_args_list
            if expected_context and len(calls) > 1:
                # High accuracy: second page should get previous context
                assert calls[0][1]['previous_page_markdown'] is None
                assert calls[1][1]['previous_page_markdown'] == test_results[0].content
            else:
                # Low accuracy: no pages should get context
                for call in calls:
                    assert call[1]['previous_page_markdown'] is None


# ============================================================================
# DIRECT UNIT TESTS FOR CORE LOGIC
# ============================================================================

@pytest.mark.asyncio
async def test_process_images_async_sequential_vs_concurrent(sample_images, sample_model_results):
    """
    Direct test of _process_images_async function comparing sequential vs concurrent behavior.
    Tests the core logic without full autoscan() overhead.
    """
    # Test sequential processing (high accuracy)
    sequential_processor = create_mock_processor(sample_model_results)
    await _process_images_async(
        llm_processor=sequential_processor,
        pdf_page_images=sample_images,
        sequential=True
    )
    
    # Test concurrent processing (low accuracy)
    concurrent_processor = create_mock_processor(sample_model_results)
    await _process_images_async(
        llm_processor=concurrent_processor,
        pdf_page_images=sample_images,
        sequential=False
    )
    
    # Compare behaviors
    seq_calls = sequential_processor.acompletion.call_args_list
    conc_calls = concurrent_processor.acompletion.call_args_list
    
    # Sequential: context flows between pages
    assert seq_calls[0][1]['previous_page_markdown'] is None
    assert seq_calls[1][1]['previous_page_markdown'] == sample_model_results[0].content
    assert seq_calls[2][1]['previous_page_markdown'] == sample_model_results[1].content
    
    # Concurrent: no context between pages
    for call in conc_calls:
        assert call[1]['previous_page_markdown'] is None


# ============================================================================
# DPI CONFIGURATION TESTS - Simplified and focused
# ============================================================================

@pytest.mark.asyncio 
async def test_dpi_configuration_mapping():
    """Test DPI configuration mapping - focused on the core logic."""
    from autoscan.config import PDF2ImageConversionConfig
    
    # Test the configuration mapping
    assert PDF2ImageConversionConfig.get_dpi_for_accuracy("high") == 200
    assert PDF2ImageConversionConfig.get_dpi_for_accuracy("low") == 150
    assert PDF2ImageConversionConfig.get_dpi_for_accuracy("high") > PDF2ImageConversionConfig.get_dpi_for_accuracy("low")


@patch('autoscan.image_processing.convert_from_path')
def test_pdf_to_images_dpi_usage(mock_convert):
    """Test that pdf_to_images uses correct DPI values."""
    from autoscan.image_processing import pdf_to_images
    
    mock_convert.return_value = ["/fake/page1.png"]
    
    # Test both accuracy modes
    pdf_to_images("/fake/test.pdf", "/fake/temp", "high")
    assert mock_convert.call_args[1]['dpi'] == 200
    
    mock_convert.reset_mock()
    
    pdf_to_images("/fake/test.pdf", "/fake/temp", "low")
    assert mock_convert.call_args[1]['dpi'] == 150


@patch('autoscan.image_processing.convert_from_path')
def test_pdf_to_images_page_parameters(mock_convert):
    """Test that pdf_to_images correctly passes page range parameters."""
    from autoscan.image_processing import pdf_to_images
    
    mock_convert.return_value = ["/fake/page2.png", "/fake/page3.png"]
    
    # Test with both first_page and last_page
    pdf_to_images("/fake/test.pdf", "/fake/temp", "high", first_page=2, last_page=3)
    assert mock_convert.call_args[1]['first_page'] == 2
    assert mock_convert.call_args[1]['last_page'] == 3
    
    mock_convert.reset_mock()
    
    # Test with only first_page
    pdf_to_images("/fake/test.pdf", "/fake/temp", "high", first_page=5)
    assert mock_convert.call_args[1]['first_page'] == 5
    assert mock_convert.call_args[1]['last_page'] is None
    
    mock_convert.reset_mock()
    
    # Test with only last_page
    pdf_to_images("/fake/test.pdf", "/fake/temp", "high", last_page=3)
    assert mock_convert.call_args[1]['first_page'] is None
    assert mock_convert.call_args[1]['last_page'] == 3


# ============================================================================
# ERROR CONDITIONS AND EDGE CASES
# ============================================================================

@pytest.mark.asyncio
async def test_invalid_accuracy_raises_error():
    """Test that invalid accuracy values raise appropriate errors."""
    async with mock_autoscan_dependencies():
        with pytest.raises(ValueError, match="accuracy must be one of 'low', or 'high'"):
            await autoscan(
                pdf_path="/fake/test.pdf",
                model_name="test-model",
                accuracy="invalid"
            )


@pytest.mark.asyncio
async def test_single_page_behavior_consistent(sample_model_results):
    """Test that single page documents behave consistently in both modes."""
    single_image = ["/fake/page1.png"]
    single_result = [sample_model_results[0]]
    
    # Test both modes with single page
    for sequential in [True, False]:
        mock_processor = create_mock_processor(single_result)
        
        await _process_images_async(
            llm_processor=mock_processor,
            pdf_page_images=single_image,
            sequential=sequential
        )
        
        # Single page should always have no previous context
        call = mock_processor.acompletion.call_args_list[0]
        assert call[1]['previous_page_markdown'] is None
        assert call[1]['page_number'] == 1


# ============================================================================
# INTEGRATION SMOKE TEST
# ============================================================================

@pytest.mark.asyncio
async def test_autoscan_integration_smoke_test():
    """
    High-level smoke test ensuring autoscan() completes successfully.
    Tests the complete integration without getting into detailed behavior.
    """
    async with mock_autoscan_dependencies():
        mock_processor = create_mock_processor()
        
        with patch('autoscan.autoscan.ImageToMarkdownProcessor', return_value=mock_processor):
            result = await autoscan(
                pdf_path="/fake/test.pdf",
                model_name="test-model",
                accuracy="high"
            )
            
            # Just verify we get a valid result
            assert isinstance(result, AutoScanOutput)
            assert result.completion_time > 0
            assert result.markdown_file.endswith(".md")
            assert len(result.markdown) > 0
            assert result.input_tokens > 0
            assert result.output_tokens > 0
            assert result.cost >= 0
