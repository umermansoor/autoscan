"""
Refactored unit tests for autoscan.py - Focused on testing previous_page_markdown 
parameter behavior with improved maintainability and reduced brittleness.
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


@asynccontextmanager
async def mock_autoscan_dependencies():
    """
    Context manager that mocks all external dependencies for autoscan function.
    This reduces duplication and makes tests more maintainable.
    """
    with patch('autoscan.autoscan._create_temp_dir', return_value=("/fake/temp", None)), \
         patch('autoscan.autoscan.get_or_download_file', return_value="/fake/test.pdf"), \
         patch('autoscan.autoscan.pdf_to_images', return_value=["/fake/page1.png"]), \
         patch('autoscan.autoscan.write_text_to_file', return_value="/fake/output/test.md"), \
         patch('os.makedirs'), \
         patch('os.path.join', return_value="/fake/output"), \
         patch('os.getcwd', return_value="/fake"), \
         patch('os.path.basename', return_value="test.pdf"), \
         patch('os.path.splitext', return_value=("test", ".pdf")), \
         patch('asyncio.to_thread', return_value=["/fake/page1.png"]):
        yield


def create_mock_processor(return_values=None):
    """
    Helper function to create a mock processor with optional return values.
    """
    mock_processor = MagicMock()
    if return_values:
        mock_processor.acompletion = AsyncMock(side_effect=return_values)
    else:
        default_result = ModelResult("# Test\nContent", 100, 50, 0.01)
        mock_processor.acompletion = AsyncMock(return_value=default_result)
    return mock_processor


# ============================================================================
# CORE FUNCTIONALITY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_processor_initialization_high_accuracy():
    """
    Test that ImageToMarkdownProcessor is initialized with pass_previous_page_context=True 
    when accuracy is 'high'.
    """
    async with mock_autoscan_dependencies():
        with patch('autoscan.autoscan.ImageToMarkdownProcessor') as mock_processor_class:
            mock_processor_class.return_value = create_mock_processor()
            
            await autoscan(
                pdf_path="/fake/test.pdf",
                model_name="test-model",
                accuracy="high"
            )
            
            # Verify the key parameter
            call_kwargs = mock_processor_class.call_args[1]
            assert call_kwargs['pass_previous_page_context'] is True


@pytest.mark.asyncio
async def test_processor_initialization_low_accuracy():
    """
    Test that ImageToMarkdownProcessor is initialized with pass_previous_page_context=False 
    when accuracy is 'low'.
    """
    async with mock_autoscan_dependencies():
        with patch('autoscan.autoscan.ImageToMarkdownProcessor') as mock_processor_class:
            mock_processor_class.return_value = create_mock_processor()
            
            await autoscan(
                pdf_path="/fake/test.pdf",
                model_name="test-model",
                accuracy="low"
            )
            
            # Verify the key parameter
            call_kwargs = mock_processor_class.call_args[1]
            assert call_kwargs['pass_previous_page_context'] is False


# ============================================================================
# PROCESS IMAGES ASYNC TESTS (Direct unit tests)
# ============================================================================

@pytest.mark.asyncio
async def test_sequential_processing_passes_previous_context(sample_images, sample_model_results):
    """
    Test that sequential processing (high accuracy) correctly passes previous page context.
    """
    mock_processor = create_mock_processor(sample_model_results)
    
    # Execute sequential processing
    aggregated_markdown, _, _, _ = await _process_images_async(
        llm_processor=mock_processor,
        pdf_page_images=sample_images,
        sequential=True
    )
    
    # Verify call pattern
    calls = mock_processor.acompletion.call_args_list
    assert len(calls) == 3
    
    # Check previous context flow
    assert calls[0][1]['previous_page_markdown'] is None  # First page: no context
    assert calls[1][1]['previous_page_markdown'] == sample_model_results[0].content  # Second page: page 1 context
    assert calls[2][1]['previous_page_markdown'] == sample_model_results[1].content  # Third page: page 2 context
    
    # Verify page numbers are correct
    for i, call in enumerate(calls):
        assert call[1]['page_number'] == i + 1


@pytest.mark.asyncio
async def test_concurrent_processing_no_previous_context(sample_images, sample_model_results):
    """
    Test that concurrent processing (low accuracy) does NOT pass previous page context.
    """
    mock_processor = create_mock_processor(sample_model_results)
    
    # Execute concurrent processing
    aggregated_markdown, _, _, _ = await _process_images_async(
        llm_processor=mock_processor,
        pdf_page_images=sample_images,
        sequential=False
    )
    
    # Verify call pattern
    calls = mock_processor.acompletion.call_args_list
    assert len(calls) == 3
    
    # All pages should have no previous context
    for call in calls:
        assert call[1]['previous_page_markdown'] is None


@pytest.mark.asyncio
async def test_single_page_behavior(sample_model_results):
    """
    Test that single page documents behave consistently in both modes.
    """
    single_image = ["/fake/page1.png"]
    single_result = [sample_model_results[0]]
    
    # Test both sequential and concurrent modes
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
# INTEGRATION TESTS (Focused on behavior verification)
# ============================================================================

@pytest.mark.asyncio
async def test_high_accuracy_workflow():
    """
    Integration test verifying the complete high accuracy workflow with context passing.
    """
    test_images = ["/fake/page1.png", "/fake/page2.png"]
    test_results = [
        ModelResult("# Page 1\nContent 1", 100, 50, 0.01),
        ModelResult("# Page 2\nContent 2", 110, 55, 0.012)
    ]
    
    with patch('autoscan.autoscan._create_temp_dir', return_value=("/fake/temp", None)), \
         patch('autoscan.autoscan.get_or_download_file', return_value="/fake/test.pdf"), \
         patch('autoscan.autoscan.pdf_to_images', return_value=test_images), \
         patch('autoscan.autoscan.write_text_to_file', return_value="/fake/output/test.md"), \
         patch('os.makedirs'), \
         patch('asyncio.to_thread', return_value=test_images):
        
        mock_processor = create_mock_processor(test_results)
        
        with patch('autoscan.autoscan.ImageToMarkdownProcessor', return_value=mock_processor):
            result = await autoscan(
                pdf_path="/fake/test.pdf",
                model_name="test-model",
                accuracy="high"
            )
            
            # Verify high-level behavior
            assert isinstance(result, AutoScanOutput)
            assert result.accuracy == "high"
            
            # Verify context was passed correctly
            calls = mock_processor.acompletion.call_args_list
            assert calls[0][1]['previous_page_markdown'] is None
            assert calls[1][1]['previous_page_markdown'] == test_results[0].content


@pytest.mark.asyncio
async def test_low_accuracy_workflow():
    """
    Integration test verifying the complete low accuracy workflow without context passing.
    """
    test_images = ["/fake/page1.png", "/fake/page2.png"]
    test_results = [
        ModelResult("# Page 1\nContent 1", 100, 50, 0.01),
        ModelResult("# Page 2\nContent 2", 110, 55, 0.012)
    ]
    
    with patch('autoscan.autoscan._create_temp_dir', return_value=("/fake/temp", None)), \
         patch('autoscan.autoscan.get_or_download_file', return_value="/fake/test.pdf"), \
         patch('autoscan.autoscan.pdf_to_images', return_value=test_images), \
         patch('autoscan.autoscan.write_text_to_file', return_value="/fake/output/test.md"), \
         patch('os.makedirs'), \
         patch('asyncio.to_thread', return_value=test_images):
        
        mock_processor = create_mock_processor(test_results)
        
        with patch('autoscan.autoscan.ImageToMarkdownProcessor', return_value=mock_processor):
            result = await autoscan(
                pdf_path="/fake/test.pdf",
                model_name="test-model",
                accuracy="low"
            )
            
            # Verify high-level behavior
            assert isinstance(result, AutoScanOutput)
            assert result.accuracy == "low"
            
            # Verify no context was passed
            calls = mock_processor.acompletion.call_args_list
            for call in calls:
                assert call[1]['previous_page_markdown'] is None


# ============================================================================
# EDGE CASES AND ERROR CONDITIONS
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


# ============================================================================
# BEHAVIOR COMPARISON TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_accuracy_mode_comparison(sample_images):
    """
    Direct comparison test showing the payload differences between accuracy modes.
    This test clearly demonstrates the core difference being tested.
    """
    test_results = [
        ModelResult("Page 1 content", 100, 50, 0.01),
        ModelResult("Page 2 content", 100, 50, 0.01)
    ]
    
    # Test high accuracy (sequential)
    high_processor = create_mock_processor(test_results)
    await _process_images_async(
        llm_processor=high_processor,
        pdf_page_images=sample_images[:2],  # Use 2 pages for clarity
        sequential=True
    )
    
    # Test low accuracy (concurrent) 
    low_processor = create_mock_processor(test_results)
    await _process_images_async(
        llm_processor=low_processor,
        pdf_page_images=sample_images[:2],
        sequential=False
    )
    
    # Compare behaviors
    high_calls = high_processor.acompletion.call_args_list
    low_calls = low_processor.acompletion.call_args_list
    
    # High accuracy: second page gets context
    assert high_calls[0][1]['previous_page_markdown'] is None
    assert high_calls[1][1]['previous_page_markdown'] == test_results[0].content
    
    # Low accuracy: no pages get context
    assert low_calls[0][1]['previous_page_markdown'] is None  
    assert low_calls[1][1]['previous_page_markdown'] is None
    
    # Both should have same page numbers and image paths
    for i in range(2):
        assert high_calls[i][1]['page_number'] == low_calls[i][1]['page_number']
        assert high_calls[i][1]['image_path'] == low_calls[i][1]['image_path']


# ============================================================================
# PARAMETRIZED TESTS FOR EFFICIENCY
# ============================================================================

@pytest.mark.parametrize("accuracy,expected_context", [
    ("high", True),
    ("low", False),
])
@pytest.mark.asyncio
async def test_accuracy_mode_processor_initialization(accuracy, expected_context):
    """
    Parametrized test for processor initialization based on accuracy mode.
    """
    async with mock_autoscan_dependencies():
        with patch('autoscan.autoscan.ImageToMarkdownProcessor') as mock_processor_class:
            mock_processor_class.return_value = create_mock_processor()
            
            await autoscan(
                pdf_path="/fake/test.pdf",
                model_name="test-model",
                accuracy=accuracy
            )
            
            call_kwargs = mock_processor_class.call_args[1]
            assert call_kwargs['pass_previous_page_context'] is expected_context
