"""
Test suite for ImageToMarkdownProcessor class which is responsible
for converting PDF page images to Markdown format using an LLM.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from autoscan.llm_processors.img_to_md_processor import ImageToMarkdownProcessor
from autoscan.types import ModelResult
from autoscan.image_processing import image_to_base64
from autoscan.errors import LLMProcessingError


@pytest.fixture
def processor():
    """
    Create a test instance of ImageToMarkdownProcessor with default test configuration.

    """
    return ImageToMarkdownProcessor(
        model_name="test-model",
        system_prompt="system",
        user_prompt="user",
        pass_previous_page_context=True,
    )

@pytest.mark.asyncio
async def test_acompletion_raises_on_missing_image_path(processor):
    """
    Test that acompletion method raises ValueError when image_path is not provided.
    """
    with pytest.raises(ValueError):
        await processor.acompletion(page_number=1)

@pytest.mark.asyncio
async def test_acompletion_success(processor):
    """
    Test successful completion of image-to-markdown conversion with all required components.
    
    This test verifies the complete workflow of the ImageToMarkdownProcessor:
    1. Image encoding to base64 (mocked)
    2. LLM call execution (mocked)
    3. Previous page context integration when provided
    4. Proper return of ModelResult

    """
    
    with patch('autoscan.llm_processors.img_to_md_processor.image_to_base64', return_value='base64string'):
        # Create a fake LLM response 
        fake_result = ModelResult("some markdown", 1, 2, 0.01)
        
        # Mock the internal LLM call method to return our fake result
        with patch.object(processor, '_allm_call', new_callable=AsyncMock, return_value=fake_result):
            # Test the complete acompletion workflow with all parameters
            result = await processor.acompletion(
                page_number=1,                              # Page identifier for logging
                image_path="dummy.png",                     # Mock file path (won't be accessed)
                previous_page_markdown='Previous content'   # Context for continuity
            )
            # Verify that the processor returns the expected result
            assert result == fake_result

@pytest.mark.asyncio
async def test_acompletion_with_base64_encoding_error(processor):
    """
    Test that acompletion raises ValueError when image_to_base64 fails.
    
    This simulates an error during the base64 encoding process (e.g., file not found,
    permission denied, etc.) to ensure the processor properly handles and re-raises
    these errors as ValueError with descriptive messages.
    """
    with patch('autoscan.llm_processors.img_to_md_processor.image_to_base64', side_effect=Exception("Encoding error")):
        with pytest.raises(ValueError, match=r".*dummy\.png.*base64.*"):
            await processor.acompletion(
                page_number=1,
                image_path="dummy.png"
            )

@pytest.mark.asyncio
async def test_acompletion_with_llm_call_error(processor):
    """
    Test that acompletion raises LLMProcessingError when the internal LLM call fails.
    """
    with patch('autoscan.llm_processors.img_to_md_processor.image_to_base64', return_value='base64string'):
        with patch.object(processor, '_allm_call', new_callable=AsyncMock, side_effect=LLMProcessingError("LLM API error")):
            with pytest.raises(LLMProcessingError):
                await processor.acompletion(
                    page_number=1,
                    image_path="dummy.png"
                )

@pytest.mark.asyncio
async def test_acompletion_without_previous_page_context():
    """
    Test that acompletion does not include previous page markdown when 
    pass_previous_page_context is False, even if provided.
    """
    processor = ImageToMarkdownProcessor(
        model_name="test-model",
        system_prompt="system",
        user_prompt="user",
        pass_previous_page_context=False,
    )
    with patch('autoscan.llm_processors.img_to_md_processor.image_to_base64', return_value='base64string'):
        fake_result = ModelResult("md", 1, 2, 0.01)
        with patch.object(processor, '_allm_call', new_callable=AsyncMock, return_value=fake_result) as mock_call:
            result = await processor.acompletion(
                page_number=1,
                image_path="dummy.png",
                previous_page_markdown="should be ignored"
            )
            # Inspect the messages that would be sent to _allm_call
            called_messages = mock_call.call_args[1]['messages']
            # Should NOT include previous page markdown
            assert "should be ignored" not in str(called_messages)
            assert result == fake_result

@pytest.mark.asyncio
async def test_acompletion_with_previous_page_context():
    """
    Test that acompletion includes previous page markdown when 
    pass_previous_page_context is True.
    """
    processor = ImageToMarkdownProcessor(
        model_name="test-model",
        system_prompt="system",
        user_prompt="user",
        pass_previous_page_context=True,
    )
    with patch('autoscan.llm_processors.img_to_md_processor.image_to_base64', return_value='base64string'):
        fake_result = ModelResult("md", 1, 2, 0.01)
        with patch.object(processor, '_allm_call', new_callable=AsyncMock, return_value=fake_result) as mock_call:
            result = await processor.acompletion(
                page_number=1,
                image_path="dummy.png",
                previous_page_markdown="should be included"
            )
            # Inspect the messages that would be sent to _allm_call
            called_messages = mock_call.call_args[1]['messages']
            # Should include previous page markdown
            assert "should be included" in str(called_messages)
            assert result == fake_result

@pytest.mark.asyncio
async def test_acompletion_with_user_prompt():
    """
    Test that acompletion includes user_prompt in the messages sent to the LLM.
    """
    processor = ImageToMarkdownProcessor(
        model_name="test-model",
        system_prompt="sys",
        user_prompt="USER INSTRUCTION!",
        pass_previous_page_context=False,
    )
    with patch('autoscan.llm_processors.img_to_md_processor.image_to_base64', return_value='abc'):
        fake_result = ModelResult("markdown", 1, 2, 0.01)
        with patch.object(processor, '_allm_call', new_callable=AsyncMock, return_value=fake_result) as mock_call:
            await processor.acompletion(
                page_number=1,
                image_path="dummy.png"
            )
            # Ensure user_prompt is in the message
            messages = mock_call.call_args[1]['messages']
            assert "USER INSTRUCTION!" in str(messages)

@pytest.mark.asyncio
async def test_acompletion_assembles_messages_correctly():
    """
    Test that acompletion correctly assembles the messages sent to the LLM,
    including system prompt, user prompt, image base64, and previous page context.
    """
    # Arrange test inputs
    system_prompt = "This is the system prompt"
    user_prompt = "Do it like this!"
    previous_page_md = "PREV MD"
    base64_string = "FAKEBASE64"
    
    processor = ImageToMarkdownProcessor(
        model_name="test-model",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        pass_previous_page_context=True,
    )
    
    with patch('autoscan.llm_processors.img_to_md_processor.image_to_base64', return_value=base64_string):
        fake_result = ModelResult("whatever", 1, 2, 0.01)
        with patch.object(processor, '_allm_call', new_callable=AsyncMock, return_value=fake_result) as mock_call:
            await processor.acompletion(
                page_number=7,
                image_path="foo.png",
                previous_page_markdown=previous_page_md
            )

            # ---- Check the actual messages argument passed to _allm_call ----
            called_messages = mock_call.call_args[1]['messages']

            # First message should be the system prompt
            assert called_messages[0]['role'] == "system"
            assert called_messages[0]['content'] == system_prompt

            # Second message should be the user content
            user_content = called_messages[1]['content']
            # Should always start with the conversion prompt and the image
            assert user_content[0]['type'] == "text"
            assert "Convert the following image" in user_content[0]['text']
            assert user_content[1]['type'] == "image_url"
            assert base64_string in user_content[1]['image_url']['url']

            # Should include previous page markdown (with intro) as a later user_content entry
            found_prev_context = any(
                previous_page_md in entry.get('text', '')
                for entry in user_content
            )
            assert found_prev_context

            # Should include the user prompt
            found_user_prompt = any(
                user_prompt == entry.get('text', '')
                for entry in user_content
            )
            assert found_user_prompt


