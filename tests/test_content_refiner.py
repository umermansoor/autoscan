"""
Test suite for ContentRefiner class which is responsible
for refining and improving Markdown content using an LLM.
Part of AutoScan's Content Refinement feature.
"""

import pytest
from unittest.mock import patch, AsyncMock
from autoscan.llm_processors.content_refiner import ContentRefiner
from autoscan.types import ModelResult


@pytest.fixture
def refiner():
    """
    Create a test instance of ContentRefiner with default test configuration.
    """
    return ContentRefiner(
        model_name="test-model",
        system_prompt="Clean up this markdown",
        user_prompt="Make it professional",
    )


@pytest.mark.asyncio
async def test_acompletion_raises_on_missing_markdown_content(refiner):
    """
    Test that acompletion method raises ValueError when markdown_content is not provided.
    """
    with pytest.raises(ValueError, match="markdown_content must be provided"):
        await refiner.acompletion()


@pytest.mark.asyncio
async def test_acompletion_handles_empty_content(refiner):
    """
    Test that acompletion handles empty markdown content gracefully.
    """
    result = await refiner.acompletion(markdown_content="")
    assert result.content == ""
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.cost == 0.0


@pytest.mark.asyncio
async def test_acompletion_success(refiner):
    """
    Test successful completion of content refinement.
    """
    # Create a fake LLM response
    fake_result = ModelResult("# Refined Markdown\nThis is much better!", 150, 75, 0.02)
    
    # Mock the internal LLM call method to return our fake result
    with patch.object(refiner, '_allm_call', new_callable=AsyncMock, return_value=fake_result):
        result = await refiner.acompletion(
            markdown_content="# messy markdown\nthis needs refinement..."
        )
        # Verify that the refiner returns the expected result
        assert result == fake_result


@pytest.mark.asyncio
async def test_acompletion_with_user_instructions(refiner):
    """
    Test that acompletion includes user instructions in the messages sent to the LLM.
    """
    refiner.user_prompt = "Focus on table formatting"
    fake_result = ModelResult("refined content", 100, 50, 0.01)
    
    with patch.object(refiner, '_allm_call', new_callable=AsyncMock, return_value=fake_result) as mock_call:
        await refiner.acompletion(
            markdown_content="Some markdown content"
        )
        
        # Check that user instructions were included
        messages = mock_call.call_args[1]['messages']
        assert len(messages) == 3  # system, user content, user instructions
        assert "Focus on table formatting" in str(messages)


@pytest.mark.asyncio
async def test_acompletion_assembles_messages_correctly():
    """
    Test that acompletion correctly assembles the messages sent to the LLM.
    """
    system_prompt = "You are a markdown formatter"
    user_prompt = "Focus on headers"
    markdown_content = "# Bad Header\nsome content"
    
    refiner = ContentRefiner(
        model_name="test-model",
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    
    fake_result = ModelResult("formatted content", 100, 50, 0.01)
    
    with patch.object(refiner, '_allm_call', new_callable=AsyncMock, return_value=fake_result) as mock_call:
        await refiner.acompletion(markdown_content=markdown_content)
        
        # Check the actual messages argument passed to _allm_call
        called_messages = mock_call.call_args[1]['messages']
        
        # First message should be the system prompt
        assert called_messages[0]['role'] == "system"
        assert called_messages[0]['content'] == system_prompt
        
        # Second message should contain the markdown content
        assert called_messages[1]['role'] == "user"
        assert markdown_content in called_messages[1]['content']
        
        # Third message should contain user instructions
        assert called_messages[2]['role'] == "user"
        assert user_prompt in called_messages[2]['content']


@pytest.mark.asyncio
async def test_acompletion_without_user_prompt():
    """
    Test that acompletion works correctly when no user prompt is provided.
    """
    refiner = ContentRefiner(
        model_name="test-model",
        system_prompt="Clean markdown",
        user_prompt="",  # Empty user prompt
    )
    
    fake_result = ModelResult("clean content", 80, 40, 0.008)
    
    with patch.object(refiner, '_allm_call', new_callable=AsyncMock, return_value=fake_result) as mock_call:
        await refiner.acompletion(markdown_content="messy content")
        
        # Should only have system and user content messages, no additional instructions
        messages = mock_call.call_args[1]['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == "system"
        assert messages[1]['role'] == "user"
