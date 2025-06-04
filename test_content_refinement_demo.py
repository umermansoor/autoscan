#!/usr/bin/env python3
"""
Simple integration test for the new content refinement feature.
This test doesn't require API keys since it mocks the LLM calls.
"""

import asyncio
from unittest.mock import patch, AsyncMock
from autoscan.llm_processors.content_refiner import ContentRefiner
from autoscan.prompts import POST_PROCESSING_PROMPT
from autoscan.types import ModelResult


async def test_content_refinement_demo():
    """
    Demonstrates the content refinement feature working.
    """
    print("🧪 Testing AutoScan Content Refinement Feature")
    print("=" * 60)
    
    # Sample messy markdown that would come from PDF conversion
    messy_markdown = """
# Page 1 Header
Some content here.

| Name | Age |
|------|-----|
| John | 25

# Page 1 Header  
More content.

| Name | Age |
|------|-----|
| Jane | 30

Some fragmented
sentence that got
split across lines.

# Page 2 Header
Final content.
    """
    
    # What we expect after refinement
    refined_markdown = """
# Document Title

## Personal Information

| Name | Age |
|------|-----|
| John | 25  |
| Jane | 30  |

Some fragmented sentence that got split across lines has been properly reconstructed.

## Additional Content

Final content.
    """
    
    print("📄 Original messy markdown:")
    print("-" * 40)
    print(messy_markdown.strip())
    print()
    
    # Create the content refiner
    content_refiner = ContentRefiner(
        model_name="test-model",
        system_prompt=POST_PROCESSING_PROMPT,
        user_prompt="Focus on table formatting and content organization",
    )
    
    # Mock the LLM call to return our refined content
    fake_result = ModelResult(
        content=refined_markdown.strip(),
        prompt_tokens=150,
        completion_tokens=80,
        cost=0.015
    )
    
    print("🔄 Processing with content refinement...")
    
    with patch.object(content_refiner, '_allm_call', new_callable=AsyncMock, return_value=fake_result):
        result = await content_refiner.acompletion(markdown_content=messy_markdown)
        
        print("✅ Content refinement completed!")
        print(f"📊 Tokens used: {result.prompt_tokens} input, {result.completion_tokens} output")
        print(f"💰 Cost: ${result.cost:.4f}")
        print()
        print("📄 Refined markdown:")
        print("-" * 40)
        print(result.content)
        print()
        
        # Verify the feature worked
        assert result.content == refined_markdown.strip()
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0
        assert result.cost > 0
        
        print("🎉 Content refinement feature test passed!")
        return True


if __name__ == "__main__":
    asyncio.run(test_content_refinement_demo())
