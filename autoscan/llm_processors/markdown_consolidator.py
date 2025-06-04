from .base_llm_processor import BaseLLMProcessor
from autoscan.types import ModelResult
from typing import Any
import logging


logger = logging.getLogger(__name__)

class MarkdownConsolidator(BaseLLMProcessor):
    """
    Processor responsible for consolidating Markdown generated from individual PDF pages
    into a single, coherent document with improved consistency and formatting.
    
    This processor takes fragmented Markdown content from page-by-page processing and:
    - Merges content that was split across page boundaries
    - Reconstructs broken tables and ensures proper alignment
    - Eliminates duplicate headers, footers, and page artifacts
    - Improves overall document structure and flow
    - Maintains all original information while enhancing readability
    
    Part of AutoScan's Output Polishing feature.
    """

    def _initialize_processor(self, **kwargs) -> None:
        """
        Initialize processor-specific parameters for output polishing.

        Args:
            **kwargs: Additional parameters specific to the consolidation process.
        """
        self.save_llm_calls = kwargs.get('save_llm_calls', False)

    async def acompletion(
        self,
        **kwargs: Any
    ) -> ModelResult:
        """
        Consolidate fragmented Markdown from individual pages into a coherent document.

        Args:
            **kwargs: Should contain 'markdown_content' with the raw page-by-page markdown to consolidate

        Returns:
            ModelResult: The consolidated markdown content with token usage information

        Raises:
            ValueError: If markdown_content is not provided
        """
        markdown_content = kwargs.get("markdown_content", None)

        if markdown_content is None:
            raise ValueError("markdown_content must be provided for output polishing")
        
        if not markdown_content.strip():
            logger.warning("Empty markdown content provided for consolidation")
            return ModelResult(content="", prompt_tokens=0, completion_tokens=0, cost=0.0)

        logger.debug(f"🔄 Consolidating page-by-page markdown content ({len(markdown_content)} characters)")

        # Build the user message with the markdown content
        user_message = f"Please consolidate, clean up, and reorganize the following Markdown document that was generated from individual PDF pages:\n\n{markdown_content}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Add user instructions if provided
        if self.user_prompt:
            logger.debug(f"📝 Adding user instructions for output polishing ({len(self.user_prompt)} chars)")
            messages.append({"role": "user", "content": f"Additional instructions: {self.user_prompt}"})

        logger.debug(f"🔍 Sending output polishing request to {self.model_name}")

        return await self._allm_call(
            messages=messages,
            is_strip_code_fences=True
        )
