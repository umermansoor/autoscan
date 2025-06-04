from .base_llm_processor import BaseLLMProcessor
from autoscan.types import ModelResult
from typing import Any
import logging


logger = logging.getLogger(__name__)

class ContentRefiner(BaseLLMProcessor):
    """
    Processor for refining and improving Markdown content using an LLM.
    
    This processor takes already-generated Markdown and improves its formatting,
    structure, and organization while preserving all original information.
    Part of AutoScan's Content Refinement feature.
    """

    def _initialize_processor(self, **kwargs) -> None:
        """
        Initialize processor-specific parameters for content refinement.

        Args:
            **kwargs: Additional parameters specific to the content refinement.
        """
        self.save_llm_calls = kwargs.get('save_llm_calls', False)

    async def acompletion(
        self,
        **kwargs: Any
    ) -> ModelResult:
        """
        Refine the provided Markdown content to improve formatting and organization.

        Args:
            **kwargs: Should contain 'markdown_content' with the raw markdown to refine

        Returns:
            ModelResult: The refined markdown content with token usage information

        Raises:
            ValueError: If markdown_content is not provided
        """
        markdown_content = kwargs.get("markdown_content", None)

        if markdown_content is None:
            raise ValueError("markdown_content must be provided for content refinement")
        
        if not markdown_content.strip():
            logger.warning("Empty markdown content provided for content refinement")
            return ModelResult(content="", prompt_tokens=0, completion_tokens=0, cost=0.0)

        logger.debug(f"🔄 Refining markdown content ({len(markdown_content)} characters)")

        # Build the user message with the markdown content
        user_message = f"Please clean up, reformat, and reorganize the following Markdown document:\n\n{markdown_content}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]

        # Add user instructions if provided
        if self.user_prompt:
            logger.debug(f"📝 Adding user instructions for content refinement ({len(self.user_prompt)} chars)")
            messages.append({"role": "user", "content": f"Additional instructions: {self.user_prompt}"})

        logger.debug(f"🔍 Sending content refinement request to {self.model_name}")

        return await self._allm_call(
            messages=messages,
            is_strip_code_fences=True
        )
