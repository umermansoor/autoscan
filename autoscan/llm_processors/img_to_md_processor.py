from .base_llm_processor import BaseLLMProcessor
from autoscan.types import ModelResult
from autoscan.image_processing import image_to_base64
from typing import Any, Dict, List
import logging


logger = logging.getLogger(__name__)

class ImageToMarkdownProcessor(BaseLLMProcessor):
    """
    Processor for converting images to Markdown format using an LLM.
    
    Inherits from BaseLLMProcessor and implements the image-to-Markdown conversion logic.
    """

    def _initialize_processor(self, **kwargs) -> None:
        """
        A class that converts a PDF page (provided as an image) into Markdown using an LLM.

        Args:
            **kwargs: Additional parameters specific to the image-to-Markdown processing.
        """

        self.pass_previous_page_context = kwargs.get("pass_previous_page_context", False)
        self.save_llm_calls = kwargs.get('save_llm_calls', False)

    async def acompletion(
        self,
        **kwargs: Any
    ) -> ModelResult:
        page_number = kwargs.get("page_number", -1)
        image_path = kwargs.get("image_path", None)
        previous_page_markdown = kwargs.get("previous_page_markdown", None)

        if not image_path:
            raise ValueError("image_path must be provided for Image-to-Markdown conversion")
        
        try:
            base64_image = image_to_base64(image_path)
            logger.debug(f"📁 {page_number}: Image encoded to base64 ({len(base64_image)} chars)")
        except Exception as e:
            logger.error(f"❌ {page_number}: Failed to encode image to base64: {e}")
            raise ValueError(f"Failed to encode image at {image_path} to base64") from e
        
        # -- 1. Current page to be converted to Markdown
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": "Convert the following image to markdown."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            },
        ]
        

        # --2. Previous page context if available
        if self.pass_previous_page_context and previous_page_markdown:
            logger.debug(f"🔗 {page_number}: Adding previous page context ({len(previous_page_markdown)} chars)")

            context_md = previous_page_markdown
            intro = (
                "Here is the previous page markdown for continuity context. "
                "IMPORTANT: Do NOT repeat any content from the previous page. "
                "If tables CONTINUE across pages, ONLY provide data rows (NO headers, NO separators). "
                "Ensure seamless continuation without duplicating previous content."
            )
            
            # Add the context markdown only (removing previous page image to prevent duplication)
            user_content.append({
                "type": "text",
                "text": f"{intro}\n<!-- PAGE SEPARATOR -->\n{context_md}"
            })

        # # -- 3. any ad-hoc user instructions 
        if self.user_prompt:
            logger.debug(f"📝 {page_number}: Adding user instructions ({len(self.user_prompt)} chars)")
            user_content.append({"type": "text", "text": self.user_prompt})

        system_prompt = self.system_prompt
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        logger.debug(f"🔍 {page_number}: Sending request to {self.model_name}")

        return await self._allm_call(
            messages=messages,
            is_strip_code_fences=True
        )


        



