import os
import logging
from typing import List, Dict, Any, Optional

import litellm
from litellm import acompletion

from .image_processing import image_to_base64
from .types import ModelResult
from .prompts import DEFAULT_SYSTEM_PROMPT, FINAL_REVIEW_PROMPT
from .config import LLMConfig
from .utils.env import ensure_env_for_model
import tiktoken
from .errors import LLMProcessingError

logger = logging.getLogger(__name__)

class LlmModel:
    """
    A model class that converts a PDF page (provided as an image) into markdown
    using an LLM. It can maintain formatting consistency with previously processed pages.
    """

    def __init__(self, model_name: str = "openai/gpt-4o", debug: bool = False, accuracy: str = "medium"):
        """
        Initialize the LLM model interface.

        Args:
            model_name (str): The model name to use. Defaults to "openai/gpt-4o".
            debug (bool): If True, enable litellm debug logging.
        """
        self._model_name = model_name
        self._debug = debug
        self._accuracy = accuracy
        self._system_prompt = DEFAULT_SYSTEM_PROMPT
        ensure_env_for_model(model_name)

        if self._debug and hasattr(litellm, "set_verbose"):
            try:
                litellm.set_verbose(True)
            except Exception:
                logger.debug("Failed to enable litellm verbose mode")

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        """Remove enclosing triple backticks and optional language tags."""
        if content.startswith("```") and content.endswith("```"):
            content = content.removeprefix("```").removesuffix("```")
            for lang_tag in ("markdown", "md"):
                if content.startswith(lang_tag):
                    content = content[len(lang_tag):]
                    break
            content = content.strip()
        return content

    @property
    def accuracy(self) -> str:
        """Return the accuracy level for this model."""
        return self._accuracy

    @property
    def system_prompt(self) -> str:
        """
        The current system prompt.

        Returns:
            str: The system prompt string.
        """
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, prompt: str) -> None:
        """
        Set a new system prompt.

        Args:
            prompt (str): The new system prompt.
        """
        self._system_prompt = prompt

    async def image_to_markdown(
            self,
            image_path: str,
            previous_page_markdown: Optional[str] = None,
            user_instructions: Optional[str] = None,
    ) -> ModelResult:
        """
        Generate a markdown representation of a PDF page from an image.

        Args:
            image_path (str): Path to the image file of the PDF page.
            previous_page_markdown: Optional markdown of previous page in PDF file
            user_instructions: Additional instructions provided by the user
        Returns:
            ModelCompletionResult: The generated markdown and token usage details.
        """

        logger.debug(f"Converting image {image_path} to markdown")
        base64_image = image_to_base64(image_path)
        user_content = [
            {
                "type": "text",
                "text": "Convert the following image to markdown."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]
        if previous_page_markdown:
            user_content.append({
                "type": "text",
                "text": (
                    "Here are the last few characters in Markdown format from the previous page to provide you context. "
                    "The final output has no page breaks."
                    f"\n<!-- PAGE SEPARATOR -->\n{previous_page_markdown[-100:]}"
                ),
            })
        if user_instructions:
            user_content.append({"type": "text", "text": user_instructions})

        system_prompt = self._system_prompt

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        if self._debug:
            logger.debug(f"System prompt: {system_prompt}")
            for item in user_content:
                if item["type"] == "text":
                    logger.debug(f"User content: {item['text']}")

        try:
            response = await acompletion(
                model=self._model_name,
                messages=messages,
            )
            content = self._strip_code_fences(response.choices[0].message.content.strip())
            usage = response.usage  # Extract token usage
    
            try:
                total_cost = LLMConfig.get_costs_for_model(self._model_name, usage.prompt_tokens, usage.completion_tokens)
            except ValueError as e:
                logger.error(f"Cost calculation failed: {e}")
                total_cost = 0.0
            logger.debug(
                "Tokens prompt=%s completion=%s cost=%s",
                usage.prompt_tokens,
                usage.completion_tokens,
                total_cost,
            )

            # Extract required information and return a CompletionResult
            return ModelResult(
                content=content,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost=total_cost
            )
        except Exception as err: 
            raise LLMProcessingError(
                f"Image to markdown LLM call failed: {err}"
            ) from err


    async def postprocess_markdown(self, markdowns: List[str]) -> ModelResult:
        logger.debug("Post-processing %s markdown chunks", len(markdowns))

        input_tokens = self._calculate_tokens(self._model_name, "\n\n".join(markdowns))
        if input_tokens >= LLMConfig.get_max_tokens_for_model(self._model_name)["output_tokens"]:
            raise RuntimeError("Too many tokens to post-process.")


        # Combine all markdown input into a single string
        separator = "\n\n---PAGE BREAK---\n\n"
        user_content = separator.join(markdowns)

        messages = [
            {"role": "system", "content": FINAL_REVIEW_PROMPT},
            {"role": "user", "content": user_content}
        ]

        if self._debug:
            logger.debug(f"System prompt: {FINAL_REVIEW_PROMPT}")
            logger.debug(f"User content: {user_content}")

        try:
            response = await acompletion(
                model=self._model_name,
                messages=messages,
            )
            content = self._strip_code_fences(response.choices[0].message.content.strip())

            try:
                total_cost = LLMConfig.get_costs_for_model(self._model_name, response.usage.prompt_tokens, response.usage.completion_tokens)
            except ValueError as e:
                logger.error(f"Cost calculation failed: {e}")
                total_cost = 0.0
            logger.debug(
                "Tokens prompt=%s completion=%s cost=%s",
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                total_cost,
            )

            return ModelResult(
                content=content,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cost=total_cost
            )

        except Exception as err:
            raise LLMProcessingError(
                f"Post-process LLM call failed: {err}"
            ) from err
  
    
    def _calculate_tokens(self, model_name: str, content: str) -> int:
        """
        Calculate the number of tokens in the given content.

        Args:
            content (str): The content to calculate tokens for.

        Returns:
            int: The number of tokens.
        """
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # fallback if model not recognized
            encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(content))
        logger.debug("Calculated %s tokens for model %s", token_count, model_name)
        return token_count
