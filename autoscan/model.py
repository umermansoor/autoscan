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
    A model class that converts a PDF page (provided as an image) into Markdown
    using an LLM. It can maintain formatting consistency with previously 
    processed pages.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-4o",
        accuracy: str = "medium"
    ):
        """
        Initialize the LLM model interface.

        Args:
            model_name (str): The model name to use. Defaults to "openai/gpt-4o".
            accuracy (str): An accuracy level descriptor. Defaults to "medium".
        """
        self._model_name = model_name
        self._accuracy = accuracy
        self._system_prompt = DEFAULT_SYSTEM_PROMPT
        ensure_env_for_model(model_name)

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        """
        Remove enclosing triple backticks and optional language tags if the 
        entire string is fenced. 
        """
        content = content.strip()
        if content.startswith("```") and content.endswith("```"):
            content = content.removeprefix("```").removesuffix("```").strip()
            for lang_tag in ("markdown", "md"):
                if content.startswith(lang_tag):
                    content = content[len(lang_tag):]
                    break
            content = content.strip()
        return content

    @property
    def accuracy(self) -> str:
        """
        Return the accuracy level for this model.
        """
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

    def _maybe_log_debug_messages(self, messages: List[Dict[str, Any]]) -> None:
        """
        Log messages at DEBUG level if the logger is configured accordingly.

        Args:
            messages (List[Dict[str, Any]]): The messages to log.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return

        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content")
            logger.debug("%s:", role)
            # The content can be a string or a list of content items.
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        logger.debug(item.get("text"))
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image"):
                            base64_str = url.split(",", 1)[1]
                            preview = f"{base64_str[:100]}...{base64_str[-10:]}"
                            logger.debug("[IMAGE BASE64] %s", preview)
                        else:
                            logger.debug("[IMAGE URL] %s", url)
            else:
                logger.debug(content)

    def _calculate_cost(self, usage) -> float:
        """
        Safely calculate the cost for the given usage object. 
        If cost calculation fails, return 0.0 and log an error.

        Args:
            usage (Any): The usage object returned by the LLM API.

        Returns:
            float: The calculated cost.
        """
        try:
            return LLMConfig.get_costs_for_model(
                self._model_name, 
                usage.prompt_tokens, 
                usage.completion_tokens
            )
        except (ValueError, AttributeError) as e:
            logger.error(f"Cost calculation failed: {e}")
            return 0.0

    def _calculate_tokens(self, model_name: str, content: str) -> int:
        """
        Calculate the number of tokens in the given content for the specified model.

        Args:
            model_name (str): The name of the model to use for tokenization.
            content (str): The text content to calculate tokens for.

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

    async def image_to_markdown(
        self,
        image_path: str,
        previous_page_markdown: Optional[str] = None,
        user_instructions: Optional[str] = None,
    ) -> ModelResult:
        """
        Generate a Markdown representation of a PDF page from an image.

        Args:
            image_path (str): Path to the image file of the PDF page.
            previous_page_markdown (Optional[str]): Markdown of the previous page 
                                                    (for formatting context).
            user_instructions (Optional[str]): Additional instructions from the user.

        Returns:
            ModelResult: The generated Markdown and token usage details.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path does not exist: {image_path}")

        logger.debug(f"Converting image {image_path} to markdown")
        try:
            base64_image = image_to_base64(image_path)
        except Exception as e:
            raise LLMProcessingError(
                f"Failed to convert image to base64: {str(e)}"
            ) from e

        user_content = [
            {
                "type": "text",
                "text": (
                    "Convert the following image to markdown."
                )
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]

        # Provide a snippet of the previous page for context if available.
        if previous_page_markdown:
            user_content.append({
                "type": "text",
                "text": (
                    "Here are the last few characters in Markdown format from the previous page to provide context.\n"
                    "The final output has no page breaks. "
                    "For consistency, use the same style.\n"
                    f"<!-- PAGE SEPARATOR -->\n{previous_page_markdown[-100:]}"
                ),
            })

        # If there are additional user instructions, include them.
        if user_instructions:
            user_content.append({"type": "text", "text": user_instructions})

        system_prompt = self._system_prompt

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # Log messages if DEBUG level is enabled
        self._maybe_log_debug_messages(messages)

        try:
            response = await acompletion(
                model=self._model_name,
                messages=messages,
            )

            # Extract relevant parts of the LLM response
            raw_content = response.choices[0].message.content.strip()
            content = self._strip_code_fences(raw_content)

            usage = response.usage
            total_cost = self._calculate_cost(usage)

            # Log the assistant's response if DEBUG level is enabled
            self._maybe_log_debug_messages([
                {"role": "assistant", "content": content}
            ])

            logger.debug(
                "Tokens prompt=%s completion=%s cost=%s",
                usage.prompt_tokens,
                usage.completion_tokens,
                total_cost
            )

            return ModelResult(
                content=content,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost=total_cost
            )
        except Exception as err: 
            raise LLMProcessingError(
                f"Image to Markdown LLM call failed: {err}"
            ) from err

    async def postprocess_markdown(self, markdowns: List[str]) -> ModelResult:
        """
        Combine and finalize Markdown chunks. Optionally clean or optimize 
        formatting, headings, etc.

        Args:
            markdowns (List[str]): List of individual Markdown strings to combine.

        Returns:
            ModelResult: The final combined Markdown and token usage details.
        """
        logger.debug("Post-processing %s markdown chunks", len(markdowns))

        combined_markdown = "\n".join(markdowns)
        input_tokens = self._calculate_tokens(self._model_name, combined_markdown)
        max_output_tokens = LLMConfig.get_max_tokens_for_model(self._model_name)["output_tokens"]
        if input_tokens >= max_output_tokens:
            raise RuntimeError("Too many tokens to post-process.")

        separator = "\n---PAGE BREAK---\n"
        user_content = separator.join(markdowns)

        messages = [
            {"role": "system", "content": FINAL_REVIEW_PROMPT},
            {"role": "user", "content": user_content}
        ]

        # Log messages if DEBUG level is enabled
        self._maybe_log_debug_messages(messages)

        try:
            response = await acompletion(
                model=self._model_name,
                messages=messages,
            )

            raw_content = response.choices[0].message.content.strip()
            content = self._strip_code_fences(raw_content)

            # Log the final content if DEBUG level is enabled
            self._maybe_log_debug_messages([
                {"role": "assistant", "content": content}
            ])

            usage = response.usage
            total_cost = self._calculate_cost(usage)

            return ModelResult(
                content=content,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost=total_cost
            )

        except Exception as err:
            raise LLMProcessingError(
                f"Post-processing LLM call failed: {err}"
            ) from err