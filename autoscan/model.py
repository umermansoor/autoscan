import os
from typing import List, Dict, Any, Optional

import litellm
from litellm import acompletion

from .image_processing import image_to_base64
from .types import ModelResult
from .prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT_IMAGE_TRANSCRIPTION, FINAL_REVIEW_PROMPT
from .config import LLMConfig
from .utils.env import ensure_env_for_model
import tiktoken

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
        self._system_prompt_image_transcription = DEFAULT_SYSTEM_PROMPT_IMAGE_TRANSCRIPTION
        ensure_env_for_model(model_name)

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
            transcribe_images=False,
            previous_page_markdown: Optional[str] = None,
    ) -> ModelResult:
        """
        Generate a markdown representation of a PDF page from an image.

        Args:
            image_path (str): Path to the image file of the PDF page.
            transcribe_images: Describes images in words within the markdown.
            previous_page_markdown: Optional markdown of previous page in PDF file
        Returns:
            ModelCompletionResult: The generated markdown and token usage details.
        """

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
                "text": f"Here are the last 50 characters in Markdown format from the previous page to provide you context. "
                        f"The final output has no page breaks."
                        f"\n<!-- PAGE SEPARATOR -->\n{previous_page_markdown[-100:]}"
            })

        system_prompt = self._system_prompt_image_transcription if transcribe_images else self._system_prompt

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        if self._debug:
            system_prompt = system_prompt
            print(f"System prompt: {system_prompt}")
            for item in user_content:
                if item["type"] == "text":
                    print(f"User content: {item['text']}")

        try:
            response = await acompletion(
                model=self._model_name,
                messages=messages,
            )
            content = response.choices[0].message.content.strip()
            usage = response.usage  # Extract token usage

            if content.startswith("```") and content.endswith("```"):
                # Remove leading and trailing triple backticks
                content = content.removeprefix("```").removesuffix("```")
                
                # Remove optional language specifiers at the start
                for lang_tag in ("markdown", "md"):
                    if content.startswith(lang_tag):
                        content = content[len(lang_tag):]
                        break
                
                # Clean up leading/trailing whitespace
                content = content.strip()
    
            try:
                total_cost = LLMConfig.get_costs_for_model(self._model_name, usage.prompt_tokens, usage.completion_tokens)
            except ValueError:
                total_cost = 0.0

            # Extract required information and return a CompletionResult
            return ModelResult(
                content=content,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost=total_cost
            )
        except Exception as err:
            raise RuntimeError("Error: Image to markdown LLM call failed") from err


    async def postprocess_markdown(self, markdowns: List[str]) -> ModelResult:

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
            print(f"System prompt: {FINAL_REVIEW_PROMPT}")
            print(f"User content: {user_content}")

        try:
            response = await acompletion(
                model=self._model_name,
                messages=messages,
            )
            content = response.choices[0].message.content.strip()

            # Remove enclosing triple backticks if present
            if content.startswith("```") and content.endswith("```"):
                content = content.removeprefix("```").removesuffix("```").strip()

                # Remove optional language specifiers
                for lang_tag in ("markdown", "md"):
                    if content.startswith(lang_tag):
                        content = content[len(lang_tag):].strip()
                        break

            try:
                total_cost = LLMConfig.get_costs_for_model(self._model_name, response.usage.prompt_tokens, response.usage.completion_tokens)
            except ValueError:
                total_cost = 0.0

            return ModelResult(
                content=content,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cost=total_cost
            )

        except Exception as err:
            raise RuntimeError("Error: Post-process LLM call failed.") from err
  
    
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

        return len(encoding.encode(content))
