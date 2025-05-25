import os
import logging
from typing import List, Dict, Any, Optional
import datetime

import litellm
from litellm import acompletion

from .image_processing import image_to_base64
from .types import ModelResult
from .prompts import DEFAULT_SYSTEM_PROMPT
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
        accuracy: str = "medium",
        save_llm_calls: bool = False
    ):
        """
        Initialize the LLM model interface.

        Args:
            model_name (str): The model name to use. Defaults to "openai/gpt-4o".
            accuracy (str): An accuracy level descriptor. Defaults to "medium".
            save_llm_calls (bool): Whether to save LLM calls to a file. Defaults to False.
        """
        self._model_name = model_name
        self._accuracy = accuracy
        self._system_prompt = DEFAULT_SYSTEM_PROMPT
        self._save_llm_calls = save_llm_calls
        self._log_file_path = None  # Will be set when first log call is made
        ensure_env_for_model(model_name)

    @staticmethod
    def _strip_code_fences(content: str) -> str:
        """
        Remove enclosing triple backticks and optional language tags if the 
        entire string is fenced. Preserves internal whitespace/indentation.
        """
        content = content.rstrip()
        if content.startswith("```") and content.endswith("```"):
            # Remove opening and closing code fences
            content = content.removeprefix("```").removesuffix("```")
            
            # Remove trailing whitespace only
            content = content.rstrip()
            
            # Check for language tags at the beginning and remove them
            for lang_tag in ("markdown", "md"):
                if content.startswith(lang_tag):
                    content = content[len(lang_tag):]
                    # Only strip leading whitespace from the language tag line, preserve content indentation
                    content = content.lstrip()
                    break
            else:
                # If no language tag found, only strip leading newlines/whitespace from the very beginning
                content = content.lstrip('\n\r\t ')
       
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

    def _log_llm_call_to_file(
        self,
        page_number: Optional[int],
        system_prompt: str,
        user_prompt_content: List[Dict[str, Any]],
        response_or_error: str,
        is_error: bool = False
    ) -> None:
        """
        Log the LLM call details to a structured file.
        """
    
        if not self._save_llm_calls:
            return

        # Create timestamped log file path if not already set
        if self._log_file_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"llm_calls_{self._accuracy}_{timestamp}.txt"
            self._log_file_path = os.path.join(os.getcwd(), "logs", log_filename)
            os.makedirs(os.path.dirname(self._log_file_path), exist_ok=True)
            
            # Write header to new file
            header = f"""=== AUTOSCAN LLM CALL LOG ===
Accuracy Mode: {self._accuracy.upper()}
Model: {self._model_name}
Log Created: {datetime.datetime.now().isoformat()}
================================

"""
            with open(self._log_file_path, "w", encoding="utf-8") as f:
                f.write(header)

        timestamp = datetime.datetime.now().isoformat()
        page_num_str = f"Page {page_number}" if page_number is not None else "Page N/A"

        # Construct formatted user prompt
        user_prompt_lines = []
        image_index_in_prompt = 0
        for item in user_prompt_content:
            if item.get("type") == "text":
                user_prompt_lines.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                image_index_in_prompt += 1
                image_descriptor = "Current Page Image"
                if image_index_in_prompt == 1:
                    image_descriptor = "Current Page Image"
                elif image_index_in_prompt == 2:
                    image_descriptor = "Previous Page Image"
                
                url_data = item.get("image_url", {}).get("url", "")
                if isinstance(url_data, str) and url_data.startswith("data:image"):
                    base64_str = url_data.split(",", 1)[-1]
                    user_prompt_lines.append(f"[{image_descriptor} (last 50 chars): ...{base64_str[-50:]}]")
                else:
                    user_prompt_lines.append(f"[{image_descriptor} URL: {url_data}]")
        formatted_user_prompt = "\n".join(user_prompt_lines)

        log_lines = [
            f"Timestamp: {timestamp}",
            f"Page Number: {page_num_str}",
            "System Prompt:",
            system_prompt,
            "User Prompt:",
            formatted_user_prompt,
            f"{'Error' if is_error else '\nAssistant Response'}:",
            response_or_error,
            "\n\n----------------------END LLM INTERACTION---------------------------\n\n"
        ]
        log_entry = "\n".join(log_lines) + "\n"

        try:
            with open(self._log_file_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            logger.error(f"Failed to write LLM call to log file: {e}")
            
    def get_log_file_path(self) -> Optional[str]:
        """Return the path to the current log file."""
        return self._log_file_path

    def _calculate_cost(self, usage) -> float:
        """
        Safely calculate the cost for the given usage object. If cost calculation fails, return 0.0 and log an error.

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
            # Fall back to default encoding if model not recognized
            encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(content))
        logger.debug("Calculated %s tokens for model %s", token_count, model_name)
        return token_count

    def _get_last_n_tokens(self, text: str, n: int) -> str:
        """Return the last ``n`` tokens from ``text`` using the model's tokenizer."""
        try:
            encoding = tiktoken.encoding_for_model(self._model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(text)
        last_tokens = tokens[-n:]
        return encoding.decode(last_tokens)
    
    async def image_to_markdown(
        self,
        image_path: str,
        previous_page_markdown: Optional[str] = None,
        user_instructions: Optional[str] = None,
        previous_page_image_path: Optional[str] = None,
        page_number: Optional[int] = None,
    ) -> ModelResult:
        """Generate a Markdown representation of a PDF page from an image."""
        page_str = f"Page {page_number}" if page_number is not None else "Unknown page"
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image path does not exist: {image_path}")

        logger.debug(f"🖼️  {page_str}: Converting image to base64: {os.path.basename(image_path)}")
        try:
            base64_image = image_to_base64(image_path)
            logger.debug(f"📁 {page_str}: Image encoded to base64 ({len(base64_image)} chars)")
        except Exception as e:
            raise LLMProcessingError(f"Failed to convert image to base64: {e}") from e

        # ---- 1. current-page prompt ----------------------------------------
        user_content: List[Dict[str, Any]] = [
            {"type": "text", "text": "Convert the following image to markdown."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            },
        ]

        # ---- 2. previous-page context (if any) ------------------------------
        if previous_page_markdown and self._accuracy == "high":
            logger.debug(f"🔗 {page_str}: Adding previous page context ({len(previous_page_markdown)} chars)")
            
            # High accuracy: Full previous page markdown + previous page image
            context_md = previous_page_markdown
            intro = (
                "Here is the previous page (image + markdown) so you can "
                "maintain style consistency. Do NOT re-emit that content."
            )
            
            # 2b. previous-page image (only in high accuracy mode)
            if previous_page_image_path:
                logger.debug(f"🖼️  {page_str}: Adding previous page image: {os.path.basename(previous_page_image_path)}")
                if previous_page_image_path and not os.path.exists(previous_page_image_path):
                    raise FileNotFoundError(f"Previous page image does not exist: {previous_page_image_path}")
                try:
                    base64_prev = image_to_base64(previous_page_image_path)
                except Exception as e:
                    raise LLMProcessingError(
                        f"Failed to convert previous page image to base64: {e}"
                    ) from e

                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_prev}"}
                })
                user_content.append({
                    "type": "text",
                    "text": "This is the image of the previous page."
                })

            # Add the context markdown
            user_content.append({
                "type": "text",
                "text": f"{intro}\n<!-- PAGE SEPARATOR -->\n{context_md}"
            })
        elif previous_page_markdown and self._accuracy != "high":
            # Low/Medium accuracy modes use concurrent processing and should not receive context
            # This indicates a logic error in the calling code
            logger.warning(f"⚠️  {page_str}: Ignoring previous page context in {self._accuracy} accuracy mode (concurrent processing)")
            logger.debug(f"🚫 {page_str}: Previous page context is only used in high accuracy mode (sequential processing)")

        # ---- 3. any ad-hoc user instructions --------------------------------
        if user_instructions:
            logger.debug(f"📝 {page_str}: Adding user instructions ({len(user_instructions)} chars)")
            user_content.append({"type": "text", "text": user_instructions})

        # ---- 4. construct & send messages -----------------------------------
        system_prompt = self._system_prompt
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Log the request details
        page_str = f"Page {page_number}" if page_number is not None else "Unknown page"
        logger.debug(f"🔍 {page_str}: Sending request to {self._model_name}")
        
        try:
            response = await acompletion(model=self._model_name, messages=messages)
            # Preserve internal whitespace, only strip trailing
            raw = response.choices[0].message.content.rstrip()
            content = self._strip_code_fences(raw)
            usage = response.usage
            cost = self._calculate_cost(usage)

            # Enhanced logging with page information
            logger.debug(
                f"✨ {page_str}: LLM response received - "
                f"tokens(in/out)={usage.prompt_tokens}/{usage.completion_tokens}, "
                f"cost=${cost:.4f}, "
                f"content_length={len(content)} chars"
            )

            if self._save_llm_calls:
                self._log_llm_call_to_file(
                    page_number, system_prompt, user_content, content
                )

            return ModelResult(
                content=content,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost=cost,
            )

        except Exception as err:
            logger.error(f"🚨 {page_str}: LLM call failed - {err}")
            if self._save_llm_calls:
                self._log_llm_call_to_file(
                    page_number, system_prompt, user_content, str(err), is_error=True
                )
            raise LLMProcessingError(f"Image-to-Markdown LLM call failed: {err}") from err

