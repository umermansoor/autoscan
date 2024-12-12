import logging
import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from autoscan.image_processing import image_to_base64

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LlmModel:
    """
    A model class that converts a PDF page (provided as an image) into markdown
    using an LLM. It can maintain formatting consistency with previously processed pages.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "Your job is to convert the following PDF page to markdown. "
        "You must convert the entire page to markdown including all text, tables, etc. "
        "If you encounter any images, you must describe them in the markdown. "
        "Only return the markdown with no explanation."
    )

    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the LLM model interface.

        Args:
            model_name (str): The model name to use. Defaults to "gpt-4o".
        """
        self._model_name = model_name
        self._system_prompt = self.DEFAULT_SYSTEM_PROMPT
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    def completion(self, image_path: str, prior_page: Optional[str] = "") -> str:
        """
        Generate a markdown representation of a PDF page from an image.

        Args:
            image_path (str): Path to the image file of the PDF page.
            prior_page (str, optional): The markdown of a previously processed page for formatting consistency.

        Returns:
            str: The generated markdown.
        """
        messages = self._get_messages(image_path=image_path, prior_page=prior_page)

        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception as err:
            logger.exception("Error while processing LLM request.")
            raise RuntimeError("Error: Unable to process request. Please try again later.") from err

    def _get_messages(self, image_path: str, prior_page: Optional[str]) -> List[Dict[str, Any]]:
        """
        Construct the message payload for the LLM.

        Args:
            image_path (str): The image path of the PDF page.
            prior_page (str, optional): The previously processed page's markdown, if any.

        Returns:
            List[Dict[str, Any]]: A list of message objects for the LLM API.
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

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]

        # If a prior page is provided, you can use it to maintain formatting consistency.
        if prior_page:
            # Insert a system-level message guiding the model to maintain formatting consistency.
            formatting_message = (
                "Below is previously processed markdown. Maintain similar formatting:\n\n"
                f'"""{prior_page}"""'
            )
            messages.insert(1, {"role": "system", "content": formatting_message})

        return messages
