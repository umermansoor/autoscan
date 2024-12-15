import os
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from .image_processing import image_to_base64
from .types import ModelCompletionResult
from .prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT_IMAGE_TRANSCRIPTION

class LlmModel:
    """
    A model class that converts a PDF page (provided as an image) into markdown
    using an LLM. It can maintain formatting consistency with previously processed pages.
    """

    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the LLM model interface.

        Args:
            model_name (str): The model name to use. Defaults to "gpt-4o".
        """
        self._model_name = model_name
        self._system_prompt = DEFAULT_SYSTEM_PROMPT
        self._system_prompt_image_transcription = DEFAULT_SYSTEM_PROMPT_IMAGE_TRANSCRIPTION
        if not "OPENAI_API_KEY" in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

    async def completion(self, image_path: str, transcribe_images = False) -> ModelCompletionResult:
        """
        Generate a markdown representation of a PDF page from an image.

        Args:
            image_path (str): Path to the image file of the PDF page.
            transcribe_images: Describes images in words within the markdown.

        Returns:
            ModelCompletionResult: The generated markdown and token usage details.
        """
        messages = self._get_messages(image_path=image_path, transcibe_images=transcribe_images)

        try:
            response = await self.client.chat.completions.create(
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

            # Extract required information and return a CompletionResult
            return ModelCompletionResult(
                page_markdown=content,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost=self.calculate_costs(usage.prompt_tokens, usage.completion_tokens)
            )
        except Exception as err:
            raise RuntimeError("Error: Unable to process request. Please try again later.") from err

    def _get_messages(self, image_path: str, transcibe_images = False) -> List[Dict[str, Any]]:
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

        system_prompt = self._system_prompt_image_transcription if transcibe_images else self._system_prompt

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        return messages
    
    def calculate_costs(self, prompt_tokens: int, completion_tokens: int) -> float:
        """
        Calculate the cost of the completion based on the token usage.

        Args:
            prompt_tokens (int): The number of tokens used in the prompt.
            completion_tokens (int): The number of tokens used in the completion.

        Returns:
            float: The cost of the completion.
        """
        
        input_token_costs_per_1k = 0.0
        completion_token_costs_per_1k = 0.0

        if self._model_name == "gpt-4o":
            input_token_costs_per_1k = 0.00250
            completion_token_costs_per_1k = 0.01000

        input_token_costs = (prompt_tokens / 1000) * input_token_costs_per_1k
        completion_token_costs = (completion_tokens / 1000) * completion_token_costs_per_1k

        return input_token_costs + completion_token_costs
