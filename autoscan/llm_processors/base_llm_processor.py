from abc import ABC, abstractmethod
from typing import Any, Dict, List
import os
import logging
from autoscan.utils.env import get_env_var_for_model
from autoscan.types import ModelResult
from litellm import cost_per_token
from litellm import acompletion
from autoscan.utils.llm import strip_code_fences
from autoscan.errors import LLMProcessingError

logger = logging.getLogger(__name__)

class BaseLLMProcessor(ABC):
    """
    Abstract base class for all LLM processors.
    
    Provides a common interface for LLM-based processing tasks.
    """

    def __init__(self, model_name: str, system_prompt: str, user_prompt: str, **kwargs: Any) -> None:
        """
        Initialize the LLM processor with a model name and optional parameters.
        
        :param model_name: The name of the LLM model to use.
        :param system_prompt: The system prompt to use for the LLM.
        :param user_prompt: The user prompt to use for the LLM.
        :param kwargs: Additional keyword arguments for specific configurations.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        if self.system_prompt is None:
            raise ValueError("system_prompt and user_prompt must be provided")

        self._validate_model(model_name)
        self._initialize_processor(**kwargs)

        logger.debug(f"Initialized {self.__class__.__name__} with model: {self.model_name} and parameters: {kwargs}")

    
    def _validate_model(self, model_name: str) -> None:
        """Validate the model name."""
        if not model_name or not isinstance(model_name, str):
            raise ValueError("Model name must be a non-empty string")
        
        env_var = get_env_var_for_model(model_name)
        if env_var and not os.environ.get(env_var):
            raise ValueError(f"{env_var} environment variable is not set.")
        

    @abstractmethod
    def _initialize_processor(self, **kwargs) -> None:
        """
        Initialize processor-specific parameters.
        Subclasses must implement this to handle their specific kwargs.
        
        Args:
            **kwargs: Processor-specific parameters
        """
        pass

    @abstractmethod
    async def acompletion(
        self,
        **kwargs: Any
    ) -> ModelResult:
        """
        Generic completion method that calls LiteLLM.

        Args:
            **kwargs: Additional parameters

        Returns:
            ModelResult: The LLM response object

        Raises:
            Exception: If the LLM call fails
        """
        pass

    def _calculate_cost(self, input_tokens: int, completion_tokens: int) -> float:
        """
        Calculate the cost of the LLM call based on input and completion tokens.
        
        Args:
            input_tokens (int): Number of input tokens
            completion_tokens (int): Number of completion tokens
        
        Returns:
            float: Total cost of the LLM call
        """
        try:
            prompt_cost, completion_cost = cost_per_token(
                model=self.model_name,
                prompt_tokens=input_tokens,
                completion_tokens=completion_tokens
            )
            return prompt_cost + completion_cost
        except Exception as e:
            raise ValueError(f"Error retrieving cost for model '{self.model_name}': {e}")


    async def _allm_call(
        self, 
        messages: List[Dict[str, Any]],
        is_strip_code_fences: bool = False,
    ) -> ModelResult:
        try:
            response = await acompletion(model=self.model_name, messages=messages)
            raw = response.choices[0].message.content
            content = strip_code_fences(raw) if is_strip_code_fences else raw
            usage = response.usage
            cost = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)

            logger.debug(
                f"✨ LLM response received - "
                f"tokens(in/out)={usage.prompt_tokens}/{usage.completion_tokens}, "
                f"cost=${cost:.4f}, "
                f"content_length={len(content)} chars"
            )

            return ModelResult(
                content=content,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost=cost,
            )
        except Exception as err:
            logger.error(f"🚨 LLM call failed - {err}")
            raise LLMProcessingError(f"Image-to-Markdown LLM call failed: {err}") from err



