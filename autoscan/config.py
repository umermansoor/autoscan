from typing import Dict
from litellm import model_cost, get_max_tokens, cost_per_token

class PDF2ImageConversionConfig:
    NUM_THREADS = 4
    DPI = 300
    FORMAT = "png"
    USE_PDFTOCAIRO = True
    SIZE = (None, 1535)

class LLMConfig:
    DEFAULT_MODEL = "openai/gpt-4o"

    @classmethod
    def get_costs_for_model(cls, model: str, input_tokens: int, completion_tokens: int) -> float:
        """
        Calculate the total cost for a given model based on input and completion tokens.
        """
        try:
            prompt_cost, completion_cost = cost_per_token(
                model=model,
                prompt_tokens=input_tokens,
                completion_tokens=completion_tokens
            )
            return prompt_cost + completion_cost
        except Exception as e:
            raise ValueError(f"Error retrieving cost for model '{model}': {e}")

    @classmethod
    def get_max_tokens_for_model(cls, model: str) -> Dict[str, int]:
        """
        Retrieve the maximum input and output tokens allowed for the specified model.
        """
        try:
            # Fetch the model cost details which includes max_tokens
            model_info = model_cost.get(model)
            if not model_info:
                raise ValueError(f"Model '{model}' not found in model cost data.")

            max_input_tokens = model_info.get("max_input_tokens")
            max_output_tokens = model_info.get("max_output_tokens")

            if max_input_tokens is None or max_output_tokens is None:
                # Fallback to get_max_tokens if specific values are not available
                max_tokens = get_max_tokens(model)
                # Assuming equal split if specific input/output limits are not provided
                max_input_tokens = max_output_tokens = max_tokens // 2

            return {
                "input_tokens": int(max_input_tokens),
                "output_tokens": int(max_output_tokens)
            }
        except Exception as e:
            raise ValueError(f"Error retrieving max tokens for model '{model}': {e}")
