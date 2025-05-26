from typing import Dict
from litellm import model_cost, get_max_tokens, cost_per_token

class PDF2ImageConversionConfig:
    NUM_THREADS = 4
    FORMAT = "png"
    USE_PDFTOCAIRO = True
    
    # DPI settings by accuracy level
    DPI_HIGH = 200     # Best quality for complex documents, tables, handwriting
    DPI_LOW = 150      # Good quality, fastest processing, lower costs
    
    @classmethod
    def get_dpi_for_accuracy(cls, accuracy: str) -> int:
        """Get the appropriate DPI setting for the given accuracy level."""
        if accuracy == "high":
            return cls.DPI_HIGH
        else:  # "low", "medium"
            return cls.DPI_LOW

class LLMConfig:
    DEFAULT_MODEL = "openai/gpt-4o"

    # DELETE
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
