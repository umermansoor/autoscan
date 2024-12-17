from typing import Tuple

class PDF2ImageConversionConfig:
    NUM_THREADS = 4
    DPI = 300
    FORMAT = "png"
    USE_PDFTOCAIRO = True
    SIZE = (None, 1535)


class LLMConfig:
    DEFAULT_MODEL = "gpt-4o"
    MODEL_PROPERTIES = {
        "gpt-4o": {
            "max_output_tokens": 16384,
            "max_input_tokens": 128000,
            "input_costs_per_1k_tokens": 0.00250,
            "completion_per_1k_tokens": 0.01000
        }
    }

    @classmethod
    def get_costs_for_model(cls, model: str, input_tokens: int, completion_tokens: int) -> float:
        model_properties = cls.MODEL_PROPERTIES.get(model)
        if not model_properties:
            raise ValueError(f"Model '{model}' not found in model properties.")

        input_costs = (input_tokens / 1000) * model_properties["input_costs_per_1k_tokens"]
        completion_costs = (completion_tokens / 1000) * model_properties["completion_per_1k_tokens"]

        return input_costs + completion_costs
    
    @classmethod
    def get_max_tokens_for_model(cls, model: str) -> dict:
        model_properties = cls.MODEL_PROPERTIES.get(model)
        if not model_properties:
            raise ValueError(f"Model '{model}' not found in model properties.")

        return {
            "input_tokens": int(model_properties["max_input_tokens"]),
            "output_tokens": int(model_properties["max_output_tokens"])
    }

    




