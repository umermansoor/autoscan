from dataclasses import dataclass

@dataclass
class AutoScanOutput:
    completion_time: float
    markdown_file: str
    markdown: str
    input_tokens: int
    output_tokens: int
    contextual_conversion: bool


@dataclass
class ModelResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
