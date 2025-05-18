# AutoScan

AutoScan converts PDF files into Markdown using AI. It is designed for small but complex documents that require very high fidelity—for example, detailed lab reports, invoices or other official forms. The resulting Markdown can then be fed into another large language model for downstream processing. When perfect accuracy is not essential, faster and cheaper alternatives may be more suitable. Using higher accuracy modes will also take more time and consume more tokens (increasing cost).

## Features

- **High accuracy** conversion of complex PDFs to Markdown, preserving tables and layout.
- **Image transcription** so visuals are described in text rather than embedded.
- **Handwriting OCR** when handwritten notes are present.
- **Multi-language** support.
- **Metadata extraction** for titles, authors and more.
- **Works with any LLM** through the [LiteLLM](https://github.com/BerriAI/litellm) library.

![Example 1 ](https://private-user-images.githubusercontent.com/862952/395720191-296f93c4-8f04-4771-887c-08c45fdd1d95.png)

![Example 2](https://private-user-images.githubusercontent.com/862952/395720236-44d3ea28-2ca8-4d86-ab79-29683e5529c1.png)

## Installation

To install the required dependencies, run:

```sh
poetry install
```

### Install `poppler`

#### On Mac

```sh
brew install poppler
```

#### On Linux: 

```sh
sudo apt-get install poppler-utils
```

### To Test

To run tests:

```sh
poetry run pytest tests/
```

## Set `OPENAI_API_KEY`

### On Mac/Linux

```sh
export OPENAI_API_KEY=your_api_key
```

## Usage

After installing the dependencies (for example with `poetry install`), you can
run AutoScan directly from the command line:

```sh
autoscan path/to/your/file.pdf

# To process all PDFs in a directory:
autoscan --directory path/to/your/pdf_directory

# Choose accuracy level (low, medium, high):
autoscan --accuracy high path/to/your/file.pdf

# Enable litellm debug logging to see the raw prompts:
autoscan --debug path/to/your/file.pdf
```

### Accuracy Levels

AutoScan supports `low`, `medium` and `high` accuracy settings. Higher accuracy processes pages sequentially and performs an additional review step. This leads to more tokens being sent to the LLM (higher cost) and increases runtime.

### Example

Here is an example of how to use AutoScan programmatically:

```python
import asyncio
from autoscan.autoscan import autoscan

async def main():
    pdf_path = "path/to/your/pdf_file.pdf"
    output = await autoscan(pdf_path, debug=True)
    print(output.markdown)

asyncio.run(main())
```

## How It Works

1. Convert PDF to Images: Each page of the PDF is converted into an image.
2. Process Images with LLM: The images are processed by a language model to generate Markdown.
3. Aggregate Markdown: The generated Markdown for each page is aggregated into a single file.

## Configuration
You can configure the model and other parameters in the `autoscan` function:

```python
async def autoscan(
    pdf_path: str,
    model_name: str = "openai/gpt-4o",
    accuracy: str = "medium",
    transcribe_images: Optional[bool] = True,
    output_dir: Optional[str] = None,
    temp_dir: Optional[str] = None,
    cleanup_temp: bool = True,
    concurrency: Optional[int] = 10,
    debug: bool = False,
) -> AutoScanOutput:
    ...
```
Set `debug=True` to print the exact prompts being sent to the LLM.

## Output
The output of the `autoscan` function includes:

- `completion_time`: Time taken to complete the conversion.
- `markdown_file`: Path to the generated Markdown file.
- `markdown`: The aggregated Markdown content.
- `input_tokens`: Number of input tokens used.
- `output_tokens`: Number of output tokens generated.
- `accuracy`: The accuracy level used for processing.

## Examples

Sample PDFs for testing are available in the `examples` directory.



