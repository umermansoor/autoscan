# AutoScan

AutoScan converts PDF files into Markdown using LLMs (e.g. GPT or Gemini). It is designed for smaller yet complex documents that require high fidelity—for example, medical documents, invoices, or other official forms. The resulting Markdown can then be fed into another LLM for downstream processing. 

When perfect accuracy is not essential, faster and cheaper alternatives (e.g. [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)) may be more suitable. Using higher accuracy modes in AutoScan will take more time and use more tokens (increasing cost).

## Features

- **High accuracy** conversion of complex PDFs to Markdown, preserving tables and layout.
- **Image transcription** so visuals are described in text rather than embedded.
- **Handwriting OCR** when handwritten notes are present.
- **Multi-language** support.
- **Optional user instructions** passed directly to the LLM for custom behavior.
- **Works with any LLM** via [LiteLLM](https://github.com/BerriAI/litellm).

![Example 1](assets/pdf_to_md_eg_1.png)
![Example 2](assets/pdf_to_md_eg_2.png)

## Installation

### 1. Install Python dependencies

```bash
poetry install
```

### 2. Install `poppler`

AutoScan uses [Poppler](https://poppler.freedesktop.org/) to convert PDF pages to images.

**macOS**:

```sh
brew install poppler
```

**Linux**:

```sh
sudo apt-get install poppler-utils
```

**Windows**:  
You can use a package manager like [Chocolatey](https://chocolatey.org/) or [Scoop](https://scoop.sh/):


```sh
choco install poppler
``` 

```sh
scoop install poppler
```

### 3. Set your API key

Depending on the model provider you choose, set the appropriate environment variable:

- **OpenAI**: `OPENAI_API_KEY`
- **Anthropic (Claude)**: `ANTHROPIC_API_KEY`
- **Google Gemini**: `GOOGLE_API_KEY` or `GEMINI_API_KEY` if using Google AI Studio (e.g.   `gemini/gemini-2.0-flash`)

**macOS/Linux example**:

`export OPENAI_API_KEY=your_api_key`

**Windows example**:

`$env:OPENAI_API_KEY="your_api_key"`

Replace `your_api_key` with your actual provider API key.
AutoScan will check for the appropriate variable based on the
provider prefix in the `--model` option. For example, if you use
`--model anthropic/claude-3-sonnet-20240229`, ensure
`ANTHROPIC_API_KEY` is defined.

## **Usage**

After installing dependencies (e.g. `poetry install`), run `autoscan` from the command line:

```sh
autoscan path/to/your/file.pdf

# Choose accuracy level (low or high):
autoscan --accuracy high path/to/your/file.pdf

# Specify a model (default is `openai/gpt-4o`):
autoscan --model gemini/gemini-2.0-flash path/to/your/file.pdf

# Provide additional instructions for the LLM:
autoscan --instructions "This is an invoice; use GitHub tables" path/to/your/file.pdf
```

### Accuracy Levels

Two accuracy levels are available: `low` and `high`. `low` processes pages in parallel, while `high` processes them sequentially. In `high` mode the entire previous page is returned to the model so it can rewrite that page together with the current one for consistent formatting. The model responds in JSON with `page_1` and `page_2` fields so AutoScan can replace the prior page with the updated version. This increases token usage and runtime.

### Programmatic Example

You can also invoke AutoScan in your Python code:

```python
import asyncio
from autoscan.autoscan import autoscan

async def main():
    pdf_path = "path/to/your/pdf_file.pdf"
    output = await autoscan(pdf_path)
    print(output.markdown)

asyncio.run(main())
```

## How It Works

1. **Convert PDF to Images**: Each page of the PDF is converted into an image.
2. **Process Images with LLM**: The images are processed by the LLM to generate Markdown.
3. **Aggregate Markdown**: All Markdown output is combined into one file using a simple algorithm.

## Configuration

Configure models and other parameters using the `autoscan` function signature:

```python
async def autoscan(
    pdf_path: str,
    model_name: str = "openai/gpt-4o",
    accuracy: str = "low",
    user_instructions: Optional[str] = None,
    temp_dir: Optional[str] = None,
    cleanup_temp: bool = True,
    concurrency: Optional[int] = 10,
) -> AutoScanOutput:
```

## Output

The `autoscan` function returns an object with the following attributes:

* **completion_time**: Time taken to complete the conversion.  
* **markdown_file**: Path to the generated Markdown file.  
* **markdown**: The aggregated Markdown content.  
* **input_tokens**: Number of input tokens used.  
* **output_tokens**: Number of output tokens generated.  
* **accuracy**: The accuracy level used.

## Examples

Sample PDFs are available in the `examples` directory for testing and demonstration.

E.g. 

```sh
autoscan --accuracy high --model gemini/gemini-2.0-flash --log-level DEBUG ./examples/table.pdf
```

## Testing

To run the test suite:

```sh
poetry run pytest tests/
```