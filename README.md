# AutoScan

AutoScan converts PDF files into Markdown using LLMs (e.g., GPT or Gemini). It is designed for smaller yet complex documents that require high fidelity—for example, medical documents, invoices, or other official forms. The resulting Markdown can then be fed into another LLM for downstream processing. 

When perfect accuracy is not essential, faster and cheaper alternatives (e.g. [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)) may be more suitable. Using higher accuracy modes in AutoScan will take more time and use more tokens (increasing cost).

## Features

- **High accuracy** conversion of complex PDFs to Markdown, preserving tables and layout.
- **Image transcription** so visuals are described in text rather than embedded.
- **Handwriting OCR** when handwritten notes are present.
- **Multi-language** support.
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

### 3. Set your `OPENAI_API_KEY`

**macOS/Linux**:

`export OPENAI_API_KEY=your_api_key`

**Windows**:

`$env:OPENAI_API_KEY="your_api_key"`

Replace `your_api_key` with your actual OpenAI API key.

## **Usage**

After installing dependencies (e.g. `poetry install`), run `autoscan` from the command line:

```sh
autoscan path/to/your/file.pdf

# Choose accuracy level (low, medium, high):
autoscan --accuracy high path/to/your/file.pdf
```

### **Accuracy Levels**

`low`, `medium`, and `high` are supported. Higher accuracy processes pages sequentially and performs an additional review step, which increases token usage (cost) and runtime.

### **Programmatic Example**

You can also invoke AutoScan in your Python code:

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

1. **Convert PDF to Images**: Each page of the PDF is converted into an image.
2. **Process Images with LLM**: The images are processed by the LLM to generate Markdown.
3. **Aggregate Markdown**: All Markdown output is combined into one file (on `accuracy==low` markdowns are combined together using a simple algorithm; on `accuracy==medium,high` an LLM is used to combined all outputs together)

## **Configuration**

Configure models and other parameters using the `autoscan` function signature:

```python
async def autoscan(
    pdf_path: str,
    model_name: str = "openai/gpt-4o",
    accuracy: str = "medium",
    transcribe_images: Optional[bool] = True,
    temp_dir: Optional[str] = None,
    cleanup_temp: bool = True,
    concurrency: Optional[int] = 10,
    debug: bool = False,
) -> AutoScanOutput:
```

Set `debug=True` to print the exact prompts sent to the LLM.

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

## Testing

To run the test suite:

```sh
poetry run pytest tests/
```