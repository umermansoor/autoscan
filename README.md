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
pip install -r requirements.txt
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
- **Google Gemini**: `GEMINI_API_KEY` (for `gemini/` models) or `GOOGLE_API_KEY` (for `google/` models)

**macOS/Linux example**:

`export OPENAI_API_KEY=your_api_key`

**Windows example**:

`$env:OPENAI_API_KEY="your_api_key"`

Replace `your_api_key` with your actual provider API key.
AutoScan will check for the appropriate variable based on the
provider prefix in the `--model` option. For example, if you use
`--model anthropic/claude-3-sonnet-20240229`, ensure
`ANTHROPIC_API_KEY` is defined.

#### Setting up Google Gemini

To use Google Gemini models, you'll need to get an API key from Google AI Studio:

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key or use an existing one
3. Set the environment variable:

**macOS/Linux**:
```bash
export GEMINI_API_KEY=your_gemini_api_key
```

**Windows**:
```powershell
$env:GEMINI_API_KEY="your_gemini_api_key"
```

Gemini models are currently available for free with generous rate limits, making them an excellent choice for cost-effective high-quality PDF conversion.

## **Usage**

After installing dependencies (e.g. `pip install -r requirements.txt`), run `autoscan` from the command line:

```sh
autoscan path/to/your/file.pdf

# Choose accuracy level (low, medium, high):
autoscan --accuracy high path/to/your/file.pdf

# Specify a model (default is `openai/gpt-4o`):
autoscan --model gemini/gemini-2.0-flash path/to/your/file.pdf

# Provide additional instructions for the LLM:
autoscan --instructions "This is an invoice; use GitHub tables" path/to/your/file.pdf
```

### Accuracy Levels

- **`low`** and **`medium`**: Pages processed concurrently (faster). Pages are processed independently without previous page context for maximum speed.
- **`high`**: Pages processed sequentially (slower but more accurate). The entire previous page markdown AND the previous page image are sent as context, which increases token usage (cost) and runtime but provides better formatting consistency.

Note: `medium` is treated the same as `low` for backwards compatibility.

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

## Model Performance Comparison

### Model Recommendations

**Use Gemini 2.0 Flash for**:
- High-volume processing (fast processing speed)
- Detailed image descriptions
- Cost-sensitive applications

**Use GPT-4o for**:
- Table-heavy documents
- When token efficiency matters
- Consistent formatting requirements

Both models achieve excellent results with 100% success rates in testing across various document types including simple text, complex multi-page documents, tables, and lists.

## Configuration

Configure models and other parameters using the `autoscan` function signature:

```python
async def autoscan(
    pdf_path: str,
    model_name: str = "openai/gpt-4o",
    accuracy: str = "medium",
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
autoscan --accuracy high --model gemini/gemini-2.0-flash ./examples/table.pdf

# Save LLM prompts and responses for debugging:
autoscan --accuracy high --model gemini/gemini-2.0-flash --save-llm-calls ./examples/table.pdf
```

## Known Issues

While AutoScan achieves excellent results with 100% success rates in testing, there are some minor issues to be aware of:

### Minor Issues Found
1. **GPT-4o**: Occasional duplicate headings in output
2. **Gemini**: Inconsistent table break handling (may add empty rows between table sections)
3. **Both models**: Missing main headings for pure list documents
4. **Heading Levels**: Some inconsistency between `#` and `##` for main titles

These issues are cosmetic and don't affect the core functionality or data accuracy. The output remains highly usable for downstream processing.

## Testing

To run the test suite:

```sh
pytest tests/
```