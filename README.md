# AutoScan

AutoScan converts PDF files into Markdown using LLMs (GPT-4o, Gemini, etc.) with high fidelity. It's designed for complex documents like medical records, invoices, and technical papers where accuracy is critical.

## Features

- **High accuracy** conversion preserving tables, layouts, and formatting
- **Image transcription** with detailed descriptions
- **Handwriting OCR** support
- **Multi-language** document processing
- **Custom instructions** for specialized output formats
- **Multiple LLM support** via [LiteLLM](https://github.com/BerriAI/litellm)
- **Adaptive DPI settings** (150-200 DPI) automatically optimized for accuracy level
- **Flexible accuracy levels** (low=fast/concurrent, high=accurate/sequential)

![Example 1](assets/pdf_to_md_eg_1.png)
![Example 2](assets/pdf_to_md_eg_2.png)

## Quick Start

### Prerequisites

- **Python 3.12+** required
- **Poppler** (for PDF processing):
  - **macOS**: `brew install poppler`  
  - **Linux**: `sudo apt-get install poppler-utils`  
  - **Windows**: `choco install poppler` or `scoop install poppler`

### Installation

```bash
git clone https://github.com/umermansoor/autoscan.git
cd autoscan
poetry install
poetry shell  # Activates the virtual environment
```

### API Keys

Set your API key based on the model you plan to use:

```bash
# For OpenAI models (gpt-4o, etc.)
export OPENAI_API_KEY="your_openai_key"

# For Google Gemini models (recommended for cost-effectiveness)
export GEMINI_API_KEY="your_gemini_key"

# For Anthropic models
export ANTHROPIC_API_KEY="your_anthropic_key"
```

**Get Gemini API Key**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey) - free with generous rate limits!

### Verify Installation

Test your setup with the included examples:

```bash
# Test with OpenAI GPT-4o (default)
autoscan examples/helloworld.pdf

# Test with Gemini (cost-effective option)
autoscan --model gemini/gemini-2.0-flash examples/table.pdf
```

**Expected output**: 
- Success message with processing stats
- Markdown files created in `output/` directory
- Example: `output/helloworld.md` and `output/table.md`

## Usage

### Command Line Interface

```bash
autoscan path/to/your/file.pdf

# Choose accuracy level (low, medium, high):
autoscan --accuracy high path/to/your/file.pdf

# Specify a model (default is `openai/gpt-4o`):
autoscan --model gemini/gemini-2.0-flash path/to/your/file.pdf

# Provide additional instructions for the LLM:
autoscan --instructions "This is an invoice; skip disclaimers" path/to/your/file.pdf
```

### Accuracy Levels

AutoScan uses different processing strategies and image quality settings for each accuracy level:

- **`low`**: Pages processed concurrently (faster). Pages are processed independently without previous page context for maximum speed and lower costs. Uses **150 DPI** for smaller file sizes and faster processing.
- **`high`**: Pages processed sequentially (slower but more accurate). The entire previous page markdown AND the previous page image are sent as context, which increases token usage (cost) and runtime but provides better formatting consistency. Uses **200 DPI** for higher image quality and better text recognition.

**DPI (Dots Per Inch) Impact:**
- **Higher DPI** = Better text clarity and OCR accuracy, but larger files and higher costs
- **Lower DPI** = Faster processing and lower costs, but slightly reduced quality for fine details
- DPI automatically adjusts based on accuracy level - no manual configuration needed

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

Based on extensive testing with various document types:

### Model Recommendations

**Use Gemini 2.0 Flash for**:
- **Cost-effectiveness**: Significantly cheaper than GPT-4o for typical documents
- **Speed**: Fast processing times (5-9 seconds vs 20+ seconds)
- **High-volume processing**: Better for batch operations
- **Detailed image descriptions**: Excellent at describing visual elements

**Use GPT-4o for**:
- **Table formatting**: Superior table structure preservation
- **Token efficiency**: More concise output (fewer tokens generated)
- **Consistent formatting**: Better heading hierarchy consistency
- **Complex layouts**: Better handling of multi-column documents

### Performance Metrics (Tested)

| Model | Speed | Cost | Accuracy | Best For |
|-------|-------|------|----------|----------|
| Gemini 2.0 Flash | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | General use, high-volume |
| GPT-4o | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Tables, precision |

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
    concurrency: Optional[int] = 10,
) -> AutoScanOutput:
```

### Output Files

- **Markdown**: Generated in `output/` directory with same name as PDF
- **LLM Logs** (optional): When using `--save-llm-calls`, logs are saved in `logs/` directory containing:
  - Original prompts sent to LLM
  - Complete responses received
  - Useful for debugging and understanding processing

## Examples

Sample PDFs are available in the `examples/` directory for testing and demonstration.

### Basic Usage Examples

```bash
# Process a simple document (fastest, cheapest)
autoscan --model gemini/gemini-2.0-flash --accuracy low examples/helloworld.pdf

# Process a table-heavy document (best quality)
autoscan --accuracy high --model openai/gpt-4o examples/table.pdf

# Process with custom instructions
autoscan --instructions "Format as GitHub-flavored markdown tables" examples/table.pdf

# Debug mode - save all LLM interactions
autoscan --save-llm-calls --model gemini/gemini-2.0-flash examples/table.pdf
```

### Expected Performance

- **Simple documents**: 5-20 seconds, $0.0006-$0.013
- **Complex multi-page**: 15-60 seconds, $0.002-$0.05  
- **Table-heavy documents**: 10-30 seconds, $0.001-$0.02

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

```bash
pytest tests/
```

## Troubleshooting

### Common Issues

**ImportError or Module Not Found**
- Ensure you've **activated your virtual environment**: `poetry shell`
- Verify installation: `poetry install`

**PDF Processing Errors**
- Make sure Poppler is correctly installed and accessible in your PATH
- Test with: `pdftoppm -h` (should show help if Poppler is installed)

**Python Version Issues**
- Requires Python 3.12+. Check with: `python --version`
- If using older Python, consider upgrading or using pyenv

**High Costs**
- Use Gemini models (significantly cheaper for similar documents)
- Use `--accuracy low` to reduce token usage
- Test with smaller documents first

**Slow Processing**
- Use `--accuracy low` or `medium` for concurrent processing
- Consider Gemini 2.0 Flash for faster speeds
- Check your internet connection for API calls

### Performance Tips

1. **Cost Optimization**: Use `gemini/gemini-2.0-flash` for best price/performance ratio
2. **Speed Optimization**: Use `--accuracy low` for concurrent page processing
3. **Quality Optimization**: Use `--accuracy high` with `openai/gpt-4o` for best results
4. **Debugging**: Use `--save-llm-calls` to save prompts and responses for analysis

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `pytest tests/`
5. Submit a pull request

## License

This project is licensed under the MIT License. Use at your own risk.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.