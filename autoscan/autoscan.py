import os
import logging
from typing import List, Optional
from datetime import datetime
import tempfile

from .image_processing import pdf_to_images
from .model import LlmModel
from .types import AutoScanOutput

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def autoscan(
    pdf_path: str, 
    model_name: str = "gpt-4o",
    output_dir: Optional[str] = None,
    temp_dir: Optional[str] = None,
    cleanup_temp: bool = True
) -> AutoScanOutput:
    """
    Convert a PDF to markdown by:
      1. Converting PDF pages into images.
      2. Using an LLM to process each page image into markdown.
      3. Aggregating all markdown pages into a single file.

    Args:
        pdf_path (str): The path to the PDF file.
        model_name (str, optional): The name of the model to use. Defaults to "gpt-4o".
        output_dir (str, optional): Directory to save the output markdown.
                                    If None, saves alongside the PDF.
        temp_dir (str, optional): Directory for temporary images. If None, a temporary directory
                                  is created and automatically cleaned up.
        cleanup_temp (bool, optional): Whether to clean up temporary images if `temp_dir` is provided. Defaults to True.

    Returns:
        AutoScanOutput: Contains completion time, markdown file path, markdown content, and token usage.
    """
    # Validate input PDF path
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    logging.info(f"Starting autoscan for PDF: {pdf_path}")

    start_time = datetime.now()

    # Determine output directory
    output_dir = _determine_output_dir(pdf_path, output_dir)

    # Determine temporary directory
    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        temp_directory = temp_dir
        temp_dir_context = None  # No context manager needed
    else:
        temp_dir_context = tempfile.TemporaryDirectory()
        temp_directory = temp_dir_context.name
    logging.info(f"Using temporary directory: {temp_directory}")

    # Convert PDF to images
    images = pdf_to_images(pdf_path, temp_directory)
    if not images:
        raise RuntimeError("No images were generated from the PDF.")
    logging.info(f"Generated {len(images)} images from PDF.")

    # Initialize the model
    model = LlmModel(model_name=model_name)
    logging.info(f"Initialized model: {model_name}")

    # Process images and aggregate markdown
    aggregated_markdown, total_prompt_tokens, total_completion_tokens = _process_images(
        images, model
    )

    # Construct output file path
    output_file = _write_markdown(pdf_path, output_dir, aggregated_markdown)

    # Cleanup temporary files
    if temp_dir_context:
        temp_dir_context.cleanup()
    elif cleanup_temp:
        _cleanup_temp_files(images)

    end_time = datetime.now()
    completion_time = (end_time - start_time).total_seconds()

    logging.info(f"Autoscan completed in {completion_time:.2f} seconds. Tokens Usage - Input: {total_prompt_tokens}, Output: {total_completion_tokens}")
    return AutoScanOutput(
        completion_time=completion_time,
        markdown_file=output_file,
        markdown="\n\n".join(aggregated_markdown),
        input_tokens=total_prompt_tokens,
        output_tokens=total_completion_tokens,
    )

# Helper functions
def _determine_output_dir(pdf_path: str, output_dir: Optional[str]) -> str:
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.dirname(os.path.abspath(pdf_path)) or os.getcwd()
    return output_dir

def _process_images(images: List[str], model: LlmModel):
    aggregated_markdown = []
    prior_page_markdown = ""
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for image_path in images:
        try:
            # Get the result for the current page
            result = model.completion(image_path, prior_page_markdown)

            # Validate the generated markdown
            if not result.page_markdown.strip():
                raise ValueError(f"Generated markdown for image '{image_path}' is empty.")
            
            aggregated_markdown.append(result.page_markdown)
            prior_page_markdown = result.page_markdown  # Maintain context

            # Accumulate tokens
            total_prompt_tokens += result.prompt_tokens
            total_completion_tokens += result.completion_tokens

        except Exception as e:
            # Log the error and continue with the next image
            logging.error(f"Error processing image '{image_path}': {e}")
            continue

    # Ensure that at least some markdown was generated
    if not aggregated_markdown:
        raise RuntimeError("No valid markdown was generated from the images.")

    return aggregated_markdown, total_prompt_tokens, total_completion_tokens

def _write_markdown(pdf_path: str, output_dir: str, aggregated_markdown: List[str]) -> str:
    base_name = os.path.basename(pdf_path)
    base_name_without_ext = os.path.splitext(base_name)[0]
    output_file = os.path.join(output_dir, f"{base_name_without_ext}.md")

    final_markdown = "\n\n".join(aggregated_markdown)
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_markdown)
        logging.info(f"Markdown written to: {output_file}")
    except Exception as e:
        raise IOError(f"Failed to write markdown file: {e}")

    return output_file

def _cleanup_temp_files(images: List[str]):
    for img in images:
        try:
            if os.path.exists(img):
                os.remove(img)
        except Exception as e:
            logging.warning(f"Failed to delete temporary image '{img}': {e}")
