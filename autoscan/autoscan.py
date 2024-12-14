import asyncio
import os
import logging
from typing import List, Optional
from datetime import datetime
import tempfile

from .image_processing import pdf_to_images
from .model import LlmModel
from .types import AutoScanOutput
from  .common import get_or_download_file
from .errors import PDFFileNotFoundError, PDFPageToImageConversion

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def autoscan(
    pdf_path: str, 
    model_name: str = "gpt-4o",
    transcribe_images: Optional[bool] = True,
    output_dir: Optional[str] = None,
    temp_dir: Optional[str] = None,
    cleanup_temp: bool = True
) -> AutoScanOutput:
    """
    Convert a PDF to markdown by:
      1. Converting PDF pages into images.
      2. Using an LLM to process each page image into markdown.
      3. Aggregating all markdown pages into a single file.

    ## Args
    - `pdf_path` (str): **Required.** Path to the input PDF file.
    - `model_name` (str, optional): Name of the AI model to process image-to-text conversion. Defaults to `"gpt-4o"`.
    - `transcribe_images` (bool, optional): Whether to process images for transcription. Defaults to `True`.
    - `output_dir` (str, optional): Directory to store the final output Markdown file. Defaults to the same location as the input PDF if not provided.
    - `temp_dir` (str, optional): Directory for storing temporary images during processing. If not specified, a temporary directory will be created and cleaned automatically.
    - `cleanup_temp` (bool, optional): If `True`, cleans up temporary files when the process completes. Defaults to `True`.

    Returns:
        AutoScanOutput: Contains completion time, markdown file path, markdown content, and token usage.
    """

    if temp_dir:
        os.makedirs(temp_dir, exist_ok=True)
        temp_directory = temp_dir
        temp_dir_context = None
    else:
        temp_dir_context = tempfile.TemporaryDirectory()
        temp_directory = temp_dir_context.name
    logging.info(f"Using temporary directory: {temp_directory}")
    
    local_path = await get_or_download_file(pdf_path, temp_directory)
    if not local_path:
        raise PDFFileNotFoundError(f"Failed to access or download PDF from: {pdf_path}")

    if not output_dir:
        output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    start_time = datetime.now()
    images = await asyncio.to_thread(pdf_to_images, local_path, temp_directory)
    if not images:
        raise PDFPageToImageConversion("Failed to convert PDF pages to images.")
    
    logging.info(f"Generated {len(images)} images from PDF.")

    model = LlmModel(model_name=model_name)
    logging.info(f"Initialized model: {model_name}")

    aggregated_markdown, total_prompt_tokens, total_completion_tokens, total_cost = await _process_images_async(images, model, transcribe_images)

    end_time = datetime.now()
    completion_time = (end_time - start_time).total_seconds()

    output_file = await asyncio.to_thread(_write_markdown, local_path, output_dir, aggregated_markdown)

    if temp_dir_context:
        temp_dir_context.cleanup()
    elif cleanup_temp:
        await asyncio.to_thread(_cleanup_temp_files, images)

    

    logging.info(f"Autoscan completed in {completion_time:.2f} seconds. Tokens Usage - Input: {total_prompt_tokens}, Output: {total_completion_tokens}. Cost = ${total_cost:.2f}." )
    return AutoScanOutput(
        completion_time=completion_time,
        markdown_file=output_file,
        markdown="\n\n".join(aggregated_markdown),
        input_tokens=total_prompt_tokens,
        output_tokens=total_completion_tokens,
    )

async def _process_images_async(images: List[str], model: LlmModel, transcribe_images):
    aggregated_markdown = []
    prior_page_markdown = ""
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    async def process_single_image(image_path):
        try:
            # Await the async completion method
            result = await model.completion(image_path, prior_page_markdown, transcribe_images=transcribe_images)
            if not result.page_markdown.strip():
                raise ValueError(f"Generated markdown for image '{image_path}' is empty.")
            return result
        except Exception as e:
            logging.error(f"Error processing image '{image_path}': {e}")
            return None

    tasks = [process_single_image(image) for image in images]
    results = await asyncio.gather(*tasks)

    for result in results:
        if result:
            aggregated_markdown.append(result.page_markdown)
            prior_page_markdown = result.page_markdown
            total_prompt_tokens += result.prompt_tokens
            total_completion_tokens += result.completion_tokens
            total_cost += result.cost

    if not aggregated_markdown:
        raise RuntimeError("No valid markdown was generated from the images.")

    return aggregated_markdown, total_prompt_tokens, total_completion_tokens, total_cost


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