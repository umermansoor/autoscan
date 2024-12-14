import asyncio
import os
import logging
from typing import List, Optional
from datetime import datetime
import tempfile

from .image_processing import pdf_to_images
from .model import LlmModel
from .types import AutoScanOutput
from  .common import get_or_download_file, write_text_to_file
from .errors import PDFFileNotFoundError, PDFPageToImageConversion, MarkdownFileWriteError

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
    - `cleanup_temp` (bool, optional): If `True`, cleans up  temporary, intermediate files when the process completes. Defaults to `True`.

    Returns:
        AutoScanOutput: Contains completion time, markdown file path, markdown content, and token usage.
    """

    temp_directory = _create_temp_dir(temp_dir, cleanup_temp)
    
    local_path = await  get_or_download_file(pdf_path, temp_directory)
    if not local_path:
        raise PDFFileNotFoundError(f"Failed to access or download PDF from: {pdf_path}")

    if not output_dir:
        output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Using temporary directory: {temp_directory} and output directory: {output_dir}")

    start_time = datetime.now()

    images = await asyncio.to_thread(pdf_to_images, local_path, temp_directory)
    if not images:
        raise PDFPageToImageConversion("Failed to convert PDF pages to images.")
    
    logging.info(f"Generated {len(images)} images.")

    model = LlmModel(model_name=model_name)
    logging.info(f"Initialized model: {model_name}")

    aggregated_markdown, total_prompt_tokens, total_completion_tokens, total_cost = await _process_images_async(images, model, transcribe_images)

    end_time = datetime.now()
    completion_time = (end_time - start_time).total_seconds()

    output_filename = await write_text_to_file(os.path.splitext(os.path.basename(pdf_path))[0] + ".md", output_dir, "\n\n".join(aggregated_markdown))
    if not output_filename:
        raise MarkdownFileWriteError(f"Failed to write markdown file: {output_filename}")

    logging.info(f"Autoscan completed in {completion_time:.2f} seconds. Tokens Usage - Input: {total_prompt_tokens}, Output: {total_completion_tokens}. Cost = ${total_cost:.2f}." )

    if cleanup_temp: 
        await asyncio.to_thread(_cleanup_temp_files, images)

    return AutoScanOutput(
        completion_time=completion_time,
        markdown_file=output_filename,
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
            result = await model.completion(image_path, prior_page_markdown, transcribe_images=transcribe_images)
            return result
        except Exception as e:
            raise RuntimeError(f"Error processing image '{image_path}': {e}. Aborting.") 

    tasks = [process_single_image(image) for image in images]
    results = await asyncio.gather(*tasks)

    for result in results:
        if result:
            aggregated_markdown.append(result.page_markdown)
            prior_page_markdown = result.page_markdown
            total_prompt_tokens += result.prompt_tokens
            total_completion_tokens += result.completion_tokens
            total_cost += result.cost

    return aggregated_markdown, total_prompt_tokens, total_completion_tokens, total_cost

def _create_temp_dir(temp_dir: Optional[str], cleanup: bool = True ) -> str:
    if not temp_dir: # if user didn't specify a temp directory, create one
        temp_dir = tempfile.TemporaryDirectory(delete=cleanup).name
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir

def _cleanup_temp_files(images: List[str]):
    for img in images:
        try:
            if os.path.exists(img):
                os.remove(img)
        except Exception as e:
            logging.warning(f"Failed to delete temporary image '{img}': {e}")