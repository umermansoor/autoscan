import asyncio
import os
import logging
from typing import List, Optional, Tuple
from datetime import datetime
import tempfile

from .image_processing import pdf_to_images
from .model import LlmModel
from .types import AutoScanOutput
from .common import get_or_download_file, write_text_to_file
from .errors import PDFFileNotFoundError, PDFPageToImageConversionError, MarkdownFileWriteError, LLMProcessingError

# Get the logger
logger = logging.getLogger(__name__)

async def autoscan(
    pdf_path: str,
    model_name: str = "gpt-4o",
    transcribe_images: bool = True,
    output_dir: Optional[str] = None,
    temp_dir: Optional[str] = None,
    cleanup_temp: bool = True,
    concurrency: Optional[int] = 10 
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
    - `output_dir` (str, optional): Directory to store the final output Markdown file. Defaults to the current directory's "output" subfolder if not provided.
    - `temp_dir` (str, optional): Directory for storing temporary images. If not specified, a temporary directory will be created and cleaned automatically after processing.
    - `cleanup_temp` (bool, optional): If `True`, cleans up temporary, intermediate files upon completion. Defaults to `True`.
    - `concurrency` (int, optional): Maximum number of concurrent model calls. Defaults to 10.

    Returns:
        AutoScanOutput: Contains completion time, markdown file path, markdown content, and token usage.
    """
    images = None
    try:
        # Prepare temporary directory for storing intermediate files
        temp_directory, _ = _create_temp_dir(temp_dir, cleanup_temp)

        # Retrieve or download the PDF
        local_path = await get_or_download_file(pdf_path, temp_directory)
        if not local_path:
            raise PDFFileNotFoundError(f"Failed to access or download PDF from: {pdf_path}")

        # Set up output directory
        if not output_dir:
            output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Using temporary directory: {temp_directory} and output directory: {output_dir}")

        start_time = datetime.now()

        # Convert PDF to images (Each page becomes separate image)
        images = await asyncio.to_thread(pdf_to_images, local_path, temp_directory)
        if not images:
            raise PDFPageToImageConversionError("Failed to convert PDF pages to images.")

        logger.info(f"Generated {len(images)} page images from PDF.")

        # Initialize the LLM
        model = LlmModel(model_name=model_name)
        logger.info(f"Initialized model: {model_name}")

        # Process images
        (aggregated_markdown, 
         total_prompt_tokens, 
         total_completion_tokens, 
         total_cost) = await _process_images_async(
            images, model, transcribe_images, concurrency=concurrency
        )

        end_time = datetime.now()
        completion_time = (end_time - start_time).total_seconds()

        # Write the aggregated markdown to file
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        markdown_content = "\n\n".join(aggregated_markdown)
        output_filename = await write_text_to_file(f"{base_name}.md", output_dir, markdown_content)
        if not output_filename:
            raise MarkdownFileWriteError(f"Failed to write markdown file: {output_filename}")

        logger.info(
            f"Autoscan completed in {completion_time:.2f} seconds. "
            f"Markdown file: {output_filename}. "
            f"Tokens Usage - Input: {total_prompt_tokens}, Output: {total_completion_tokens}. "
            f"Cost = ${total_cost:.2f}."
        )

        return AutoScanOutput(
            completion_time=completion_time,
            markdown_file=output_filename,
            markdown=markdown_content,
            input_tokens=total_prompt_tokens,
            output_tokens=total_completion_tokens,
        )
    finally:
        # Clean up temp files if requested
        if cleanup_temp and images:
            await asyncio.to_thread(_cleanup_temp_files, images)


async def _process_images_async(
    images: List[str],
    model: LlmModel,
    transcribe_images: bool,
    concurrency: Optional[int] = 10
) -> Tuple[List[str], int, int, float]:
    """
    Process each image using the given model to extract text.

    Args:
        images: List of image file paths.
        model: LlmModel instance for text extraction.
        transcribe_images: Whether or not to perform transcription.
        concurrency: Maximum concurrency for model calls.

    Returns:
        A tuple containing:
            - A list of aggregated markdown strings for each page.
            - Total prompt tokens used.
            - Total completion tokens used.
            - Total cost incurred.
    """

    # Use semaphore if concurrency is specified
    semaphore = asyncio.Semaphore(concurrency) if concurrency is not None else None

    async def process_single_image(image_path: str):
        if semaphore:
            async with semaphore:
                return await _run_model_completion(model, image_path, transcribe_images)
        else:
            return await _run_model_completion(model, image_path, transcribe_images)

    tasks = [process_single_image(img) for img in images]
    results = await asyncio.gather(*tasks)

    aggregated_markdown = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    # Aggregate results
    for result in results:
        if result:
            aggregated_markdown.append(result.page_markdown)
            total_prompt_tokens += result.prompt_tokens
            total_completion_tokens += result.completion_tokens
            total_cost += result.cost

    return aggregated_markdown, total_prompt_tokens, total_completion_tokens, total_cost


async def _run_model_completion(model: LlmModel, image_path: str, transcribe_images: bool):
    """
    Run a single model completion task for the given image.
    
    Args:
        model: LlmModel instance.
        image_path: The path to the image to be processed.
        transcribe_images: Whether transcription is required.

    Returns:
        The result from the model completion if successful.

    Raises:
        RuntimeError: If an error occurs during model processing.
    """
    try:
        return await model.completion(image_path, transcribe_images=transcribe_images)
    except Exception as e:
        raise LLMProcessingError(f"Error processing image '{image_path}': {e}. Aborting.")


def _create_temp_dir(temp_dir: Optional[str] = None, cleanup: bool = True) -> Tuple[str, Optional[tempfile.TemporaryDirectory]]:
    """
    Creates or prepares a temporary directory.
    
    Args:
        temp_dir: An existing directory path to use. If provided, no automatic cleanup occurs.
        cleanup: If True and no `temp_dir` is provided, create a TemporaryDirectory that
                 is automatically deleted when the object is destroyed.

    Returns:
        A tuple of:
            - The path to the directory.
            - The TemporaryDirectory object if one was created, otherwise None.
    """
    if temp_dir is not None:
        # Use the provided directory
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir, None
    else:
        # Create a temporary directory
        temp_dir_obj = tempfile.TemporaryDirectory(delete=cleanup)
        temp_dir_path = temp_dir_obj.name
        os.makedirs(temp_dir_path, exist_ok=True)
        return temp_dir_path, temp_dir_obj


def _cleanup_temp_files(images: List[str]):
    """
    Attempts to remove the provided image files from the filesystem.
    
    Args:
        images: A list of image file paths to remove.
    """
    for img in images:
        try:
            if os.path.exists(img):
                os.remove(img)
        except Exception as e:
            logger.warning(f"Failed to delete temporary image '{img}': {e}")
