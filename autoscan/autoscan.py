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
    model_name: str = "openai/gpt-4o",
    accuracy: str = "high", 
    user_instructions: Optional[str] = None,
    output_dir: Optional[str] = None,
    temp_dir: Optional[str] = None,
    cleanup_temp: bool = True,
    concurrency: Optional[int] = 10,
    save_llm_calls: bool = False, # Added save_llm_calls
) -> AutoScanOutput:
    """
    Convert a PDF to markdown by:
      1. Converting PDF pages into images.
      2. Using an LLM to process each page image into markdown.
      3. Aggregating all markdown pages into a single file.

    ## Args
    - `pdf_path` (str): **Required.** Path to the input PDF file.
    - `model_name` (str, optional): Name of the AI model to process image-to-text conversion. Defaults to `"openai/gpt-4o"`.
    - `accuracy` (str, optional): One of `low` or `high` determining processing strategy. Defaults to `"high"`.
    - `user_instructions` (str, optional): Additional context or instructions passed directly to the LLM.
    - `output_dir` (str, optional): Directory to store the final output Markdown file. Defaults to the current directory's "output" subfolder if not provided.
    - `temp_dir` (str, optional): Directory for storing temporary images. If not specified, a temporary directory will be created and cleaned automatically after processing.
    - `cleanup_temp` (bool, optional): If `True`, cleans up temporary, intermediate files upon completion. Defaults to `True`.
    - `concurrency` (int, optional): Maximum number of concurrent model calls. Defaults to 10.
    - `save_llm_calls` (bool, optional): Whether to save LLM calls to a file. Defaults to False.

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

        logger.info(f"📄 Processing PDF: {os.path.basename(pdf_path)} → {os.path.basename(output_dir)}")

        start_time = datetime.now()

        # Convert PDF to images (Each page becomes separate image)
        pdf_conversion_start = datetime.now()
        images = await asyncio.to_thread(pdf_to_images, local_path, temp_directory)
        if not images:
            raise PDFPageToImageConversionError("Failed to convert PDF pages to images.")
        
        pdf_conversion_time = (datetime.now() - pdf_conversion_start).total_seconds()
        logger.debug(f"Generated {len(images)} page images from PDF in {pdf_conversion_time:.2f} seconds.")

        # Initialize the LLM
        model = LlmModel(model_name=model_name, accuracy=accuracy, save_llm_calls=save_llm_calls) # Pass save_llm_calls
        logger.info(f"Using model: {model_name} ({accuracy} accuracy)")

        # Process images
        if accuracy not in {"low", "high"}:  
            raise ValueError("accuracy must be one of 'low' or 'high'")

        sequential = accuracy == "high"
        
        processing_mode = "sequential (with context)" if sequential else f"concurrent (max {concurrency})"
        logger.info(f"🚀 Processing {len(images)} pages - {processing_mode}")

        llm_processing_start = datetime.now()
        (
            aggregated_markdown,
            total_prompt_tokens,
            total_completion_tokens,
            total_cost,
        ) = await _process_images_async(
            images,
            model,
            concurrency=concurrency,
            sequential=sequential,
            user_instructions=user_instructions,
            # save_llm_calls is part of the model instance now, no need to pass here
            # page_number will be handled inside _process_images_async
        )
        
        llm_processing_time = (datetime.now() - llm_processing_start).total_seconds()
        logger.debug(f"LLM processing completed in {llm_processing_time:.2f} seconds")

        # New logic for joining markdown pages
        if not aggregated_markdown:
            markdown_content = ""
        else:
            # First, handle any potential "---PAGE BREAK---" markers within page content itself.
            # This is a safeguard; current prompts don't request this marker.
            cleaned_pages = [page.replace("---PAGE BREAK---", "") for page in aggregated_markdown]

            # Strip whitespace from each page and filter out pages that become empty
            # after cleaning and stripping.
            valid_pages = [page.strip() for page in cleaned_pages if page.strip()]

            if not valid_pages:
                markdown_content = ""
            else:
                markdown_content_parts = [valid_pages[0]]
                for i in range(1, len(valid_pages)):
                    prev_page_md_stripped = valid_pages[i-1]
                    current_page_md_stripped = valid_pages[i]

                    # Condition for single newline: previous ends like a table row, current starts like one.
                    if prev_page_md_stripped.endswith("|") and current_page_md_stripped.startswith("|"):
                        # Join with a single newline to continue the table
                        markdown_content_parts.append("\n" + current_page_md_stripped)
                    else:
                        # Default join with two newlines for separate blocks
                        markdown_content_parts.append("\n\n" + current_page_md_stripped)
                markdown_content = "".join(markdown_content_parts)

        end_time = datetime.now()
        completion_time = (end_time - start_time).total_seconds()

        # Write the aggregated markdown to file
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_filename = await write_text_to_file(f"{base_name}.md", output_dir, markdown_content)
        if not output_filename:
            raise MarkdownFileWriteError(f"Failed to write markdown file: {output_filename}")

        # Calculate averages for better insights
        avg_prompt_tokens = total_prompt_tokens / len(aggregated_markdown) if aggregated_markdown else 0
        avg_completion_tokens = total_completion_tokens / len(aggregated_markdown) if aggregated_markdown else 0
        avg_cost_per_page = total_cost / len(aggregated_markdown) if aggregated_markdown else 0

        summary = "\n".join(
            [
                "🎉 AutoScan completed successfully!",
                f"  📄 Output file      : {output_filename}",
                f"  ⏱️  Completion time  : {completion_time:.2f} seconds",
                f"  📊 Pages processed  : {len(aggregated_markdown)}",
                f"  🔢 Tokens (in/out)  : {total_prompt_tokens:,}/{total_completion_tokens:,}",
                f"  💰 Total cost      : ${total_cost:.4f}",
                f"  📈 Avg per page     : {avg_prompt_tokens:.0f}/{avg_completion_tokens:.0f} tokens, ${avg_cost_per_page:.4f}",
                f"  🎯 Accuracy level   : {accuracy}",
                f"  📝 Content length   : {len(markdown_content):,} characters",
            ]
        )
        logger.info(summary)
        logger.info(
            f"To copy the markdown content to clipboard, run:\n"
            f"cat {output_filename} | pbcopy"
        )

        return AutoScanOutput(
            completion_time=completion_time,
            markdown_file=output_filename,
            markdown=markdown_content,
            input_tokens=total_prompt_tokens,
            output_tokens=total_completion_tokens,
            accuracy=accuracy,
        )
    finally:
        # Clean up temp files if requested
        if cleanup_temp and images:
            logger.debug(f"Cleaning up {len(images)} temporary image files...")
            await asyncio.to_thread(_cleanup_temp_files, images)


async def _process_images_async(
    pdf_page_images: List[str],
    model: LlmModel,
    concurrency: Optional[int] = 10,
    sequential: bool = False,
    user_instructions: Optional[str] = None,
    # save_llm_calls is part of the model instance
) -> Tuple[List[str], int, int, float]:
    """
    Process each image using the given model to extract text.

    When ``sequential`` is True, each page is processed one after another and
    the markdown from the previous page is provided as context to the next.
    """

    if not concurrency:
        concurrency = len(pdf_page_images)

    context = asyncio.Semaphore(concurrency)

    async def process_single_image(image_path: str, page_num: int, previous_page_markdown: Optional[str] = None, previous_page_image_path: Optional[str] = None):
        async with context:
            logger.info(f"🔄 Processing page {page_num} of {len(pdf_page_images)}: {os.path.basename(image_path)}")
            try:
                result = await model.image_to_markdown(
                    image_path,
                    previous_page_markdown=previous_page_markdown,
                    user_instructions=user_instructions,
                    previous_page_image_path=previous_page_image_path,
                    page_number=page_num # Pass page_number
                )
                
                # Log successful completion with token info
                logger.info(
                    f"✅ Page {page_num} completed: "
                    f"tokens(in/out)={result.prompt_tokens}/{result.completion_tokens}, "
                    f"cost=${result.cost:.4f}"
                )
                return result
            except Exception as e:
                logger.error(f"❌ Page {page_num} failed: {e}")
                raise LLMProcessingError(
                    f"Error processing image '{image_path}': {e}"
                ) from e

    if sequential:
        logger.debug("Starting sequential processing (with previous page context)")
        valid_results = []
        last_page_markdown = None
        last_page_image_path = None
        for i, img in enumerate(pdf_page_images):
            page_num = i + 1
            result = await process_single_image(
                img, 
                page_num=page_num, 
                previous_page_markdown=last_page_markdown,
                previous_page_image_path=last_page_image_path
            )
            if result:
                valid_results.append(result)
                last_page_markdown = result.content
                last_page_image_path = img  # Store current image as previous for next iteration
                logger.debug(f"Sequential: Page {page_num} processed, result stored for next page context")
    else:
        logger.debug("Starting concurrent processing")
        # Create tasks for concurrent processing
        tasks = [
            process_single_image(img, page_num=i + 1)
            for i, img in enumerate(pdf_page_images)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [r for r in results if r and not isinstance(r, Exception)]
        
        # Log any exceptions that occurred
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"❌ Page {i + 1} failed with exception: {r}")
        
        logger.info(f"Concurrent processing completed: {len(valid_results)}/{len(pdf_page_images)} pages successful")

    aggregated_markdown = [r.content for r in valid_results]
    total_prompt_tokens = sum(r.prompt_tokens for r in valid_results)
    total_completion_tokens = sum(r.completion_tokens for r in valid_results)
    total_cost = sum(r.cost for r in valid_results)

    logger.debug(
        f"Processing summary: {len(valid_results)}/{len(pdf_page_images)} pages successful, "
        f"total tokens(in/out)={total_prompt_tokens}/{total_completion_tokens}, "
        f"total cost=${total_cost:.4f}"
    )

    return aggregated_markdown, total_prompt_tokens, total_completion_tokens, total_cost
          

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
    cleaned_count = 0
    failed_count = 0
    
    for img in images:
        try:
            if os.path.exists(img):
                os.remove(img)
                cleaned_count += 1
                logger.debug(f"🗑️  Deleted: {os.path.basename(img)}")
            else:
                logger.debug(f"⚠️  File not found: {os.path.basename(img)}")
        except Exception as e:
            failed_count += 1
            logger.warning(f"❌ Failed to delete temporary image '{img}': {e}")
    
    logger.debug(f"Cleanup completed: {cleaned_count} files deleted, {failed_count} failed")
