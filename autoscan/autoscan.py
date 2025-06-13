import asyncio
import os
import logging
from typing import List, Optional, Tuple
from datetime import datetime
import tempfile
from autoscan.llm_processors.base_llm_processor import BaseLLMProcessor
from autoscan.llm_processors.img_to_md_processor import ImageToMarkdownProcessor
from autoscan.llm_processors.markdown_consolidator import MarkdownConsolidator
from autoscan.prompts import IMG_TO_MARKDOWN_PROMPT, POST_PROCESSING_PROMPT

from .image_processing import pdf_to_images
from .types import AutoScanOutput
from .common import get_or_download_file, write_text_to_file
from .errors import PDFFileNotFoundError, PDFPageToImageConversionError, MarkdownFileWriteError, LLMProcessingError

logger = logging.getLogger(__name__)

async def autoscan(
    pdf_path: str,
    model_name: str = "openai/gpt-4o",
    accuracy: str = "high", 
    user_instructions: Optional[str] = None,
    output_dir: Optional[str] = None,
    temp_dir: Optional[str] = None,
    concurrency: Optional[int] = 10,
    save_llm_calls: bool = False,
    polish_output: bool = False,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None,
) -> AutoScanOutput:
    """
    Convert a PDF to markdown by:
      1. Converting PDF pages into images.
      2. Using an LLM to process each page image into markdown.
      3. Aggregating all markdown pages into a single file.
      4. Optionally post-processing for improved formatting and organization.

    ## Args
    - `pdf_path` (str): **Required.** Path to the input PDF file.
    - `model_name` (str, optional): Name of the AI model to process image-to-text conversion. Defaults to `"openai/gpt-4o"`.
    - `accuracy` (str, optional): One of `low` or `high` determining processing strategy. Defaults to `"high"`.
    - `user_instructions` (str, optional): Additional context or instructions passed directly to the LLM.
    - `output_dir` (str, optional): Directory to store the final output Markdown file. Defaults to the current directory's "output" subfolder if not provided.
    - `temp_dir` (str, optional): Directory for storing temporary images. If not specified, a temporary directory will be created and cleaned automatically after processing.
    - `concurrency` (int, optional): Maximum number of concurrent model calls. Defaults to 10.
    - `save_llm_calls` (bool, optional): Whether to save LLM calls to a file. Defaults to False.
    - `polish_output` (bool, optional): Whether to apply an additional LLM pass to improve formatting, fix broken tables, and enhance document structure. Defaults to False.
    - `first_page` (int, optional): First page to process, defaults to None (process from beginning).
    - `last_page` (int, optional): Last page to process before stopping, defaults to None (process to end).

    Returns:
        AutoScanOutput: Contains completion time, markdown file path, markdown content, and token usage.
    """
    images = None
    temp_dir_obj = None
    try:
        # Prepare temporary directory for storing intermediate files
        temp_directory, temp_dir_obj = _create_temp_dir(temp_dir)

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
        images = await asyncio.to_thread(pdf_to_images, local_path, temp_directory, accuracy, first_page, last_page)
        if not images:
            raise PDFPageToImageConversionError("Failed to convert PDF pages to images.")
        
        pdf_conversion_time = (datetime.now() - pdf_conversion_start).total_seconds()
        logger.debug(f"Generated {len(images)} page images from PDF in {pdf_conversion_time:.2f} seconds.")

        # Process images
        if accuracy not in {"low", "high"}:
            raise ValueError("accuracy must be one of 'low', or 'high'")

        # Initialize the LLM
        llm_processor: BaseLLMProcessor = ImageToMarkdownProcessor(
            model_name=model_name,
            system_prompt=IMG_TO_MARKDOWN_PROMPT,
            user_prompt=user_instructions or "",
            pass_previous_page_context=(accuracy == "high"),
        )

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
            llm_processor,
            images,
            concurrency=concurrency,
            sequential=sequential,
        )
        
        llm_processing_time = (datetime.now() - llm_processing_start).total_seconds()
        logger.debug(f"LLM processing completed in {llm_processing_time:.2f} seconds")

        # Join markdown pages
        markdown_content = _join_markdown_pages(aggregated_markdown)

        # Polish the output if requested
        if polish_output and markdown_content.strip():
            logger.info("✨ Output polishing enabled - applying additional LLM pass to improve formatting...")
            post_processing_start = datetime.now()
            
            try:
                markdown_consolidator = MarkdownConsolidator(
                    model_name=model_name,
                    system_prompt=POST_PROCESSING_PROMPT,
                    user_prompt=user_instructions or "",
                )
                
                post_result = await markdown_consolidator.acompletion(
                    markdown_content=markdown_content
                )
                
                # Update the content and token counts
                markdown_content = post_result.content
                total_prompt_tokens += post_result.prompt_tokens
                total_completion_tokens += post_result.completion_tokens
                total_cost += post_result.cost
                
                post_processing_time = (datetime.now() - post_processing_start).total_seconds()
                logger.info(
                    f"✅ Output polishing completed in {post_processing_time:.2f}s: "
                    f"tokens(in/out)={post_result.prompt_tokens}/{post_result.completion_tokens}, "
                    f"cost=${post_result.cost:.4f}"
                )
                
            except Exception as e:
                logger.error(f"❌ Output polishing failed: {e}")
                logger.info("Proceeding with original markdown content...")
        elif polish_output:
            logger.warning("Output polishing requested but no content to process")

        end_time = datetime.now()
        completion_time = (end_time - start_time).total_seconds()

        # Write the aggregated markdown to file
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_filename = await write_text_to_file(f"{base_name}.md", output_dir, markdown_content)
        if not output_filename:
            raise MarkdownFileWriteError(f"Failed to write markdown file: {output_filename}")

        # Calculate averages for better insights
        num_pages = len(aggregated_markdown) if aggregated_markdown else 1  # Avoid division by zero
        avg_prompt_tokens = total_prompt_tokens / num_pages
        avg_completion_tokens = total_completion_tokens / num_pages
        avg_cost_per_page = total_cost / num_pages

        summary_lines = [
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
        
        if polish_output:
            summary_lines.append(f"  ✨ Output polishing: enabled")
        
        summary = "\n".join(summary_lines)
        logger.info(summary)
        
        # Show clipboard copy command with better formatting
        logger.info("")
        logger.info("📋 Copy to clipboard:")
        logger.info(f"   cat {output_filename} | pbcopy")
        logger.info("")

        return AutoScanOutput(
            completion_time=completion_time,
            markdown_file=output_filename,
            markdown=markdown_content,
            input_tokens=total_prompt_tokens,
            output_tokens=total_completion_tokens,
            cost=total_cost,
            accuracy=accuracy,
        )
    finally:
        # Clean up temp files only if we created the temp directory
        # If user provided temp_dir, they are responsible for cleanup
        if temp_dir_obj and images:
            # We created the temp directory, so cleanup individual files before directory cleanup
            logger.debug(f"Cleaning up {len(images)} temporary image files...")
            await asyncio.to_thread(_cleanup_temp_files, images)
        
        # Clean up temp directory if it was auto-created
        if temp_dir_obj:
            temp_dir_obj.cleanup()


async def _process_images_async(
    llm_processor: BaseLLMProcessor,
    pdf_page_images: List[str],
    concurrency: Optional[int] = 10,
    sequential: bool = False,
) -> Tuple[List[str], int, int, float]:
    """
    Process each image using the given model to extract text.

    When ``sequential`` is True, each page is processed one after another and
    the markdown from the previous page is provided as context to the next.
    """

    if not concurrency:
        concurrency = len(pdf_page_images)

    context = asyncio.Semaphore(concurrency)

    async def process_single_image(image_path: str, page_num: int, previous_page_markdown: Optional[str] = None):
        async with context:
            logger.info(f"🔄 Processing page {page_num} of {len(pdf_page_images)}: {os.path.basename(image_path)}")
            try:
                result = await llm_processor.acompletion(
                    image_path=image_path,
                    previous_page_markdown=previous_page_markdown,
                    page_number=page_num
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
        for i, img in enumerate(pdf_page_images):
            page_num = i + 1
            result = await process_single_image(
                img, 
                page_num=page_num, 
                previous_page_markdown=last_page_markdown
            )
            if result:
                valid_results.append(result)
                last_page_markdown = result.content
                logger.debug(f"Sequential: Page {page_num} processed, result stored for next page context")
    else:
        logger.debug("Starting concurrent processing (pages processed independently)")
        tasks = []
        for i, img in enumerate(pdf_page_images):
            page_num = i + 1
            # Concurrent processing: each page is processed independently without previous context
            tasks.append(process_single_image(img, page_num=page_num))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"❌ Page {i + 1} failed with exception: {r}")
            elif r:
                valid_results.append(r)
        
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
          

def _create_temp_dir(temp_dir: Optional[str] = None) -> Tuple[str, Optional[tempfile.TemporaryDirectory]]:
    """
    Creates or prepares a temporary directory.
    
    Args:
        temp_dir: An existing directory path to use. If provided, no automatic cleanup occurs.

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
        # Create a temporary directory that will auto-cleanup
        temp_dir_obj = tempfile.TemporaryDirectory(delete=True)
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

def _join_markdown_pages(aggregated_markdown: List[str]) -> str:
    """
    Join markdown pages with appropriate spacing.
    
    Args:
        aggregated_markdown: List of markdown content from each page
        
    Returns:
        Combined markdown content
    """
    if not aggregated_markdown:
        return ""
    
    # Clean pages by removing any "---PAGE BREAK---" markers and stripping whitespace
    cleaned_pages = [page.replace("---PAGE BREAK---", "").rstrip() for page in aggregated_markdown]
    valid_pages = [page for page in cleaned_pages if page.rstrip()]
    
    if not valid_pages:
        return ""
    
    # Join pages with appropriate spacing
    result_parts = [valid_pages[0]]
    for i in range(1, len(valid_pages)):
        prev_page = valid_pages[i-1]
        current_page = valid_pages[i]
        
        # Use single newline for table continuations, double newline otherwise
        separator = "\n" if prev_page.endswith("|") and current_page.startswith("|") else "\n\n"
        result_parts.append(separator + current_page)
    
    return "".join(result_parts)
