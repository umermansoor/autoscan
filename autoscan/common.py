import logging
import aiohttp
import aiofiles  # type: ignore
from typing import Optional
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

async def get_or_download_file(file_path: str, destination_dir: str) -> Optional[str]:
    """
    Handles a file path or URL by ensuring the file exists locally.
    - Downloads the file if it's a URL.
    - Validates the existence of a local file path.

    Args:
        file_path (str): URL or local file path.
        destination_dir (str): Directory for storing downloaded files.

    Returns:
        Optional[str]: Local file path, or None if the operation fails.
    """
    try:
        parsed = urlparse(file_path)

        # Handle URLs
        if parsed.scheme in ("http", "https"):
            logger.debug(f"Downloading file from URL: {file_path}")
            filename = Path(parsed.path).name or "downloaded_file"
            local_path = Path(destination_dir) / filename

            # Create destination directory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download the file
            async with aiohttp.ClientSession() as session:
                async with session.get(file_path) as response:
                    if response.status != 200:
                        logger.error(
                            f"Failed to download {file_path}: status {response.status}"
                        )
                        return None
                    async with aiofiles.open(local_path, mode="wb") as f:
                        await f.write(await response.read())
            logger.debug(f"File downloaded to {local_path}")
            return str(local_path)

        # Handle local file paths
        local_path = Path(file_path).resolve()
        if not local_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        logger.debug(f"Using local file {local_path}")
        return str(local_path)

    except Exception as e:
        logger.exception(f"Failed to handle file {file_path}: {e}")
        return None


async def write_text_to_file(filename: str, output_dir: str, text:str) -> Optional[str]:
    """
    Writes text content to a file in the specified output directory.

    Args:
        filename (str): Name of the file to write.
        output_dir (str): Directory for storing the file.

    Returns:
        str: Full path to the written file.
    """

    if not filename or not output_dir:
        logger.error("Filename or output directory not provided")
        return None
    
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(output_path, mode="w", encoding="utf-8") as f:
        await f.write(text)
    logger.debug(f"Wrote text to {output_path}")

    return str(output_path)