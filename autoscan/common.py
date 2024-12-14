import aiohttp
import aiofiles
from typing import Optional
from pathlib import Path
from urllib.parse import urlparse

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
            filename = Path(parsed.path).name or "downloaded_file"
            local_path = Path(destination_dir) / filename

            # Create destination directory if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download the file
            async with aiohttp.ClientSession() as session:
                async with session.get(file_path) as response:
                    if response.status != 200:
                        return None
                    async with aiofiles.open(local_path, mode="wb") as f:
                        await f.write(await response.read())
            return str(local_path)

        # Handle local file paths
        local_path = Path(file_path).resolve()
        if not local_path.exists():
            return None
        return str(local_path)

    except Exception:
        return None
