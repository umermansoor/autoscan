import aiohttp
import aiofiles
from typing import Optional
from pathlib import Path
from urllib.parse import urlparse

async def get_or_download_file(
    file_path: str,
    destination_dir: str,
) -> Optional[str]:
    """
    Asynchronously handle a given file path:
      - If it is a URL, download it to the given destination directory.
      - If it is a local file path, ensure it exists and return the absolute path.

    Args:
        file_path (str): The path to the file, which could be a URL or a local file.
        destination_dir (str): The directory in which to store the downloaded file if `file_path` is a URL.

    Returns:
        Optional[str]: The full path to the file on the local filesystem, or None if the operation fails.
    """
    parsed = urlparse(file_path)

    # Check if the path is a URL (has a network scheme)
    if parsed.scheme in ("http", "https"):
        # URL download logic
        filename = Path(parsed.path).name
        if not filename:
            return None

        target_directory = Path(destination_dir)
        target_directory.mkdir(parents=True, exist_ok=True)
        local_path = target_directory / filename

        async with aiohttp.ClientSession() as session:
            async with session.get(file_path) as response:
                if response.status != 200:
                    return None
                async with aiofiles.open(local_path, mode='wb') as f:
                    async for chunk in response.content.iter_chunked(1024):
                        await f.write(chunk)
        return str(local_path)
    else:
        # Assume it's a local file path
        local_path = Path(file_path).resolve()
        return str(local_path) if local_path.exists() else None

