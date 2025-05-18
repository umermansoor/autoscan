import logging
from typing import List, Optional
import base64

from pdf2image import convert_from_path

from .config import PDF2ImageConversionConfig

def pdf_to_images(pdf_path: str, temp_folder: str) -> Optional[List[str]]:
    try:
        logging.debug(f"Converting PDF {pdf_path} to images in {temp_folder}")
        image_paths = convert_from_path(
            pdf_path,
            output_folder=temp_folder,
            paths_only=True,
            size=PDF2ImageConversionConfig.SIZE,
            fmt=PDF2ImageConversionConfig.FORMAT,
            dpi=PDF2ImageConversionConfig.DPI,
            use_pdftocairo=PDF2ImageConversionConfig.USE_PDFTOCAIRO,
            thread_count=PDF2ImageConversionConfig.NUM_THREADS)
        logging.debug(f"Created {len(image_paths)} images")
        return image_paths
    except Exception as e:
        logging.error(f"Error converting PDF to images: {e}")
        return None

def image_to_base64(image_path: str) -> str:
    logging.debug(f"Encoding image {image_path} to base64")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")