import logging
from typing import List, Optional
import base64

from pdf2image import convert_from_path

from .config import PDF2ImageConversionConfig

def pdf_to_images(pdf_path: str, temp_folder: str) -> Optional[List[str]]:
    try:
        logging.debug(f"Converting PDF to images: {pdf_path}")
        logging.debug(f"Output folder: {temp_folder}")
        logging.debug(f"Using config: DPI={PDF2ImageConversionConfig.DPI}, format={PDF2ImageConversionConfig.FORMAT}, threads={PDF2ImageConversionConfig.NUM_THREADS}")
        
        image_paths = convert_from_path(
            pdf_path,
            output_folder=temp_folder,
            paths_only=True,
            size=PDF2ImageConversionConfig.SIZE,
            fmt=PDF2ImageConversionConfig.FORMAT,
            dpi=PDF2ImageConversionConfig.DPI,
            use_pdftocairo=PDF2ImageConversionConfig.USE_PDFTOCAIRO,
            thread_count=PDF2ImageConversionConfig.NUM_THREADS)
        
        logging.debug(f"Successfully created {len(image_paths)} page images")
        for i, path in enumerate(image_paths, 1):
            logging.debug(f"Page {i}: {path}")
        
        return image_paths
    except Exception as e:
        logging.error(f"Error converting PDF to images: {e}")
        return None

def image_to_base64(image_path: str) -> str:
    logging.debug(f"Encoding image {image_path} to base64")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")