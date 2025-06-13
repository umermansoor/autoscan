import logging
from typing import List, Optional
import base64
import os
from PIL import Image

from pdf2image import convert_from_path

from .config import PDF2ImageConversionConfig

def pdf_to_images(pdf_path: str, temp_folder: str, accuracy: str = "high", first_page: Optional[int] = None, last_page: Optional[int] = None) -> Optional[List[str]]:
    try:
        # Get DPI based on accuracy level
        dpi = PDF2ImageConversionConfig.get_dpi_for_accuracy(accuracy)
        
        logging.debug(f"Converting PDF to images: {pdf_path}")
        logging.debug(f"Output folder: {temp_folder}")
        logging.debug(f"Using config: DPI={dpi} (accuracy={accuracy}), format={PDF2ImageConversionConfig.FORMAT}, threads={PDF2ImageConversionConfig.NUM_THREADS}")
        if first_page is not None or last_page is not None:
            logging.debug(f"Page range: first_page={first_page}, last_page={last_page}")
        
        image_paths = convert_from_path(
            pdf_path,
            output_folder=temp_folder,
            paths_only=True,
            fmt=PDF2ImageConversionConfig.FORMAT,
            dpi=dpi,
            use_pdftocairo=PDF2ImageConversionConfig.USE_PDFTOCAIRO,
            thread_count=PDF2ImageConversionConfig.NUM_THREADS,
            first_page=first_page,
            last_page=last_page)
        
        logging.debug(f"Successfully created {len(image_paths)} page images")
        
        # Collect and log image statistics
        total_size_mb = 0.0
        for i, path in enumerate(image_paths, 1):
            try:
                with Image.open(path) as img:
                    width, height = img.size
                    file_size_mb = os.path.getsize(path) / (1024 * 1024)
                    total_size_mb += file_size_mb
                    
                    # Get DPI information if available
                    dpi_info = img.info.get('dpi', (None, None))
                    dpi_str = f", DPI={dpi_info[0]:.0f}" if dpi_info[0] else f", DPI=N/A"
                    
                    logging.info(f"📸 Page {i}: {width}x{height}px{dpi_str}, {file_size_mb:.2f}MB, format={img.format}")
            except Exception as e:
                logging.warning(f"Could not read image stats for page {i}: {e}")
                logging.debug(f"Page {i}: {path}")
        
        avg_size_mb = total_size_mb / len(image_paths) if image_paths else 0
        logging.info(f"📊 Image stats: {len(image_paths)} pages, total={total_size_mb:.2f}MB, avg={avg_size_mb:.2f}MB/page, DPI={dpi} ({accuracy} accuracy)")
        
        return image_paths
    except Exception as e:
        logging.error(f"Error converting PDF to images: {e}")
        return None

def image_to_base64(image_path: str) -> str:
    logging.debug(f"Encoding image {image_path} to base64")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")