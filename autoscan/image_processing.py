import logging
import os
from typing import List, Optional, Tuple
import base64
import io

from pdf2image import convert_from_path

def pdf_to_images(pdf_path: str, temp_folder: str) -> List[str]:
    try:
        image_paths = convert_from_path(pdf_path, output_folder=temp_folder, paths_only=True, fmt="png")
        return image_paths
    except Exception as e:
        logging.error(f"Error converting PDF to images: {e}")
        return []


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")