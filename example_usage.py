import asyncio
from autoscan.autoscan import autoscan
import logging
import argparse
import os
import sys

# Configure the logging system in the application
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def process_file(pdf_path):
    logging.info(f"Processing file: {pdf_path}")
    await autoscan(pdf_path=pdf_path, contextual_conversion=False)

async def main(pdf_path=None, directory=None):
    if directory:
        # Process all PDF files in the directory
        logging.info(f"Processing all PDF files in directory: {directory}")
        for file_name in os.listdir(directory):
            if file_name.lower().endswith(".pdf"):
                file_path = os.path.join(directory, file_name)
                logging.info(f"Processing file: {file_path}")
                await process_file(file_path)
    elif pdf_path:
        # Process a single PDF file
        logging.info(f"Processing single PDF file: {pdf_path}")
        await process_file(pdf_path)
    else:
        logging.error("No valid input provided. Use --help for usage information.")
        sys.exit(1)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run autoscan on a PDF file or all PDF files in a directory.")
    parser.add_argument("--pdf_path", type=str, help="Path to a single PDF file")
    parser.add_argument("--directory", type=str, help="Path to a directory containing PDF files")

    # Show help if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    # Start the asyncio event loop
    asyncio.run(main(pdf_path=args.pdf_path, directory=args.directory))