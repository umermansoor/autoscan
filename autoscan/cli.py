import argparse
import asyncio
import logging
import os
import sys

from .autoscan import autoscan

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def _process_file(pdf_path: str, contextual_conversion: bool) -> None:
    logging.info(f"Processing file: {pdf_path}")
    await autoscan(pdf_path=pdf_path, contextual_conversion=contextual_conversion)

async def _run(pdf_path: str | None = None, directory: str | None = None, contextual_conversion: bool = False) -> None:
    if directory:
        logging.info(f"Processing all PDF files in directory: {directory}")
        for file_name in os.listdir(directory):
            if file_name.lower().endswith(".pdf"):
                await _process_file(os.path.join(directory, file_name), contextual_conversion)
    elif pdf_path:
        await _process_file(pdf_path, contextual_conversion)
    else:
        logging.error("No valid input provided. Use --help for usage information.")
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run autoscan on a PDF file or all PDF files in a directory."
    )
    parser.add_argument("pdf_path", nargs="?", help="Path to a single PDF file")
    parser.add_argument(
        "--directory",
        type=str,
        help="Path to a directory containing PDF files",
    )
    parser.add_argument(
        "--contextual-conversion",
        action="store_true",
        help="Use previous page markdown when processing subsequent pages",
    )

    args = parser.parse_args()

    # Check if OPENAI_API_KEY is defined
    if not os.environ.get("OPENAI_API_KEY"):
        logging.error(
            "OPENAI_API_KEY is not defined. "
            "Please set it as an environment variable. "
            "For example: export OPENAI_API_KEY='YOUR_API_KEY'"
        )
        sys.exit(1)

    if not args.pdf_path and not args.directory:
        parser.print_help()
        return

    asyncio.run(
        _run(
            pdf_path=args.pdf_path,
            directory=args.directory,
            contextual_conversion=args.contextual_conversion,
        )
    )

if __name__ == "__main__":
    main()
