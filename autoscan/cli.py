import argparse
import asyncio
import logging
import os
import sys

from .autoscan import autoscan
from .utils.env import get_env_var_for_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def _process_file(
    pdf_path: str,
    model: str,
    accuracy: str,
    instructions: str | None = None,
) -> None:
    logging.info(f"Processing file: {pdf_path}")
    await autoscan(
        pdf_path=pdf_path,
        model_name=model,
        accuracy=accuracy,
        user_instructions=instructions,
    )

async def _run(
    pdf_path: str | None = None,
    model: str = "openai/gpt-4o",
    accuracy: str = "medium",
    instructions: str | None = None,
) -> None:
    if pdf_path:
        await _process_file(pdf_path, model, accuracy, instructions)
    else:
        logging.error("No valid input provided. Use --help for usage information.")
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run autoscan on a PDF file"
    )
    parser.add_argument("pdf_path", nargs="?", help="Path to a single PDF file")

    parser.add_argument(
        "--accuracy",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Conversion accuracy level",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Model name to use with LiteLLM",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        help="Optional instructions passed to the LLM",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    env_var = get_env_var_for_model(args.model)
    if env_var and not os.environ.get(env_var):
        logging.error(
            f"{env_var} is not defined. "
            "Please set it as an environment variable. "
            f"For example: export {env_var}='YOUR_API_KEY'"
        )
        sys.exit(1)

    if not args.pdf_path:
        parser.print_help()
        return

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(
        _run(
            pdf_path=args.pdf_path,
            model=args.model,
            accuracy=args.accuracy,
            instructions=args.instructions,
        )
    )

if __name__ == "__main__":
    main()
