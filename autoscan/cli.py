import argparse
import asyncio
import logging
import os
import sys

from .autoscan import autoscan
from .utils.env import get_env_var_for_model

async def _process_file(
    pdf_path: str,
    model: str,
    accuracy: str,
    prompt: str | None = None,
    output_dir: str | None = None,
    save_llm_calls: bool = False,
    temp_dir: str | None = None,
    polish_output: bool = False,
    first_page: int | None = None,
    last_page: int | None = None,
) -> None:
    await autoscan(
        pdf_path=pdf_path,
        model_name=model,
        accuracy=accuracy,
        user_instructions=prompt,
        output_dir=output_dir,
        save_llm_calls=save_llm_calls,
        temp_dir=temp_dir,
        polish_output=polish_output,
        first_page=first_page,
        last_page=last_page,
    )

async def _run(
    pdf_path: str | None = None,
    model: str = "openai/gpt-4o",
    accuracy: str = "high",
    prompt: str | None = None,
    output_dir: str | None = None,
    save_llm_calls: bool = False,
    temp_dir: str | None = None,
    polish_output: bool = False,
    first_page: int | None = None,
    last_page: int | None = None,
) -> None:
    if pdf_path:
        await _process_file(pdf_path, model, accuracy, prompt, output_dir, save_llm_calls, temp_dir, polish_output, first_page, last_page)
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
        choices=["low", "high"],  
        default="high",  
        help="Conversion accuracy level",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o",
        help="Model name to use with LiteLLM",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Optional instructions passed to the LLM",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save the output markdown file (defaults to ./output/)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--save-llm-calls",
        action="store_true",
        help="Save LLM prompts and responses to logs/ directory",
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        help="Directory for storing temporary images (user is responsible for cleanup)",
    )
    parser.add_argument(
        "--polish-output",
        action="store_true",
        help="Apply additional LLM pass to improve formatting and document structure",
    )
    parser.add_argument(
        "--first-page",
        type=int,
        help="First page to process (defaults to processing from the beginning)",
    )
    parser.add_argument(
        "--last-page",
        type=int,
        help="Last page to process before stopping (defaults to processing to the end)",
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

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")
    
    # Suppress LiteLLM logging to reduce noise
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    # Suppress HTTP request logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    asyncio.run(
        _run(
            pdf_path=args.pdf_path,
            model=args.model,
            accuracy=args.accuracy,
            prompt=args.prompt,
            output_dir=args.output_dir,
            save_llm_calls=args.save_llm_calls,
            temp_dir=args.temp_dir,
            polish_output=args.polish_output,
            first_page=args.first_page,
            last_page=args.last_page,
        )
    )

if __name__ == "__main__":
    main()
