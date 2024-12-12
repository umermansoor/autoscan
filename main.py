import os
from typing import List
from autoscan.image_processing import pdf_to_images
from autoscan.model import LlmModel

def process_pdf_to_markdown(
    pdf_path: str, 
    output_file: str, 
    model_name: str = "gpt-4o"
) -> None:
    """
    Convert a PDF to markdown by:
    1. Converting PDF pages into images.
    2. Using an LLM to process each page image into markdown.
    3. Aggregating all markdown pages into a single file.

    Args:
        pdf_path (str): The path to the PDF file.
        output_file (str): The path where the aggregated markdown will be stored.
        model_name (str, optional): The name of the model to use for the LLM. Defaults to "gpt-4o".
    """
    # Create a temporary directory for images
    temp_directory = "temp_images"
    os.makedirs(temp_directory, exist_ok=True)

    # Convert the PDF to a list of page images
    images: List[str] = pdf_to_images(pdf_path, temp_directory)

    # Initialize the model
    model = LlmModel(model_name=model_name)

    aggregated_markdown = []
    prior_page_markdown = ""

    # Process each page image with the LLM
    for image_path in images:
        page_markdown = model.completion(image_path, prior_page_markdown)
        aggregated_markdown.append(page_markdown)
        # Store the current page markdown to maintain formatting in the next page (if needed)
        prior_page_markdown = page_markdown

    # Write the aggregated markdown to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(aggregated_markdown))

    # Cleanup: remove the generated images and temp directory
    for img in images:
        os.remove(img)
    os.rmdir(temp_directory)


if __name__ == "__main__":
    # Example usage
    pdf_path = "examples/independence.pdf"
    output_file = "output.md"

    process_pdf_to_markdown(pdf_path, output_file)
    print(f"Aggregated markdown saved to {output_file}")
