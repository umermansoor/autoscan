DEFAULT_SYSTEM_PROMPT = (
    "Convert the following PDF page to Markdown. "
    "Include all text, tables, lists, and images. "
    "Use proper Markdown syntax for lists and tables, and embed images with meaningful alt text. "
    "Return plain Markdown only - do not wrap the output in code fences or add explanations."
)

DEFAULT_SYSTEM_PROMPT_IMAGE_TRANSCRIPTION = """
Your task is to accurately convert the content of a single PDF page into properly structured Markdown format.  The conversion must include all textual elements, as well as tables and image descriptions. Do not provide any explanation or commentary outside of the Markdown output. 

## Instructions

1. **General Guidelines:**
    - Convert **all** meaningful textual content from the PDF page into Markdown. This includes: 
        - Headings, paragraphs, lists, footnotes, side notes, captions, and any other text blocks.
        - **Do not** include page numbers or repetitive headers/footers with no meaningful content. 
    - If underlining or other emphasis can't be directly represented, default to italics  (`*...*` or `_..._`).
    - Retain URLs, but convert them to Markdown links: `[link text](url)`
    - Use `>` for blockquotes and sidebars.
    - Use triple backticks (```) for code blocks or preformatted text. 
        - If you know the language (e.g. JSON, Python), specify it right after the opening backticks:

2. **Multi-Column Layouts:**
    - If the PDF page is presented in multiple columns, read the text in a logical order (e.g., top to bottom of the left column, then top to bottom of the next column, and so forth).
    - Integrate the text from all columns into a single, logically flowing narrative. Do not preserve columns as separate sections unless they represent distinct logical divisions (e.g., sidebars).

3. **Mathematical Equations and Formulas:**
    - Enclose all mathematical content in `$$...$$`.
    - Example: `$$E = mc^2$$`

4. **Tables**
    - If the PDF contains tables—whether originally text-based or extracted from images—convert them into Markdown tables.

5. **Images and Visuals:** The resulting markdown will be processed by a large language model that does not interpret images. Therefore, you should provide detailed descriptions of any images or visual content in the PDF and do NOT embed images using `![image](url)`.
    - Provide a **detailed blockquoted description** of each image or figure:
       > **Image Description**: Then follow with a comprehensive textual explanation, including the image's purpose, layout, labels, colors, and any relevant details.
    - For complex visuals, use additional lists or paragraphs within the blockquote to clarify content.
    - If the image is simply a textual table (or similar) that you've already transcribed into Markdown, do not add a duplicate image description.

6. **Previous Page Context**
    - If you have Markdown from previous pages, ensure your text flows consistently. Avoid duplicating content that was already captured.

**Your response must follow these rules and contain only the converted Markdown content.**
    """

FINAL_REVIEW_PROMPT = """
You have been given a combined Markdown document that represents multiple individually processed PDF pages. The current Markdown may contain inconsistencies in headings, paragraph breaks, and formatting due to a page-by-page conversion process.

**Your Task:**
- Clean, consolidate, and refine the Markdown to create a single coherent, well-structured document.
- Do not add or remove substantive content, but reorganize and correct formatting where necessary.
- Ensure headings follow a logical hierarchy throughout the entire document.
- Merge or reorganize paragraphs as needed so the text flows naturally, removing unnecessary line breaks.
- Standardize lists, tables, footnotes, image descriptions, code blocks, and other elements to ensure consistent formatting.
- Remove any `---PAGE BREAK---` markers used to separate pages.
- If headers or footers appear repeatedly, consolidate them if appropriate, or remove duplicates that do not contribute meaningfully to the final text.
- Maintain all meaningful information, but eliminate redundancies, correct obvious formatting errors, and ensure consistent Markdown syntax.
- If you encounter any LaTeX or mathematical expressions, ensure they are correctly formatted for Katex renderer. The "\\( ... \\)" formatting doesn't render correctly in Markdown. Remove "\\( ... \\)" formatting and replace with "$$...$$" which works in Katex.

**Important:**
- Return only the revised Markdown.
- Do not add commentary or explanations about the changes you made.
"""