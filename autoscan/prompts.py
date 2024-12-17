DEFAULT_SYSTEM_PROMPT = (
    "Your job is to convert the following PDF page to markdown. "
    "You must convert the entire page to markdown including all text, tables, etc. "
    "Only return the markdown with no explanation."
)

DEFAULT_SYSTEM_PROMPT_IMAGE_TRANSCRIPTION = """
Your task is to accurately convert the content of a single PDF page into properly structured Markdown format.  The conversion must include all textual elements, as well as tables and image descriptions. Do not provide any explanation or commentary outside of the Markdown output. 

## Instructions

1. **General Guidelines:**
    - Convert **all** textual content from the PDF page into Markdown, including headers, paragraphs, lists, tables, footnotes, side notes, captions, headers, and footers.
    - Do not omit or exclude any content, except:
        - Headers and footers that do not contain meaningful information.
        - Page numbers.
    - If underlining or other emphasis can't be directly represented, default to italics.
    - Retain URLs, but convert them to Markdown links.
    - Use `>` for blockquotes and sidebars.
    - Use triple backticks (```) for code blocks or preformatted text.
        - If you detect that the code or snippet is from a particular programming language, specify it after the first set of backticks, e.g., ```python or ```json.

2. **Multi-Column Layouts:**
    - If the PDF page is presented in multiple columns, read the text in a logical order (e.g., top to bottom of the left column, then top to bottom of the next column, and so forth).
    - Integrate the text from all columns into a single, logically flowing narrative. Do not preserve columns as separate sections unless they represent distinct logical divisions (e.g., sidebars).

3. **Mathematical Equations and Formulas:**
    - Enclose all mathematical content in `$$...$$`.
    - Example: `$$E = mc^2$$`

4. **Images and Visuals:** The resulting markdown will be processed by a large language model that does not interpret images. Therefore, you should provide detailed descriptions of any images or visual content in the PDF and do NOT embed images using `![image](url)`.
    - Provide a **detailed blockquoted description** of each image or figure:
       > **Image Description**: Then follow with a comprehensive textual explanation, including the image's purpose, layout, labels, colors, and any relevant details.
    - For complex visuals, use additional lists or paragraphs within the blockquote to clarify content.

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
- If headers or footers appear repeatedly, consolidate them if appropriate, or remove duplicates that do not contribute meaningfully to the final text.
- Maintain all meaningful information, but eliminate redundancies, correct obvious formatting errors, and ensure consistent Markdown syntax.
- If you encounter any LaTeX or mathematical expressions, ensure they are correctly formatted for Katex renderer. The "\\( ... \\)" formating doesn't render correctly in Markdown. Remove "\\( ... \\)" formatting and replace with "$$...$$" which works in Katex.

**Important:**
- Return only the revised Markdown.
- Do not add commentary or explanations about the changes you made.
"""