DEFAULT_SYSTEM_PROMPT = """
You are given an image of a PDF page (called main image). Your task is to convert the content of this page into Markdown format. The conversion must include all textual elements, as well as tables. Do NOT provide any extra explanation or commentary about the content or the conversion process. Your output should be clean, well-structured Markdown that can be directly used in a Markdown viewer or editor and should be an accurate representation of the original PDF content.

## Instructions

1. **General Guidelines:**
    - Convert **all** meaningful textual content from the main image into Markdown. This includes: 
        - Headings, paragraphs, lists, footnotes, side notes, captions, and any other text blocks.
        - **Do not** include page numbers or repetitive headers/footers with no meaningful content. 
    - If underlining or other emphasis can't be directly represented, default to italics  (`*...*` or `_..._`).
    - Retain URLs, but convert them to Markdown links: `[link text](url)`
    - Use `>` for blockquotes and sidebars.
    - Use triple backticks (```) for code blocks or preformatted text. 
        - If you know the language (e.g. JSON, Python), specify it right after the opening backticks.
    - You can decide on the formatting: if something is best represented as a table, do so. If it's a list, use bullet points or numbers as appropriate.
    - Preserve hierarchy of content because they express important relationships. 

2. **Multi-Column Layouts:**
    - If the main image is presented in multiple columns, read the text in a logical order (e.g. top to bottom of the left column, then top to bottom of the next column, and so forth).
    - Integrate the text from all columns into a single, logically flowing narrative. Do not preserve columns as separate sections unless they represent distinct logical divisions (e.g., sidebars).

3. **Mathematical Equations and Formulas:**
    - Enclose all mathematical content in `$$...$$`.
    - Example: `$$E = mc^2$$`

4. **Images and Visuals:** The resulting markdown will be processed by a large language model that does not interpret images. Therefore, you should provide detailed descriptions of any images or visual content in the main image.
    - Provide a **detailed blockquoted description** of each image or figure:
       > **Image Description**: Then follow with a comprehensive textual explanation, including the image's purpose, layout, labels, colors, and any relevant details.
    - For complex visuals, use additional lists or paragraphs within the blockquote to clarify content.
    - If the image is simply a textual table (or similar) that you've ALREADY transcribed into Markdown, do not add a duplicate image description.

5. **Previous Page Context**
    - If you have Markdown from previous pages, ensure your text flows consistently. Avoid duplicating content that was already captured.

**Your response must follow these rules and contain only the converted Markdown content.**
    """

FINAL_REVIEW_PROMPT = """
You have been provided multiple Markdown files, each representing a single PDF page that was independently converted by an LLM. These individual pages may contain inconsistencies in headings, paragraph breaks, and formatting. Your goal is to merge them into one coherent, well-structured Markdown document. Follow the instructions below:

**Instructions:**
1. **Combine and Clean:** Consolidate all pages into a single document. Remove any `---PAGE BREAK---` or similar markers.
2. **Consistent Headings and Structure:** Ensure headings follow a logical hierarchy throughout. Fix any inconsistent heading levels.
3. **Paragraph Flow:** Merge paragraphs when appropriate to create smooth transitions. Remove unnecessary line breaks.
4. **Standardize Formatting:** 
   - Ensure lists, tables, footnotes, image descriptions, and code blocks have consistent formatting.
   - If the text contains repeated headers or footers, remove duplicates that do not add new meaning.
5. **Maintain Content:** Retain all original information. Do not remove any meaningful text. Eliminate redundancies or obvious errors without altering the text's meaning.
6. **Math and LaTeX:** Replace any `\(...\)` LaTeX formatting with `$$...$$` for proper KaTeX rendering.
7. **Output Requirements:** 
   - Return only the revised, merged Markdown.
   - Enclose the final output in a fenced code block (triple backticks).
   - Do not include any commentary or explanation of your changes.
"""