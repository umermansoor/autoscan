DEFAULT_SYSTEM_PROMPT = """
You are given an image of a PDF page (called main image). Your task is to convert the content of this page into Markdown format. The conversion must include all textual elements, as well as tables. Do NOT provide any extra explanation or commentary about the content or the conversion process. Your output should be clean, well-structured Markdown that can be directly used in a Markdown viewer or editor and should be an accurate representation of the original PDF content.

## Instructions

1. **General Guidelines:**
    - Convert **all** meaningful textual content from the main image into Markdown. This includes: 
        - Headings, paragraphs, lists, footnotes, side notes, captions, and any other text blocks. You'll need to infer these from the layout and make best possible formatting decisions. E.g. if you notice a list implicitly e.g. the items are indented, or if you see a heading, or a caption, etc. Use appropriate Markdown syntax for each type of content:
        - **Do not** include page numbers or repetitive headers/footers with no meaningful content or have already been captured.
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

HIGH_ACCURACY_PROMPT = (
    DEFAULT_SYSTEM_PROMPT
    + """

When provided with Markdown from a previous page, rewrite that page together
with the current page so that tone, formatting, and layout are consistent.
Return your answer as JSON:

{
  "page_1": "<revised previous page>",
  "page_2": "<revised current page>"
}

If no previous page is given, return plain Markdown as usual.
"""
)
