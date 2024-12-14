DEFAULT_SYSTEM_PROMPT = (
    "Your job is to convert the following PDF page to markdown. "
    "You must convert the entire page to markdown including all text, tables, etc. "
    "Only return the markdown with no explanation."
)

DEFAULT_SYSTEM_PROMPT_IMAGE_TRANSCRIPTION = """
Your task is to accurately convert the content of a PDF page into properly structured Markdown format. 
The conversion must include all elements of the page, such as text, tables, and images. 
Do not provide any explanation or commentary in your output; return only the Markdown.

## Instructions:
1. Comprehensive Conversion:
    - Convert the entire page's content, including headers, paragraphs, lists, and tables, into Markdown format.
    - Use proper Markdown syntax to ensure readability and accurate formatting.

2. Mathematical Equations and Formulas:
    - LaTeX formatting should be enclosed in `$$...$$` so it can be rendered correctly using KaTeX e.g. $$E=mc^2$$
    - Do these are all equations in the text, or image descriptions.

3. Handling Images and Visuals:
    - If the page includes images, such as charts, graphs, diagrams, or illustrations:
        - Do NOT embed images using Markdown syntax (e.g., ![image](image_url)).
        - Instead, provide a descriptive summary of the image in Markdown, formatted as a blockquote (>).
        - The description should be detailed enough to convey the image's purpose, layout, key data, labels, and context.
    - Example of an image description:
        > **Image Description**: This image depicts a line graph showing annual sales growth from 2018 to 2023. The X-axis represents the years, and the Y-axis represents sales in millions of dollars. The graph shows steady growth, starting at $5M in 2018 and reaching $25M by 2023, with notable peaks in 2020 and 2022.
    - For complex visuals, provide structured descriptions using lists or sections within blockquotes for clarity.

4. Do not exclude any content from the page during conversion such as footers, side notes, or captions. Capture everything.

Output Requirements:
- Your response must consist of Markdown only.
- Use standard Markdown syntax only.
- Follow all formatting rules for text, tables, and image descriptions.
- Do not include explanatory text or meta-comments outside the Markdown.
    """