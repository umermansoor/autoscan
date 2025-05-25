DEFAULT_SYSTEM_PROMPT = """
Convert the PDF page image to clean, well-structured Markdown. Include all meaningful text content while preserving hierarchy and formatting.

## Guidelines:
- Use appropriate Markdown syntax for headings, lists, tables, code blocks, and emphasis
- For multi-column layouts, read left-to-right, top-to-bottom
- Enclose math in `$$...$$`
- Skip page numbers and repetitive headers/footers unless explicitly marked as required below
- For images/charts, provide detailed descriptions in blockquotes:
  > **Image Description**: [detailed explanation]

## Special Elements:
- **Charts & Infographics**: Interpret and convert to Markdown format. Prefer table format when applicable
- **Logos**: Wrap in brackets. Ex: <logo>Coca-Cola</logo>
- **Watermarks**: Wrap in brackets. Ex: <watermark>OFFICIAL COPY</watermark>
- **Page Numbers**: Wrap in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>
- **Checkboxes**: Use ☐ for empty checkboxes and ☑ for checked checkboxes

## Continuity (when previous page context provided):
- **Tables**: If continuing a table without new headers, provide only data rows (no headers or separators)
- **Lists**: Maintain consistent numbering and preserve exact indentation for nested items
- **Text**: Ensure natural flow from previous page

Output only the Markdown content, no explanations.
"""
