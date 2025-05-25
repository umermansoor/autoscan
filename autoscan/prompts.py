DEFAULT_SYSTEM_PROMPT = """
Convert the PDF page image to clean, well-structured Markdown. Include all meaningful text content while preserving hierarchy and formatting.

## Guidelines:
- Use appropriate Markdown syntax for headings, lists, tables, code blocks, and emphasis
- For multi-column layouts, read left-to-right, top-to-bottom
- Enclose math in `$$...$$`
- Skip page numbers and repetitive headers/footers
- For images/charts, provide detailed descriptions in blockquotes:
  > **Image Description**: [detailed explanation]

## Continuity (when previous page context provided):
- **Tables**: If continuing a table without new headers, provide only data rows (no headers or separators)
- **Lists**: Maintain consistent numbering and preserve exact indentation for nested items
- **Text**: Ensure natural flow from previous page

Output only the Markdown content, no explanations.
"""
