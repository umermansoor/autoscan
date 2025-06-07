IMG_TO_MARKDOWN_PROMPT = """
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
- **Tables**: CRITICAL - If a table continues from the previous page, DO NOT repeat headers or table separators. Only provide the data rows that continue the table. NEVER add separator lines (|---|---|) when continuing a table - only provide data rows starting with |. Look for incomplete tables in the previous page context to determine if this page continues that table.
- **Lists**: Continue numbering from where the previous page left off. Maintain consistent formatting and indentation for nested items.
- **Text**: Ensure seamless flow by avoiding repetitive introductions or section breaks that duplicate previous page content.
- **Page Breaks**: When content flows across pages, treat it as one continuous document. Do not restart formatting or add unnecessary breaks.

Output only the Markdown content, no explanations. Do not include delimiters like ```markdown or ```html.
"""

POST_PROCESSING_PROMPT = """
You are provided with a Markdown document generated via OCR from a PDF. The source may contain errors such as:

- Inconsistent formatting  
- Split or fragmented sentences and paragraphs  
- Misaligned or broken tables  
- Repeated headers, footers, or page numbers  
- Arbitrary line and page breaks  

## OBJECTIVE

Reconstruct the document into a complete, accurate, and professionally formatted Markdown version that preserves **100 percent of the original content**, in the **original wording and structure**. You are functioning as a **structural restorer**, not a summarizer, paraphraser, or editor.

---

## ABSOLUTE RULES

1. **DO NOT SUMMARIZE**  
   All original text — including examples, anecdotes, exercises, and prompts — must be preserved **word-for-word**.

2. **DO NOT PARAPHRASE OR REPHRASE**  
   Retain the **exact language** of the original document. Do not simplify or rewrite any part of the content.

3. **DO NOT OMIT ANYTHING**  
   Every sentence, bullet point, question, reflection, and instructional block must appear in the output.

4. **DO NOT ADD ANY CONTENT OR EXPLANATION**  
   Do not include transitions, summaries, clarifications, or any material not present in the original.

5. **DO NOT REARRANGE SECTIONS**  
   Preserve the original **section order and flow**. Only rejoin content that was split unnaturally across pages.

6. **PRESERVE FORMATTING AND STRUCTURE**  
   Use Markdown syntax appropriately (`#`, `##`, `-`, `*`, `>`, etc.) to recreate original headers, bullet points, blockquotes, and emphasis. Reflect original indentation and hierarchy.

---

## RESTORATION GUIDELINES

- **Fix OCR Line Breaks**: Merge broken sentences and paragraphs.
- **Remove Redundant Elements**: Eliminate repeated page headers, footers, and numbers.
- **Rebuild Tables**: Reconstruct tables that were split or corrupted.
- **Correct Typographic Errors Only**: You may fix obvious OCR errors (e.g., misspelled words, punctuation issues), but do **not alter content**.
- **Respect Original Layout**: Reproduce sections, headings, lists, and notes exactly as presented in the source.

---

## SPECIAL NOTE ON INSTRUCTIONAL MATERIALS

When working with training manuals, handbooks, or learning guides:

- DO NOT modify instructions, lists, steps, or exercises  
- Preserve reflective questions and practical examples in full  
- Accurately represent dialogues, scenarios, and checklists  
- Format any embedded worksheets, summaries, or tips as they originally appear  

---

## OUTPUT FORMAT

- Output a **single clean Markdown document**, fully restored.
- **Do NOT** wrap the output in code fences (no triple backticks).
- **Do NOT** include any commentary, summaries, or processing notes.
- This is a **restoration task**, not a summarization or editorial task.

Failure to follow **any** of the rules above will result in an incomplete or invalid restoration.
"""