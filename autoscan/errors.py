class PDFFileNotFoundError(Exception):
    """Raised when the PDF file is not found."""
    pass

class PDFPageToImageConversion(Exception):
    """Raised when the PDF file pages cannot be converted to images."""
    pass

class MarkdownFileWriteError(Exception):
    """Raised when the markdown file cannot be written."""
    pass