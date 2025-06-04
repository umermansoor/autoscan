"""
AutoScan - High fidelity PDF to Markdown conversion using LLMs.

Convert PDF files to well-structured Markdown with high accuracy using
GPT-4o, Gemini, and other LLMs. Perfect for documents with tables, images,
and complex layouts.
"""

__version__ = "0.1.0"
__author__ = "Umer Mansoor"
__email__ = "umermk3@gmail.com"

from .autoscan import autoscan
from .types import AutoScanOutput, ModelResult

__all__ = ["autoscan", "AutoScanOutput", "ModelResult"]