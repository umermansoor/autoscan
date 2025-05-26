#!/usr/bin/env python3
"""
Integration test script for AutoScan with Gemini 2.0 Flash model.

Processes example PDFs with safety limits: max 5 files, max 15 pages/file.
Alternates between low/high accuracy modes and reports comprehensive stats.

Usage: python tests/integration/test_examples_integration.py
"""

import asyncio
import os
import sys
import time
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Setup imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from autoscan.autoscan import autoscan


class IntegrationTestRunner:
    def __init__(self):
        self.model_name = "gemini/gemini-2.0-flash"
        self.examples_dir = project_root / "examples"
        self.output_dir = project_root / "output"
        self.MAX_FILES = 5
        self.MAX_PAGES_PER_FILE = 15
        self.results = []
        
    def get_pdf_page_count(self, pdf_path: Path) -> Optional[int]:
        """Get PDF page count using pdfinfo."""
        try:
            result = subprocess.run(['pdfinfo', str(pdf_path)], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Pages:'):
                        return int(line.split()[-1])
        except Exception:
            pass
        return None
    
    def filter_pdfs(self, pdf_files: List[Path]) -> List[Path]:
        """Apply safety filters to PDF files."""
        print("🛡️  Applying safety filters...")
        
        # Limit file count
        if len(pdf_files) > self.MAX_FILES:
            pdf_files = pdf_files[:self.MAX_FILES]
            print(f"   📊 Limited to {self.MAX_FILES} files for cost control")
        
        # Filter by page count
        filtered = []
        for pdf_file in pdf_files:
            page_count = self.get_pdf_page_count(pdf_file)
            if page_count and page_count <= self.MAX_PAGES_PER_FILE:
                print(f"   ✅ {pdf_file.name}: {page_count} pages")
                filtered.append(pdf_file)
            else:
                pages_str = f"{page_count} pages" if page_count else "unknown pages"
                print(f"   ⏭️  Skipping {pdf_file.name}: {pages_str}")
        
        print(f"   🎯 Final selection: {len(filtered)} files")
        return filtered
    
    def clear_outputs(self, pdf_files: List[Path]):
        """Clear existing output files."""
        print("🗑️  Clearing existing outputs...")
        self.output_dir.mkdir(exist_ok=True)
        
        for pdf_file in pdf_files:
            md_file = self.output_dir / f"{pdf_file.stem}.md"
            if md_file.exists():
                md_file.unlink()
                print(f"   🗑️  Removed: {md_file.name}")
    
    async def process_pdf(self, pdf_path: Path, accuracy_mode: str) -> Dict[str, Any]:
        """Process a single PDF and return results."""
        print(f"🔄 Processing: {pdf_path.name} ({accuracy_mode})")
        start_time = time.time()
        
        try:
            result = await autoscan(
                pdf_path=str(pdf_path),
                model_name=self.model_name,
                accuracy=accuracy_mode,
                output_dir=str(self.output_dir)
            )
            
            processing_time = time.time() - start_time
            output_file = Path(result.markdown_file)
            is_valid = output_file.exists() and len(result.markdown) > 10
            
            if is_valid:
                print(f"   ✅ Success: {processing_time:.2f}s, ${result.cost:.4f}, {len(result.markdown):,} chars")
            else:
                print(f"   ❌ Failed: Output validation failed")
            
            return {
                "filename": pdf_path.name,
                "accuracy_mode": accuracy_mode,
                "status": "✅ SUCCESS" if is_valid else "❌ FAILED",
                "processing_time": processing_time,
                "cost": result.cost,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "content_length": len(result.markdown),
                "is_valid": is_valid,
                "error": None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"   ❌ Error: {str(e)}")
            
            return {
                "filename": pdf_path.name,
                "accuracy_mode": accuracy_mode,
                "status": "❌ ERROR",
                "processing_time": processing_time,
                "cost": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "content_length": 0,
                "is_valid": False,
                "error": str(e)
            }
    
    def print_summary(self, results: List[Dict], total_time: float):
        """Print concise test results."""
        total_files = len(results)
        successful = sum(1 for r in results if r["is_valid"])
        total_cost = sum(r["cost"] for r in results)
        total_tokens = sum(r["input_tokens"] + r["output_tokens"] for r in results)
        
        print("\n📊 INTEGRATION TEST RESULTS")
        print("=" * 80)
        print(f"Files Processed: {total_files} | Success: {successful}/{total_files}")
        print(f"Total Time: {total_time:.2f}s | Total Cost: ${total_cost:.6f}")
        print(f"Total Tokens: {total_tokens:,}")
        print(f"Safety Limits: Max {self.MAX_FILES} files, {self.MAX_PAGES_PER_FILE} pages/file")
        
        print(f"\n{'Filename':<20} {'Mode':<6} {'Status':<12} {'Time':<6} {'Cost':<10} {'Tokens':<8}")
        print("-" * 80)
        
        for r in results:
            tokens = r["input_tokens"] + r["output_tokens"]
            print(f"{r['filename'][:19]:<20} {r['accuracy_mode']:<6} {r['status']:<12} "
                  f"{r['processing_time']:.1f}s{'':<2} ${r['cost']:.6f} {tokens:,}")
        
        print("-" * 80)
        if successful == total_files:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {total_files - successful} tests failed")
        print(f"Model: {self.model_name} | Execution time: {total_time:.2f}s")
        
    async def run_all_tests(self):
        """Main test runner."""
        print("🚀 AutoScan Integration Test Suite")
        print(f"Model: {self.model_name}")
        
        # Find and filter PDFs
        pdf_files = list(self.examples_dir.glob("*.pdf"))
        if not pdf_files:
            print("❌ No PDF files found!")
            return
            
        print(f"\n📄 Found {len(pdf_files)} PDFs: {[f.name for f in pdf_files]}")
        
        pdf_files = self.filter_pdfs(pdf_files)
        if not pdf_files:
            print("❌ No files passed safety filters!")
            return
        
        self.clear_outputs(pdf_files)
        
        # Process files
        print(f"\n🔄 Processing {len(pdf_files)} files...")
        start_time = time.time()
        
        for index, pdf_file in enumerate(pdf_files):
            accuracy_mode = "low" if index % 2 == 0 else "high"
            result = await self.process_pdf(pdf_file, accuracy_mode)
            self.results.append(result)
        
        total_time = time.time() - start_time
        self.print_summary(self.results, total_time)


async def main():
    """Main entry point for the integration test."""
    # Set up logging to reduce noise during testing
    logging.basicConfig(level=logging.WARNING)
    for logger_name in ["LiteLLM", "httpx", "httpcore"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Check if GEMINI_API_KEY is set
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY environment variable is not set!")
        print("Please set it with: export GEMINI_API_KEY='your_api_key'")
        sys.exit(1)
    
    # Run the integration tests
    runner = IntegrationTestRunner()
    await runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
