import asyncio
from datetime import datetime
from pathlib import Path

import fitz


def generate_filename(pdf_path):
    pdf_name = Path(pdf_path).stem
    # satinize name
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in pdf_name)
    return f"{safe_name}.md"


def extract_metadata(pdf_doc, pdf_path):
    metadata = pdf_doc.metadata or {}

    return {
        "source_file": str(Path(pdf_path).name),
        "title": metadata.get("title", "Unknown"),
        "author": metadata.get("author", "Unknown"),
        "pages": pdf_doc.page_count,
        "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def extract_page_text_safe(page, page_num):
    """
    Extract text from a page handling different return types

    Args:
        page: PyMuPDF page object
        page_num: Page number (0-indexed)

    Returns:
        String with page text or None if failed
    """
    try:
        # get_text() without args returns str (default)
        text = page.get_text()

        # Handle different return types
        if isinstance(text, str):
            return text if text.strip() else None
        elif isinstance(text, (list, dict)):
            # If it returns list or dict for some reason, convert to string
            return str(text)
        else:
            print(f"⚠️  Unexpected type on page {page_num + 1}: {type(text)}")
            return None

    except Exception as e:
        print(f"⚠️  Error extracting text from page {page_num + 1}: {e}")
        return None


async def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF (CPU-bound operation run in executor)

    Args:
        pdf_path: Path to PDF file

    Returns:
        Tuple of (text_content_list, metadata) or (None, None) if failed
    """
    # Run blocking PDF operations in thread pool
    loop = asyncio.get_event_loop()

    def _extract():
        # Try to open PDF
        try:
            doc = fitz.open(pdf_path)
        except FileNotFoundError:
            print(f"✗ File not found: {pdf_path}")
            return None, None
        except fitz.FileDataError:
            print(f"✗ Corrupted or invalid PDF: {pdf_path}")
            return None, None
        except PermissionError:
            print(f"✗ Permission denied: {pdf_path}")
            return None, None
        except Exception as e:
            print(f"✗ Error opening PDF: {e}")
            return None, None

        # Extract metadata
        metadata = extract_metadata(doc, pdf_path)

        # Extract text from all pages
        text_content = []

        with doc:
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = extract_page_text_safe(page, page_num)

                if text:
                    text_content.append(f"## Page {page_num + 1}\n")
                    text_content.append(text)
                    text_content.append("\n")

        return text_content, metadata

    # Run in thread pool (PDF operations are CPU-bound)
    return await loop.run_in_executor(None, _extract)


async def pdf_to_markdown(pdf_path, output_dir="output/pdfs"):
    """
    Convert a PDF to Markdown with metadata (async)

    Args:
        pdf_path: Path to PDF file
        output_dir: Output directory

    Returns:
        Dict with operation result
    """
    print(f"📄 Processing: {Path(pdf_path).name}")

    try:
        # Extract text and metadata (CPU-bound, runs in executor)
        text_content, metadata = await extract_text_from_pdf(pdf_path)

        if text_content is None or metadata is None:
            return {
                "pdf_path": str(pdf_path),
                "output_path": None,
                "success": False,
                "error": "Failed to extract content",
            }

        # Create Markdown content
        markdown_content = f"""---
source_file: {metadata["source_file"]}
title: {metadata["title"]}
author: {metadata["author"]}
pages: {metadata["pages"]}
extracted_at: {metadata["extracted_at"]}
---

# {metadata["title"]}

"""

        markdown_content += "\n".join(text_content)

        # Save to file (I/O-bound, but fast enough to not need executor)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = generate_filename(pdf_path)
        filepath = output_path / filename

        # Use async file write
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: filepath.write_text(markdown_content, encoding="utf-8")
        )

        print(f"✓ Saved: {filename}")

        return {
            "pdf_path": str(pdf_path),
            "output_path": str(filepath),
            "success": True,
            "pages": metadata["pages"],
        }

    except Exception as e:
        print(f"✗ Error processing {Path(pdf_path).name}: {e}")
        return {
            "pdf_path": str(pdf_path),
            "output_path": None,
            "success": False,
            "error": str(e),
        }


async def process_pdfs(pdf_paths, output_dir="data/pdfs", max_concurrent=5):
    """
    Process multiple PDFs concurrently

    Args:
        pdf_paths: List of PDF file paths
        output_dir: Output directory
        max_concurrent: Maximum number of concurrent PDF processing tasks

    Returns:
        List of results
    """
    print(f"🚀 Starting PDF processing...")
    print(f"   PDFs to process: {len(pdf_paths)}")
    print(f"   Max concurrent: {max_concurrent}")
    print(f"   Output directory: {output_dir}\n")

    # Semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(pdf_path):
        async with semaphore:
            result = await pdf_to_markdown(pdf_path, output_dir)
            # Small delay between PDFs to avoid overwhelming the system
            await asyncio.sleep(0.1)
            return result

    # Process all PDFs concurrently
    tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_paths]
    results = await asyncio.gather(*tasks)

    return results


def find_pdfs_in_directory(directory):
    path = Path(directory)
    pdfs = list(path.rglob("*.pdf"))
    return [str(pdf) for pdf in pdfs]
