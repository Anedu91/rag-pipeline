"""
Usage:
    python md_cleaner.py <input_folder> [--output_folder <output_folder>] [--dry-run]

Example:
    python md_cleaner.py output/threejs-docs
    python md_cleaner.py output/threejs-docs --output_folder cleaned/threejs-docs
    python md_cleaner.py output/threejs-docs --dry-run
"""

import argparse
import re
from pathlib import Path
from typing import Callable, List, Tuple


# Pure functions for cleaning operations
def remove_excessive_newlines(content: str) -> str:
    """Reduce multiple consecutive newlines to maximum 2"""
    return re.sub(r"\n{3,}", "\n\n", content)


def remove_trailing_whitespace(content: str) -> str:
    """Remove trailing whitespace from each line"""
    lines = content.split("\n")
    return "\n".join(line.rstrip() for line in lines)


def remove_empty_links(content: str) -> str:
    """Remove markdown links with empty text like [](url)"""
    return re.sub(r"\[\]\([^\)]*\)", "", content)


def remove_navigation_patterns(content: str) -> str:
    """Remove common navigation patterns like 'Skip to content', 'Menu', etc."""
    patterns = [
        r"^#+\s*(Skip to (main )?content|Menu|Navigation|Table of Contents)\s*$",
        r"^\*\*?(Skip to (main )?content|Menu|Navigation|Table of Contents)\*\*?\s*$",
    ]
    lines = content.split("\n")
    for pattern in patterns:
        lines = [line for line in lines if not re.match(pattern, line, re.IGNORECASE)]
    return "\n".join(lines)


def remove_footer_patterns(content: str) -> str:
    """Remove common footer patterns"""
    patterns = [
        r"^#+\s*(Copyright|©|Footer|Contact Us|Privacy Policy|Terms of Service)\s*$",
        r"^\*\*?(Copyright|©|Footer|Contact Us|Privacy Policy|Terms of Service)\*\*?\s*$",
    ]
    lines = content.split("\n")
    for pattern in patterns:
        lines = [line for line in lines if not re.match(pattern, line, re.IGNORECASE)]
    return "\n".join(lines)


def remove_social_media_links(content: str) -> str:
    """Remove social media share buttons and links"""
    patterns = [
        r"\[?\s*(Share on|Follow us on|Connect with us)\s*(Twitter|Facebook|LinkedIn|Instagram|GitHub)\s*\]?\([^\)]*\)",
        r"\[?\s*(Twitter|Facebook|LinkedIn|Instagram)\s*\]?\([^\)]*(?:twitter|facebook|linkedin|instagram)[^\)]*\)",
    ]
    for pattern in patterns:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE)
    return content


def remove_edit_source_links(content: str) -> str:
    """Remove 'Edit this page' or 'View source' type links"""
    patterns = [
        r"\[?\s*(Edit this page|View source|Edit on GitHub|Improve this page)\s*\]?\([^\)]*\)",
    ]
    for pattern in patterns:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE)
    return content


def clean_redundant_brackets(content: str) -> str:
    """Remove redundant empty brackets and parentheses"""
    content = re.sub(r"\[\s*\]", "", content)
    content = re.sub(r"\(\s*\)", "", content)
    return content


def normalize_headings(content: str) -> str:
    """Ensure headings have proper spacing"""
    lines = content.split("\n")
    normalized = []

    for i, line in enumerate(lines):
        if re.match(r"^#+\s+", line):
            if i > 0 and normalized and normalized[-1].strip():
                normalized.append("")
            normalized.append(line)
        else:
            normalized.append(line)

    return "\n".join(normalized)


def remove_advertisement_patterns(content: str) -> str:
    """Remove common advertisement and promotional text patterns"""
    patterns = [
        r"^.*?(Advertisement|Sponsored|Promotion|Ad\s*:).*?$",
        r"^\s*\[?(Subscribe|Sign up|Newsletter|Join us)\]?.*?$",
    ]
    lines = content.split("\n")
    for pattern in patterns:
        lines = [line for line in lines if not re.match(pattern, line, re.IGNORECASE)]
    return "\n".join(lines)


def remove_cookie_consent(content: str) -> str:
    """Remove cookie consent messages"""
    patterns = [
        r".*?cookies?.*?(accept|consent|agree).*?",
        r".*?(accept|consent).*?cookies?.*?",
    ]
    lines = content.split("\n")
    for pattern in patterns:
        lines = [line for line in lines if not re.search(pattern, line, re.IGNORECASE)]
    return "\n".join(lines)


def preserve_metadata(content: str) -> Tuple[str, str]:
    """Extract and preserve frontmatter metadata"""
    metadata_pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(metadata_pattern, content, re.DOTALL)

    if match:
        metadata = match.group(0)
        body = content[len(metadata) :]
        return metadata, body

    return "", content


def split_by_pages(content: str) -> List[Tuple[str, str]]:
    """Split PDF content by page markers (## Page N)"""
    # Split by page headings
    page_pattern = r"^## Page \d+$"
    parts = re.split(f"({page_pattern})", content, flags=re.MULTILINE)

    pages = []
    current_page_header = None

    for part in parts:
        if re.match(page_pattern, part.strip()):
            current_page_header = part.strip()
        elif current_page_header and part.strip():
            pages.append((current_page_header, part))
            current_page_header = None

    return pages


def clean_pdf_specific(content: str) -> str:
    """Clean PDF-specific noise like page headers, footers, page numbers"""
    # Line-based patterns (must match entire line)
    patterns = [
        r"^\s*\d+\s*$",  # Standalone page numbers
        r"^.*?www\.packtpub\.com.*?$",  # Publisher URLs (full line)
        r"^.*?ISBN\s+[\d-]+.*?$",  # ISBN lines
        r"^.*?Copyright\s*©\s*\d{4}.*?$",  # Copyright lines with year
        r"^Credits$",  # Credits page header
        r"^About the Author$",  # Common book section headers
        r"^About the Reviewers?$",
        r"^Production reference:.*?$",  # Production info
        r"^Published by.*?$",  # Publisher info
        r"^Cover (image|illustration|work).*?$",  # Cover credits
        r"^Livery Place$",  # Publisher address patterns
        r"^\d+\s+Livery Street$",
        r"^Birmingham.*?UK\.?$",
        r"^First published:.*?$",
        r"^All rights reserved\.?$",
        r"^BIRMINGHAM\s*-\s*MUMBAI$",  # Publisher cities
        r"^www\.[a-z]+\.(com|org|net)$",  # Standalone URLs
    ]

    lines = content.split("\n")
    for pattern in patterns:
        lines = [line for line in lines if not re.match(pattern, line, re.IGNORECASE)]

    return "\n".join(lines)


def detect_source_type(file_path: Path, content: str) -> str:
    """Detect the type of markdown source"""
    # Check parent folder name
    parent = file_path.parent.name

    if parent == "pdfs":
        return "pdf"
    elif parent == "video":
        return "video"
    elif parent in ["threejs", "discover-docs"]:
        return "web_docs"

    # Fallback: check content patterns
    if "## Page" in content[:1000]:
        return "pdf"
    elif "Transcript with Timestamps" in content[:1000]:
        return "video"

    return "generic"


def compose(*functions: Callable[[str], str]) -> Callable[[str], str]:
    """Compose multiple cleaning functions into a single function"""

    def composed(content: str) -> str:
        result = content
        for func in functions:
            result = func(result)
        return result

    return composed


# Specialized cleaning pipelines for different source types
pdf_cleaning_pipeline = compose(
    clean_pdf_specific,
    remove_excessive_newlines,
    remove_trailing_whitespace,
    remove_empty_links,
    remove_footer_patterns,
    clean_redundant_brackets,
    normalize_headings,
    remove_excessive_newlines,
)

video_cleaning_pipeline = compose(
    remove_excessive_newlines,
    remove_trailing_whitespace,
    remove_empty_links,
    clean_redundant_brackets,
    normalize_headings,
    remove_excessive_newlines,
)

web_docs_cleaning_pipeline = compose(
    remove_excessive_newlines,
    remove_trailing_whitespace,
    remove_empty_links,
    remove_navigation_patterns,
    remove_footer_patterns,
    remove_social_media_links,
    remove_edit_source_links,
    remove_advertisement_patterns,
    remove_cookie_consent,
    clean_redundant_brackets,
    normalize_headings,
    remove_excessive_newlines,
)

# Generic pipeline (fallback)
generic_cleaning_pipeline = compose(
    remove_excessive_newlines,
    remove_trailing_whitespace,
    remove_empty_links,
    remove_navigation_patterns,
    remove_footer_patterns,
    remove_social_media_links,
    remove_edit_source_links,
    remove_advertisement_patterns,
    remove_cookie_consent,
    clean_redundant_brackets,
    normalize_headings,
    remove_excessive_newlines,
)


def clean_markdown_file(file_path: Path) -> Tuple[str, str]:
    """Clean a single markdown file and return the cleaned content and source type"""
    content = file_path.read_text(encoding="utf-8")
    source_type = detect_source_type(file_path, content)

    # Preserve metadata
    metadata, body = preserve_metadata(content)

    # Apply appropriate cleaning pipeline based on source type
    if source_type == "pdf":
        # Split by pages and clean each page individually
        pages = split_by_pages(body)

        if pages:
            # Clean each page separately
            cleaned_pages = []
            for page_header, page_content in pages:
                cleaned_content = pdf_cleaning_pipeline(page_content)
                # Only add non-empty pages
                if cleaned_content.strip():
                    cleaned_pages.append(f"{page_header}\n\n{cleaned_content}")

            cleaned_body = "\n\n".join(cleaned_pages)
        else:
            # No page markers found, clean as whole
            cleaned_body = pdf_cleaning_pipeline(body)

    elif source_type == "video":
        cleaned_body = video_cleaning_pipeline(body)

    elif source_type == "web_docs":
        cleaned_body = web_docs_cleaning_pipeline(body)

    else:
        cleaned_body = generic_cleaning_pipeline(body)

    # Reconstruct with metadata
    result = metadata + cleaned_body if metadata else cleaned_body
    return result, source_type


def process_folder(
    input_folder: Path, output_folder: Path = None, dry_run: bool = False
) -> List[dict]:
    """Process all markdown files in a folder"""
    if not input_folder.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")

    if not input_folder.is_dir():
        raise ValueError(f"Input path is not a directory: {input_folder}")

    # Use input folder if no output folder specified
    if output_folder is None:
        output_folder = input_folder
    else:
        output_folder.mkdir(parents=True, exist_ok=True)

    md_files = list(input_folder.glob("*.md"))

    if not md_files:
        print(f"No markdown files found in {input_folder}")
        return []

    results = []

    for md_file in md_files:
        try:
            original_size = md_file.stat().st_size
            cleaned_content, source_type = clean_markdown_file(md_file)
            cleaned_size = len(cleaned_content.encode("utf-8"))

            output_path = output_folder / md_file.name

            if dry_run:
                print(f"[DRY RUN] Would clean: {md_file.name} ({source_type})")
                print(f"  Original size: {original_size} bytes")
                print(f"  Cleaned size: {cleaned_size} bytes")
                print(
                    f"  Reduction: {original_size - cleaned_size} bytes ({((original_size - cleaned_size) / original_size * 100):.1f}%)"
                )
            else:
                output_path.write_text(cleaned_content, encoding="utf-8")
                print(f"✓ Cleaned: {md_file.name} ({source_type})")
                print(
                    f"  Original: {original_size} bytes → Cleaned: {cleaned_size} bytes"
                )

            results.append(
                {
                    "file": md_file.name,
                    "original_size": original_size,
                    "cleaned_size": cleaned_size,
                    "reduction": original_size - cleaned_size,
                    "source_type": source_type,
                    "success": True,
                }
            )

        except Exception as e:
            print(f"✗ Error processing {md_file.name}: {e}")
            results.append({"file": md_file.name, "success": False, "error": str(e)})

    return results


def print_summary(results: List[dict]):
    """Print summary statistics with source type breakdown"""
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]

    if successful:
        # Group by source type
        by_type = {}
        for r in successful:
            source_type = r.get("source_type", "unknown")
            if source_type not in by_type:
                by_type[source_type] = []
            by_type[source_type].append(r)

        print("\n" + "=" * 50)
        print("SUMMARY BY SOURCE TYPE")
        print("=" * 50)

        total_all_original = 0
        total_all_cleaned = 0

        for source_type, items in sorted(by_type.items()):
            total_orig = sum(i["original_size"] for i in items)
            total_clean = sum(i["cleaned_size"] for i in items)
            reduction = total_orig - total_clean

            total_all_original += total_orig
            total_all_cleaned += total_clean

            print(f"\n{source_type.upper()}:")
            print(f"  Files: {len(items)}")
            print(f"  Original: {total_orig:,} bytes")
            print(f"  Cleaned: {total_clean:,} bytes")
            print(
                f"  Reduction: {reduction:,} bytes ({reduction / total_orig * 100:.1f}%)"
            )

        # Overall summary
        total_reduction = total_all_original - total_all_cleaned
        print("\n" + "=" * 50)
        print("OVERALL SUMMARY")
        print("=" * 50)
        print(f"Total files processed: {len(successful)}")
        print(f"Total original size: {total_all_original:,} bytes")
        print(f"Total cleaned size: {total_all_cleaned:,} bytes")
        print(
            f"Total reduction: {total_reduction:,} bytes ({(total_reduction / total_all_original * 100):.1f}%)"
        )

    if failed:
        print(f"\nFailed: {len(failed)} files")
        for r in failed:
            print(f"  - {r['file']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean noise from generated markdown files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean files in place
  python md_cleaner.py output/threejs-docs

  # Clean to a different folder
  python md_cleaner.py output/threejs-docs --output_folder cleaned/threejs-docs

  # Preview changes without modifying files
  python md_cleaner.py output/threejs-docs --dry-run
        """,
    )

    parser.add_argument(
        "input_folder", type=str, help="Folder containing markdown files to clean"
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default=None,
        help="Output folder for cleaned files (default: same as input)",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying files"
    )

    args = parser.parse_args()

    input_path = Path(args.input_folder)
    output_path = Path(args.output_folder) if args.output_folder else None

    print(f"Processing markdown files in: {input_path}")
    if args.dry_run:
        print("[DRY RUN MODE - No files will be modified]")
    if output_path:
        print(f"Output folder: {output_path}")
    else:
        print("Cleaning files in place")
    print()

    results = process_folder(input_path, output_path, args.dry_run)
    print_summary(results)


if __name__ == "__main__":
    main()
