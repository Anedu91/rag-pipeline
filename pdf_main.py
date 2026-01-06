import argparse
import asyncio
from pathlib import Path

from scrapers.pdf_scrapper import find_pdfs_in_directory, process_pdfs


def print_summary(results):
    """Print summary of results"""
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    total_pages = sum(r.get("pages", 0) for r in results if r["success"])

    print(f"\n✅ Completed: {successful}/{total} PDFs processed")
    print(f"📄 Total pages: {total_pages}")

    if successful < total:
        print("\n❌ Failed:")
        for r in results:
            if not r["success"]:
                print(
                    f"  - {Path(r['pdf_path']).name}: {r.get('error', 'Unknown error')}"
                )


async def run_pipeline(pdf_paths, output_dir, max_concurrent):
    """
    Run PDF processing pipeline

    Args:
        pdf_paths: List of PDF paths to process
        output_dir: Output directory
        max_concurrent: Max concurrent PDFs
    """
    results = await process_pdfs(
        pdf_paths=pdf_paths, output_dir=output_dir, max_concurrent=max_concurrent
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="PDF to Markdown converter (async)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single PDF
  python pdf_main.py /path/to/file.pdf output-folder

  # Multiple PDFs
  python pdf_main.py /path/to/file1.pdf /path/to/file2.pdf output-folder

  # All PDFs in a directory
  python pdf_main.py --directory /path/to/pdfs/ output-folder

  # With concurrency control
  python pdf_main.py --directory /path/to/pdfs/ output-folder --max-concurrent 10
        """,
    )

    parser.add_argument("inputs", nargs="*", help="PDF file(s) to process")

    parser.add_argument(
        "output", help="Output folder name (will be saved in output/NAME/)"
    )

    parser.add_argument("--directory", "-d", help="Process all PDFs in this directory")

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of concurrent PDF processing tasks (default: 5)",
    )

    args = parser.parse_args()

    # Determine which PDFs to process
    pdf_paths = []

    if args.directory:
        print(f"🔍 Searching for PDFs in: {args.directory}")
        pdf_paths = find_pdfs_in_directory(args.directory)
        print(f"   Found: {len(pdf_paths)} PDFs\n")
    elif args.inputs:
        pdf_paths = args.inputs
    else:
        parser.error("You must provide PDF files or use --directory")

    if not pdf_paths:
        print("⚠️  No PDFs found to process")
        return

    # Validate files exist
    valid_pdfs = []
    for pdf_path in pdf_paths:
        if Path(pdf_path).exists():
            valid_pdfs.append(pdf_path)
        else:
            print(f"⚠️  Not found: {pdf_path}")

    if not valid_pdfs:
        print("❌ No valid PDFs to process")
        return

    # Show configuration
    print(f"🎯 Configuration:")
    print(f"   PDFs to process: {len(valid_pdfs)}")
    print(f"   Max concurrent: {args.max_concurrent}")
    print(f"   Output: output/{args.output}/\n")

    # Run async pipeline
    output_dir = f"data/{args.output}"
    results = asyncio.run(run_pipeline(valid_pdfs, output_dir, args.max_concurrent))

    # Summary
    print_summary(results)
    print(f"📁 Files saved in: {output_dir}/")


if __name__ == "__main__":
    main()
