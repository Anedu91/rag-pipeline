import argparse
import asyncio

from scrapers.repo_scrapper import load_urls_async, process_repositories


def print_summary(results):
    """Print summary of results"""
    total = len(results)
    successful = sum(1 for r in results if r["success"])
    total_files = sum(r.get("files_extracted", 0) for r in results if r["success"])

    print(f"\n✅ Completed: {successful}/{total} repositories processed")
    print(f"📄 Total files extracted: {total_files}")

    if successful < total:
        print("\n❌ Failed:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['repo_url']}: {r.get('error', 'Unknown error')}")


async def main_async(args):
    """Main async function"""
    # Determine which repos to process
    repo_urls = []

    if args.file:
        print(f"📋 Loading repository URLs from: {args.file}")
        repo_urls = await load_urls_async(args.file)
        print(f"   Found: {len(repo_urls)} repositories\n")
    elif args.repos:
        repo_urls = args.repos
    else:
        print("❌ You must provide repository URLs or use --file")
        return

    if not repo_urls:
        print("⚠️  No repositories found to process")
        return

    # Parse folders if provided
    folders = args.folders.split(",") if args.folders else None

    # Parse file patterns if provided
    patterns = args.patterns.split(",") if args.patterns else None

    # Parse exclude patterns if provided
    exclude = args.exclude.split(",") if args.exclude else None

    # Show configuration
    print(f"🎯 Configuration:")
    print(f"   Repositories: {len(repo_urls)}")
    print(f"   Folders to extract: {folders or 'all'}")
    print(f"   File patterns: {patterns or 'default'}")
    print(f"   Exclude patterns: {exclude or 'default'}")
    print(f"   Max concurrent: {args.max_concurrent}")
    print(f"   Keep clones: {args.keep_clones}")
    print(f"   Output: output/{args.output}/\n")

    # Run async pipeline
    output_dir = f"data/{args.output}"
    results = await process_repositories(
        repo_urls,
        folders_to_extract=folders,
        file_patterns=patterns,
        exclude_patterns=exclude,
        output_dir=output_dir,
        keep_clones=args.keep_clones,
        max_concurrent=args.max_concurrent,
    )

    # Summary
    print_summary(results)
    print(f"📁 Files saved in: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Git repository extractor - clone, extract files, cleanup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract examples and docs from Three.js
  python repo_main.py https://github.com/mrdoob/three.js.git threejs-repo --folders examples,docs

  # Extract only JavaScript files from specific folders
  python repo_main.py REPO_URL output --folders src,lib --patterns "*.js,*.ts"

  # Multiple repositories from file
  python repo_main.py --file repos.txt graphics-repos --folders examples

  # Keep cloned repositories (don't delete after extraction)
  python repo_main.py REPO_URL output --keep-clones

  # Extract everything except tests and node_modules
  python repo_main.py REPO_URL output --exclude "test,node_modules,dist"
        """,
    )

    parser.add_argument("repos", nargs="*", help="Git repository URL(s) to process")

    parser.add_argument(
        "output", help="Output folder name (will be saved in output/NAME/)"
    )

    parser.add_argument(
        "--file", "-f", help="Text file with repository URLs (one per line)"
    )

    parser.add_argument(
        "--folders",
        help='Comma-separated list of folders to extract (e.g., "examples,docs,src")',
    )

    parser.add_argument(
        "--patterns",
        help="Comma-separated file patterns to match (default: *.js,*.ts,*.html,*.md,*.json)",
    )

    parser.add_argument(
        "--exclude",
        help="Comma-separated patterns to exclude (default: node_modules,.git,test,dist)",
    )

    parser.add_argument(
        "--keep-clones",
        action="store_true",
        help="Keep cloned repositories (don't delete after extraction)",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Maximum concurrent repository processing (default: 2)",
    )

    args = parser.parse_args()

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
