import argparse
import asyncio

from scrapers.video_scrapper import process_videos


def print_summary(results):
    """Print summary of results"""
    total = len(results)
    successful = sum(1 for r in results if r["success"])

    print(f"\n✅ Completed: {successful}/{total} videos processed")

    # Count by type
    manual = sum(1 for r in results if r.get("type") == "manual")
    auto = sum(1 for r in results if r.get("type") == "auto-generated")

    if manual > 0:
        print(f"   📝 Manual transcripts: {manual}")
    if auto > 0:
        print(f"   🤖 Auto-generated: {auto}")

    if successful < total:
        print("\n❌ Failed:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['video_url']}: {r.get('error', 'Unknown error')}")


def load_urls(filepath):
    """Load URLs from a text file"""
    with open(filepath, "r") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return urls


async def run_pipeline(video_urls, output_dir, languages, max_concurrent):
    """Run video processing pipeline"""
    results = await process_videos(
        video_urls=video_urls,
        output_dir=output_dir,
        languages=languages,
        max_concurrent=max_concurrent,
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="YouTube video transcript extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video
  python video_main.py https://www.youtube.com/watch?v=VIDEO_ID threejs-videos

  # Multiple videos
  python video_main.py URL1 URL2 URL3 threejs-videos

  # From file with URLs
  python video_main.py --file videos.txt threejs-videos

  # Specify languages
  python video_main.py URL threejs-videos --languages en es fr
        """,
    )

    parser.add_argument("inputs", nargs="*", help="YouTube video URL(s) to process")

    parser.add_argument(
        "output", help="Output folder name (will be saved in output/NAME/)"
    )

    parser.add_argument(
        "--file", "-f", help="Text file with YouTube URLs (one per line)"
    )

    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en", "es"],
        help="Preferred transcript languages in order (default: en es)",
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent video processing (default: 3)",
    )

    args = parser.parse_args()

    # Determine which videos to process
    video_urls = []

    if args.file:
        print(f"📋 Loading URLs from: {args.file}")
        video_urls = load_urls(args.file)
        print(f"   Found: {len(video_urls)} URLs\n")
    elif args.inputs:
        video_urls = args.inputs
    else:
        parser.error("You must provide video URLs or use --file")

    if not video_urls:
        print("⚠️  No videos found to process")
        return

    # Show configuration
    print(f"🎯 Configuration:")
    print(f"   Videos to process: {len(video_urls)}")
    print(f"   Languages: {', '.join(args.languages)}")
    print(f"   Max concurrent: {args.max_concurrent}")
    print(f"   Output: output/{args.output}/\n")

    # Run async pipeline
    output_dir = f"data/{args.output}"
    results = asyncio.run(
        run_pipeline(video_urls, output_dir, args.languages, args.max_concurrent)
    )

    # Summary
    print_summary(results)
    print(f"📁 Files saved in: {output_dir}/")


if __name__ == "__main__":
    main()
