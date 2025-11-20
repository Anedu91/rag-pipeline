import argparse
import asyncio

from scrapers.web_scraper import discover_urls_from_selector, scrape_urls


async def run_pipeline(base_url, selector, output_folder):
    urls = await discover_urls_from_selector(
        start_url=base_url, selector=selector, headless=True
    )
    output_dir = f"data/{output_folder}"
    results = await scrape_urls(
        urls=urls, output_dir=output_dir, headless=True, max_concurrent=5
    )

    return results


async def main():
    parser = argparse.ArgumentParser(
        description="Web scraper that discovers URLs and converts them to Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
      # Three.js docs
      python main.py https://threejs.org/docs/ "#panel" threejs-docs

      # MDN WebGL
      python main.py https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API ".sidebar-inner" webgl-docs

      # With options
      python main.py https://example.com/docs "nav.docs" my-docs --no-save-urls
        """,
    )

    parser.add_argument("url", help="Base URL to start scraping")

    parser.add_argument(
        "selector",
        help='CSS selector of the element containing the links (e.g., "#panel", ".sidebar")',
    )

    parser.add_argument(
        "output", help="Output folder name (will be saved in output/NAME/)"
    )

    args = parser.parse_args()

    await run_pipeline(
        base_url=args.url,
        selector=args.selector,
        output_folder=args.output,
    )


if __name__ == "__main__":
    asyncio.run(main())
