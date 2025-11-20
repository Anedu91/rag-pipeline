import asyncio
from dis import disco
from socket import timeout

from playwright.async_api import Page, async_playwright

from scrapers.utils import extract_links_from_element, save_as_markdown


async def scrape_urls(urls, output_dir, delay=1, headless=True, max_concurrent=5):
    results = []
    async with async_playwright() as playwrith:
        browser = await playwrith.chromium.launch(headless=headless)

        context = await browser.new_context()

        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_with_semaphore(url):
            async with semaphore:
                page = await context.new_page()

                result = await scrape_url(page, url, output_dir)
                await page.close()
                if delay > 0:
                    await asyncio.sleep(delay)

                return result

        tasks = [scrape_with_semaphore(url) for url in urls]

        results = await asyncio.gather(*tasks)

        await context.close()
        await browser.close()
    return results


async def scrape_url(page: Page, url: str, output_dir: str):
    try:
        await page.goto(url, wait_until="networkidle")
        html_content = await page.content()
        await save_as_markdown(html_content, url, output_dir)

        return {"url": url, "succes": True}
    except Exception as e:
        print(f"error {e}")
        return {"url": url, "success": False}


async def discover_urls_from_selector(
    start_url: str,
    selector: str,
    max_depth=1,
    max_urls=100,
    include_query=False,
    headless=True,
):
    discovered = set()
    to_visit = [(start_url, 0)]
    visited = set()

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless)
        context = await browser.new_context()
        page = await context.new_page()

        while to_visit and len(discovered) < max_urls:
            current_url, depth = to_visit.pop(0)

            if current_url in visited:
                continue

            visited.add(current_url)
            discovered.add(current_url)

            if depth >= max_depth:
                continue
            try:
                await page.goto(current_url, wait_until="networkidle", timeout=30000)
            except Exception as e:
                print(f"Error during discover_from_selectetor {e}")

            internal_links = await extract_links_from_element(
                page, start_url, selector, include_query
            )

            for link in internal_links:
                if link not in visited and link not in [url for url, _ in to_visit]:
                    to_visit.append((link, depth + 1))
            await asyncio.sleep(0.5)

            await context.close()
            await browser.close()
        results = list(discovered)
        print(f"\n founded {len(results)} unique URLs")
        return results
