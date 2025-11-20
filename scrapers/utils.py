import hashlib
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import html2text
from bs4 import BeautifulSoup
from playwright.async_api import Page


def generate_filename(url, extension=".md"):
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    parsed = urlparse(url)
    domain = parsed.netloc.replace(".", "_")
    return f"{domain}_{url_hash}{extension}"


def html_to_markdown(html_content):
    h = html2text.HTML2Text()
    h.ignore_links = False  # maintain links
    h.ignore_images = False  # maintain images
    h.ignore_emphasis = False  # maintain bold,etc
    h.body_width = 0  # No wrap lines
    h.skip_internal_links = False

    markdown = h.handle(html_content)

    return markdown


def clean_html(html_content):
    """Limpia HTML removiendo scripts, styles, etc."""
    soup = BeautifulSoup(html_content, "html.parser")

    # remove noisy elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    return str(soup)


async def save_as_markdown(html_content, url, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cleaned_html = clean_html(html_content)

    markdown_content = html_to_markdown(cleaned_html)

    metadata = f"""---
                url: {url}
                scraped_at: {time.strftime("%Y-%m-%d %H:%M:%S")}
                ---

                """

    full_content = metadata + markdown_content

    filename = generate_filename(url, extension=".md")
    filepath = output_path / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_content)

    return filepath


def is_same_domain(url: str, base_url: str):
    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)
    return parsed_url.netloc == parsed_base.netloc


def normalize_url(url):
    parsed = urlparse(url)

    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    if normalized.endswith("/") and len(parsed.path) > 1:
        normalized = normalized[:-1]
    return normalized


async def extract_links_from_element(
    page: Page, base_url: str, selector: str, include_query=False
):
    try:
        element = await page.query_selector(selector)
    except Exception as e:
        print(f"error in extract_links {e}")
        return []

    if not element:
        return []

    links = await element.eval_on_selector_all(
        "a[href]", "(elements) => elements.map(el => el.href)"
    )

    internal_links = set()

    for link in links:
        absolute_url = urljoin(base_url, link)

        if is_same_domain(absolute_url, base_url):
            if include_query:
                internal_links.add(absolute_url)
            else:
                normalized = normalize_url(absolute_url)
                internal_links.add(normalized)
    return list(internal_links)
