Web Scraper to Markdown
A fast async web scraper that discovers URLs from HTML elements and converts pages to clean Markdown.

Installation
Install uv (if you don't have it)
`curl -LsSf https://astral.sh/uv/install.sh | sh`

# Install dependencies
`uv sync`

# Install Chromium browser
uv run playwright install chromium
Usage
bashuv run main.py <URL> <CSS_SELECTOR> <OUTPUT_FOLDER>
Example:
bashuv run main.py https://threejs.org/docs/ "#panel" threejs-docs
```

This will:
1. Visit `https://threejs.org/docs/`
2. Find all links inside the `#panel` element
3. Scrape those pages and convert to Markdown
4. Save everything to `output/threejs-docs/`

## Finding Your Downloads

Your scraped files will be in:
```
output/<OUTPUT_FOLDER>/
For example:

Command: uv run main.py https://threejs.org/docs/ "#panel" threejs-docs
Files saved in: output/threejs-docs/
URL list saved in: config/threejs-docs_urls.txt



## Each file is a Markdown document with metadata:
```
markdown---
url: https://threejs.org/docs/#manual/en/introduction/Creating-a-scene
scraped_at: 2024-01-15 10:30:45
---

# Creating a Scene

Your content here in clean Markdown...
```
