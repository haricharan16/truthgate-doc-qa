"""
Airflow documentation scraper.
Scrapes https://airflow.apache.org/docs/apache-airflow/stable/

Strategy:
- Start from the main docs index
- Follow all internal doc links
- Extract text preserving h2/h3 structure (needed for section-aware chunking)
- Cache raw HTML to disk (avoid re-scraping on restart)
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Generator
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://airflow.apache.org/docs/apache-airflow/stable/"
CACHE_DIR = Path(os.getenv("CORPUS_CACHE_DIR", "./data/corpus"))

# Pages to skip (changelogs, API reference JSON, etc.)
SKIP_PATTERNS = [
    "_modules/", "_sources/", "genindex", "search.html",
    "changelog", "release-notes", ".json", ".xml", ".js"
]


class AirflowDocsScraper:
    def __init__(self, base_url: str = BASE_URL, cache_dir: Path = CACHE_DIR):
        self.base_url = base_url
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "TruthGate-RAG-Indexer/1.0 (research)"
        self.visited: set[str] = set()

    def _cache_path(self, url: str) -> Path:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.json"

    def _fetch(self, url: str) -> dict | None:
        cache = self._cache_path(url)
        if cache.exists():
            return json.loads(cache.read_text())

        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            data = {"url": url, "html": resp.text, "status": resp.status_code}
            cache.write_text(json.dumps(data))
            time.sleep(0.3)  # polite delay
            return data
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None

    def _should_skip(self, url: str) -> bool:
        return any(p in url for p in SKIP_PATTERNS)

    def _is_airflow_doc(self, url: str) -> bool:
        parsed = urlparse(url)
        return (
            "airflow.apache.org" in parsed.netloc
            and "/docs/apache-airflow/stable/" in parsed.path
        )

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(base_url, href).split("#")[0]  # strip anchors
            if self._is_airflow_doc(full_url) and not self._should_skip(full_url):
                links.append(full_url)
        return list(set(links))

    def extract_sections(self, html: str, url: str) -> list[dict]:
        """
        Extract structured sections from a doc page.
        Returns list of {title, content, url, section_anchor}.
        
        Edge cases handled:
        - Pages with no h2 (treated as single section)
        - Code blocks (preserved as-is, not stripped)
        - Tables (converted to text representation)
        - Admonition boxes (note/warning) — extracted with their text
        """
        soup = BeautifulSoup(html, "lxml")

        # Remove navigation, footer, sidebar
        for tag in soup.select("nav, footer, .sidebar, .sphinxsidebar, #searchbox, .headerlink"):
            tag.decompose()

        page_title_tag = soup.find("h1")
        page_title = page_title_tag.get_text(strip=True) if page_title_tag else "Untitled"

        # Find all h2 sections
        h2_tags = soup.find_all("h2")

        if not h2_tags:
            # Single-section page
            body = soup.find("div", class_=lambda c: c and ("body" in c or "content" in c))
            text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)
            return [{
                "title": page_title,
                "content": text[:4000],  # cap single sections
                "url": url,
                "section_anchor": "",
                "page_title": page_title,
            }]

        sections = []
        for i, h2 in enumerate(h2_tags):
            section_title = h2.get_text(strip=True)
            anchor = h2.get("id", "")

            # Collect all sibling elements until next h2
            content_parts = []
            for sibling in h2.next_siblings:
                if sibling.name == "h2":
                    break
                if hasattr(sibling, "get_text"):
                    # Handle code blocks specially
                    if sibling.name in ("pre", "code") or sibling.find("pre"):
                        content_parts.append(sibling.get_text(separator="\n"))
                    else:
                        content_parts.append(sibling.get_text(separator=" ", strip=True))

            content = "\n".join(p for p in content_parts if p.strip())

            if len(content.strip()) < 50:
                continue  # skip empty/stub sections

            sections.append({
                "title": f"{page_title} > {section_title}",
                "content": content,
                "url": url,
                "section_anchor": anchor,
                "page_title": page_title,
            })

        return sections

    def scrape(self, max_pages: int = 300) -> Generator[dict, None, None]:
        """Crawl Airflow docs and yield sections."""
        queue = [self.base_url]
        page_count = 0

        while queue and page_count < max_pages:
            url = queue.pop(0)
            if url in self.visited:
                continue
            self.visited.add(url)

            logger.info(f"Scraping [{page_count+1}/{max_pages}]: {url}")
            data = self._fetch(url)
            if not data:
                continue

            page_count += 1
            new_links = self._extract_links(data["html"], url)
            for link in new_links:
                if link not in self.visited:
                    queue.append(link)

            sections = self.extract_sections(data["html"], url)
            for section in sections:
                yield section

        logger.info(f"Scraping complete. {page_count} pages, {len(self.visited)} URLs visited.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scraper = AirflowDocsScraper()
    count = 0
    for section in scraper.scrape(max_pages=5):  # quick test
        print(f"Section: {section['title'][:60]} ({len(section['content'])} chars)")
        count += 1
    print(f"Total sections: {count}")
