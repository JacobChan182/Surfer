"""Wikipedia API client for page fetch and link extraction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import httpx

from surfer.keywords import normalize_title

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "SurferAgent/0.1 (https://github.com/surfer; educational wiki game bot)"
MAX_RETRIES = 6
BASE_BACKOFF = 1.0


class PageNotFoundError(ValueError):
    """Raised when a Wikipedia page title does not exist."""


class WikiRateLimitError(RuntimeError):
    """Raised when Wikipedia keeps returning 429 after retries."""


@dataclass(frozen=True)
class WikiLink:
    title: str


@dataclass
class WikiPage:
    title: str
    extract: str
    links: list[WikiLink]


def url_to_title(url: str) -> str:
    """Parse a Wikipedia URL into a page title."""
    parsed = urlparse(url)
    if "wikipedia.org" not in parsed.netloc:
        raise ValueError(f"Not a Wikipedia URL: {url}")

    path = parsed.path
    prefix = "/wiki/"
    if not path.startswith(prefix):
        raise ValueError(f"URL does not contain /wiki/ path: {url}")

    slug = path[len(prefix) :]
    slug = slug.split("#")[0].split("?")[0]
    return unquote(slug.replace("_", " "))


class WikiClient:
    def __init__(
        self,
        client: httpx.Client | None = None,
        request_delay: float = 0.3,
    ) -> None:
        self._client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=30.0,
        )
        self._owns_client = client is None
        self._request_delay = request_delay

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> WikiClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _get(self, params: dict) -> dict:
        last_response: httpx.Response | None = None
        for attempt in range(MAX_RETRIES):
            if self._request_delay > 0:
                time.sleep(self._request_delay)
            response = self._client.get(WIKI_API, params={**params, "format": "json"})
            last_response = response
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else BASE_BACKOFF * (2**attempt)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()

        status = last_response.status_code if last_response else "unknown"
        raise WikiRateLimitError(
            f"Wikipedia rate limit exceeded after {MAX_RETRIES} retries (last status: {status})"
        )

    def resolve_title(self, title: str) -> str:
        """Resolve redirects to the canonical page title."""
        data = self._get(
            {
                "action": "query",
                "titles": title,
                "redirects": 1,
                "formatversion": 2,
            }
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            raise PageNotFoundError(f"Page not found: {title}")
        page = pages[0]
        if page.get("missing"):
            raise PageNotFoundError(f"Page not found: {title}")
        return page["title"]

    def page_exists(self, title: str) -> bool:
        """Return True if the page exists (including via redirect)."""
        try:
            self.resolve_title(title)
            return True
        except PageNotFoundError:
            return False

    def fetch(self, title: str) -> WikiPage:
        """Fetch page extract and outbound namespace-0 links in minimal API calls."""
        data = self._get(
            {
                "action": "query",
                "titles": title,
                "redirects": 1,
                "prop": "extracts|links",
                "explaintext": 1,
                "exintro": 0,
                "plnamespace": 0,
                "pllimit": "max",
                "formatversion": 2,
            }
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing"):
            raise PageNotFoundError(f"Page not found: {title}")

        page = pages[0]
        canonical = page["title"]
        extract = page.get("extract", "")
        links = [WikiLink(title=link["title"]) for link in page.get("links", [])]

        continue_token = data.get("continue", {}).get("plcontinue")
        if continue_token:
            links.extend(self._fetch_more_links(canonical, continue_token))

        return WikiPage(title=canonical, extract=extract, links=links)

    def _fetch_more_links(self, title: str, continue_token: str) -> list[WikiLink]:
        """Paginate remaining outbound links."""
        result: list[WikiLink] = []
        token: str | None = continue_token

        while token:
            data = self._get(
                {
                    "action": "query",
                    "titles": title,
                    "prop": "links",
                    "plnamespace": 0,
                    "pllimit": "max",
                    "plcontinue": token,
                    "formatversion": 2,
                }
            )
            pages = data.get("query", {}).get("pages", [])
            if pages:
                for link in pages[0].get("links", []):
                    result.append(WikiLink(title=link["title"]))

            token = data.get("continue", {}).get("plcontinue")

        return result

    def filter_unvisited(self, links: list[WikiLink], visited: set[str]) -> list[WikiLink]:
        """Return links whose normalized title is not in visited."""
        visited_norm = {normalize_title(v) for v in visited}
        return [link for link in links if normalize_title(link.title) not in visited_norm]

    def random_page_url(self, namespace: int = 0) -> str:
        """Return a random English Wikipedia article URL (main namespace by default)."""
        data = self._get(
            {
                "action": "query",
                "list": "random",
                "rnnamespace": namespace,
                "rnlimit": 1,
                "formatversion": 2,
            }
        )
        pages = data.get("query", {}).get("random", [])
        if not pages:
            raise RuntimeError("Wikipedia API returned no random pages")
        title = pages[0]["title"]
        slug = title.replace(" ", "_")
        return f"https://en.wikipedia.org/wiki/{slug}"
