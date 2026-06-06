"""Tests for Wikipedia client URL parsing and link filtering."""

import pytest

from surfer.keywords import normalize_title
from surfer.wiki import PageNotFoundError, WikiClient, WikiLink, url_to_title


def test_url_to_title_simple():
    assert url_to_title("https://en.wikipedia.org/wiki/Dog") == "Dog"


def test_url_to_title_encoded():
    assert (
        url_to_title("https://en.wikipedia.org/wiki/Counter-Strike_(video_game)")
        == "Counter-Strike (video game)"
    )


def test_url_to_title_with_fragment():
    assert url_to_title("https://en.wikipedia.org/wiki/Dog#History") == "Dog"


def test_url_to_title_invalid_domain():
    with pytest.raises(ValueError, match="Not a Wikipedia URL"):
        url_to_title("https://example.com/wiki/Dog")


def test_normalize_title():
    assert normalize_title("Counter-Strike_(video_game)") == "counter-strike (video game)"


def test_filter_unvisited():
    client = WikiClient.__new__(WikiClient)
    links = [
        WikiLink(title="Dog"),
        WikiLink(title="Cat"),
        WikiLink(title="Valve Corporation"),
    ]
    visited = {"Dog", "Counter-Strike"}
    filtered = client.filter_unvisited(links, visited)
    titles = {link.title for link in filtered}
    assert "Dog" not in titles
    assert "Cat" in titles
    assert "Valve Corporation" in titles


def test_resolve_title(httpx_mock):
    httpx_mock.add_response(
        json={
            "query": {
                "pages": [{"title": "Counter-Strike (video game)", "pageid": 12345}],
            }
        }
    )
    with WikiClient() as client:
        title = client.resolve_title("Counter-Strike")
    assert title == "Counter-Strike (video game)"


def test_page_not_found(httpx_mock):
    missing_response = {
        "query": {
            "pages": [{"title": "Missing Page", "missing": True}],
        }
    }
    httpx_mock.add_response(json=missing_response)
    httpx_mock.add_response(json=missing_response)

    with WikiClient(request_delay=0) as client:
        with pytest.raises(PageNotFoundError, match="Page not found"):
            client.resolve_title("Missing Page")
        assert client.page_exists("Missing Page") is False


def test_page_exists_true(httpx_mock):
    httpx_mock.add_response(
        json={
            "query": {
                "pages": [{"title": "Dog", "pageid": 1}],
            }
        }
    )
    with WikiClient(request_delay=0) as client:
        assert client.page_exists("Dog") is True


def test_fetch_page(httpx_mock):
    httpx_mock.add_response(
        json={
            "query": {
                "pages": [
                    {
                        "title": "Valve Corporation",
                        "pageid": 999,
                        "extract": "Valve is a video game developer.",
                        "links": [
                            {"title": "Steam"},
                            {"title": "Counter-Strike"},
                        ],
                    }
                ],
            }
        }
    )

    with WikiClient(request_delay=0) as client:
        page = client.fetch("Valve Corporation")

    assert page.title == "Valve Corporation"
    assert "video game" in page.extract
    assert len(page.links) == 2


def test_fetch_retries_on_429(httpx_mock):
    httpx_mock.add_response(status_code=429, headers={"Retry-After": "0"})
    httpx_mock.add_response(
        json={
            "query": {
                "pages": [
                    {
                        "title": "Dog",
                        "pageid": 1,
                        "extract": "A domestic animal.",
                        "links": [],
                    }
                ],
            }
        }
    )

    with WikiClient(request_delay=0) as client:
        page = client.fetch("Dog")

    assert page.title == "Dog"


def test_random_page_url(httpx_mock):
    httpx_mock.add_response(
        json={
            "query": {
                "random": [{"id": 123, "title": "Random Article"}],
            }
        }
    )
    with WikiClient(request_delay=0) as client:
        url = client.random_page_url()
    assert url == "https://en.wikipedia.org/wiki/Random_Article"
