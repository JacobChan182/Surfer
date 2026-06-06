"""CS keyword tiers and tokenization helpers."""

from __future__ import annotations

import re
from urllib.parse import unquote

# Tier -> (keywords, points per match)
STRONG_KEYWORDS: dict[str, float] = {
    "counter-strike: global offensive": 0.6,
    "counter-strike": 0.6,
    "counter strike": 0.6,
    "cs2": 0.8,
    "csgo": 0.6,
    "cs:go": 0.6,
    "cs go": 0.6,
}

MEDIUM_KEYWORDS: dict[str, float] = {
    "valve": 0.2,
    "source engine": 0.2,
    "tactical shooter": 0.15,
    "esports": 0.2,
    "steam": 0.3,
    "first-person shooter": 0.2,
    "goldsrc": 0.15,
    "source 2": 0.2,
}

WEAK_KEYWORDS: dict[str, float] = {
    "fps": 0.05,
    "multiplayer": 0.05,
    "bomb": 0.02,
    "terrorist": 0.02,
    "defusal": 0.02,
    "shooter": 0.05,
    "video game": 0.05,
}

ALL_KEYWORDS: dict[str, float] = {
    **STRONG_KEYWORDS,
    **MEDIUM_KEYWORDS,
    **WEAK_KEYWORDS,
}

# Small negative priors for generic hub/disambiguation links
NEGATIVE_PRIORS: dict[str, float] = {
    "disambiguation": -0.3,
    "category": -0.2,
    "list": -0.1,
    "index": -0.1,
    "portal": -0.1,
}

TERMINAL_TITLES = frozenset(
    {
        "counter-strike",
        "counter-strike (video game)",
    }
)

_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_title(title: str) -> str:
    """Normalize a Wikipedia page title for comparison."""
    return unquote(title.replace("_", " ")).strip().lower()


def is_terminal_title(title: str) -> bool:
    return normalize_title(title) in TERMINAL_TITLES


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return _WORD_RE.findall(text.lower())


def slug_to_words(slug: str) -> list[str]:
    """Convert a wiki URL slug into word tokens."""
    decoded = unquote(slug.replace("_", " "))
    return tokenize(decoded)


def link_features(title: str) -> frozenset[str]:
    """Extract feature tokens from a link title (anchor + slug words)."""
    words = set(tokenize(title))
    words.update(slug_to_words(title))
    return frozenset(words)
