"""CS keyword scoring and terminal detection."""

from __future__ import annotations

from dataclasses import dataclass

from surfer.keywords import ALL_KEYWORDS, is_terminal_title


@dataclass(frozen=True)
class RewardResult:
    reward: float
    matched_keywords: list[str]
    is_terminal: bool


def score_page(title: str, extract: str, max_chars: int = 2000) -> RewardResult:
    """Score a page for CS-related content and check terminal condition."""
    if is_terminal_title(title):
        return RewardResult(reward=1.0, matched_keywords=["counter-strike"], is_terminal=True)

    text = f"{title} {extract[:max_chars]}".lower()
    matched: list[str] = []
    total = 0.0

    # Match longest keywords first to avoid partial overlaps
    for keyword, points in sorted(ALL_KEYWORDS.items(), key=lambda kv: -len(kv[0])):
        if keyword in text:
            matched.append(keyword)
            total += points

    return RewardResult(
        reward=min(total, 1.0),
        matched_keywords=matched,
        is_terminal=False,
    )
