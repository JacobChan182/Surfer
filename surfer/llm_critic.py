"""LLM critic and link picker for Wikipedia navigation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from surfer.keywords import link_features, normalize_title
from surfer.nvidia_llm import NvidiaLLM
from surfer.wiki import WikiLink, WikiPage

SelectionMode = Literal["bandit", "llm"]

CRITIC_SYSTEM_PROMPT = """You are a critic for a Wikipedia navigation agent trying to reach the \
Counter-Strike article (Counter-Strike or Counter-Strike (video game)).

Evaluate how close the current page is to that goal. Respond ONLY with valid JSON:
{
  "progress_score": <float 0.0-1.0>,
  "reasoning": "<one short sentence>",
  "weight_updates": {"<word>": <delta float -0.5 to 0.5>, ...}
}

weight_updates: tokens to boost (+) or penalize (-) in future link titles.
Favor gaming terms (valve, steam, fps, shooter, esports). Penalize off-topic \
terms (politician, botany, election). Keep 3-8 weight_updates max."""

PICKER_SYSTEM_PROMPT = """You are a Wikipedia navigation agent trying to reach the \
Counter-Strike article (Counter-Strike or Counter-Strike (video game)).

You will receive the current page and a numbered list of outbound links. Pick the \
link most likely to lead toward Counter-Strike (Valve, FPS games, esports, etc.).

Respond ONLY with valid JSON:
{
  "progress_score": <float 0.0-1.0 how close the CURRENT page is>,
  "chosen_link": "<exact title copied from the list>",
  "reasoning": "<one short sentence why this link>",
  "weight_updates": {"<word>": <delta float -0.5 to 0.5>, ...}
}

chosen_link MUST match one list entry exactly. weight_updates: boost (+) or penalize (-) \
keywords from the chosen link title and path. Keep 3-8 weight_updates max."""


@dataclass(frozen=True)
class CriticResult:
    progress_score: float
    reasoning: str
    weight_updates: dict[str, float]
    chosen_link: str | None = None


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_weight_updates(raw: object) -> dict[str, float]:
    weight_updates: dict[str, float] = {}
    if isinstance(raw, dict):
        for word, delta in raw.items():
            word_str = str(word).lower().strip()
            if word_str:
                weight_updates[word_str] = max(-0.5, min(0.5, float(delta)))
    return weight_updates


def _parse_json_response(text: str) -> dict:
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    return json.loads(cleaned)


def _parse_critic_response(text: str) -> CriticResult:
    """Parse LLM JSON response for critic-only mode."""
    data = _parse_json_response(text)
    return CriticResult(
        progress_score=_clamp_score(float(data.get("progress_score", 0.0))),
        reasoning=str(data.get("reasoning", "")),
        weight_updates=_parse_weight_updates(data.get("weight_updates", {})),
    )


def _parse_picker_response(text: str) -> CriticResult:
    """Parse LLM JSON response for link-picking mode."""
    data = _parse_json_response(text)
    chosen = data.get("chosen_link")
    chosen_str = str(chosen).strip() if chosen is not None else None
    if chosen_str == "":
        chosen_str = None
    return CriticResult(
        progress_score=_clamp_score(float(data.get("progress_score", 0.0))),
        reasoning=str(data.get("reasoning", "")),
        weight_updates=_parse_weight_updates(data.get("weight_updates", {})),
        chosen_link=chosen_str,
    )


def match_chosen_link(
    chosen_title: str | None,
    candidates: list[WikiLink],
    chosen_index: int | None = None,
) -> WikiLink | None:
    """Match LLM output to a candidate link."""
    if chosen_index is not None and 0 <= chosen_index < len(candidates):
        return candidates[chosen_index]

    if not chosen_title:
        return None

    target = normalize_title(chosen_title)
    for link in candidates:
        if normalize_title(link.title) == target:
            return link

    # Substring fallback for minor LLM formatting differences
    for link in candidates:
        norm = normalize_title(link.title)
        if target in norm or norm in target:
            return link

    return None


def blend_rewards(keyword_reward: float, llm_score: float, llm_blend: float) -> float:
    """Blend keyword and LLM progress scores."""
    return (1.0 - llm_blend) * keyword_reward + llm_blend * llm_score


class LLMCritic:
    def __init__(
        self,
        llm: NvidiaLLM,
        llm_blend: float = 0.6,
        delta_scale: float = 0.5,
        candidate_limit: int = 0,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.llm_blend = llm_blend
        self.delta_scale = delta_scale
        self.candidate_limit = candidate_limit
        self.verbose = verbose

    def evaluate(
        self,
        page: WikiPage,
        path: list[str],
        last_link: str | None,
        keyword_reward: float,
        matched_keywords: list[str],
        top_weights: list[tuple[str, float]],
    ) -> CriticResult | None:
        """Call the LLM critic (progress + weight updates only). Returns None on failure."""
        extract_snippet = page.extract[:500]
        path_tail = " -> ".join(path[-8:]) if path else page.title
        weights_str = ", ".join(f"{w}={v:.2f}" for w, v in top_weights[:10]) or "none"
        keywords_str = ", ".join(matched_keywords) or "none"
        arrived_via = last_link or "(start page)"

        user_prompt = f"""Goal: Counter-Strike Wikipedia article

Current page: {page.title}
Extract: {extract_snippet}

Path (recent): {path_tail}
Arrived via link: {arrived_via}
Keyword reward: {keyword_reward:.2f} (matched: {keywords_str})
Top learned weights: {weights_str}

How close are we? What link-title keywords should be boosted or penalized?"""

        return self._call_llm(CRITIC_SYSTEM_PROMPT, user_prompt, picker=False)

    def pick_link(
        self,
        page: WikiPage,
        path: list[str],
        last_link: str | None,
        keyword_reward: float,
        matched_keywords: list[str],
        top_weights: list[tuple[str, float]],
        candidates: list[WikiLink],
    ) -> tuple[WikiLink | None, CriticResult | None]:
        """Ask the LLM to pick the best link from candidates. Returns (link, result)."""
        if not candidates:
            return None, None

        extract_snippet = page.extract[:500]
        path_tail = " -> ".join(path[-8:]) if path else page.title
        weights_str = ", ".join(f"{w}={v:.2f}" for w, v in top_weights[:10]) or "none"
        keywords_str = ", ".join(matched_keywords) or "none"
        arrived_via = last_link or "(start page)"

        link_lines = "\n".join(f"{i + 1}. {link.title}" for i, link in enumerate(candidates))
        total_note = f"\n({len(candidates)} links — pick from this list only.)"

        user_prompt = f"""Goal: Counter-Strike Wikipedia article

Current page: {page.title}
Extract: {extract_snippet}

Path (recent): {path_tail}
Arrived via link: {arrived_via}
Keyword reward: {keyword_reward:.2f} (matched: {keywords_str})
Top learned weights: {weights_str}

Outbound links to choose from:{total_note}
{link_lines}

Pick the one link most likely to lead toward Counter-Strike."""

        result = self._call_llm(PICKER_SYSTEM_PROMPT, user_prompt, picker=True)
        if result is None:
            return None, None

        matched = match_chosen_link(result.chosen_link, candidates)
        if matched is None and self.verbose:
            print(f"  [llm] could not match chosen link: {result.chosen_link!r}")
        return matched, result

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        picker: bool,
    ) -> CriticResult | None:
        try:
            raw = self.llm.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            result = _parse_picker_response(raw) if picker else _parse_critic_response(raw)
            if self.verbose:
                if picker and result.chosen_link:
                    print(
                        f"  [llm] progress={result.progress_score:.2f} | "
                        f"picked: {result.chosen_link} | {result.reasoning}"
                    )
                else:
                    print(
                        f"  [llm] progress={result.progress_score:.2f} | {result.reasoning}"
                    )
            return result
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            if self.verbose:
                print(f"  [llm] parse failed: {exc}")
            return None
        except Exception as exc:
            if self.verbose:
                print(f"  [llm] request failed: {exc}")
            return None
