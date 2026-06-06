"""Keyword-weight contextual bandit for link selection."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from surfer.keywords import NEGATIVE_PRIORS, link_features
from surfer.wiki import WikiLink

WEIGHT_MIN = -5.0
WEIGHT_MAX = 10.0


@dataclass
class BanditPolicy:
    epsilon: float = 0.15
    learning_rate: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05
    weights: dict[str, float] = field(default_factory=dict)
    step: int = 0

    def __post_init__(self) -> None:
        for word, prior in NEGATIVE_PRIORS.items():
            self.weights.setdefault(word, prior)

    def _clamp(self, value: float) -> float:
        return max(WEIGHT_MIN, min(WEIGHT_MAX, value))

    def _score_features(self, features: frozenset[str]) -> float:
        return sum(self.weights.get(w, 0.0) for w in features)

    def score_link(self, link: WikiLink) -> tuple[float, frozenset[str]]:
        features = link_features(link.title)
        base = self._score_features(features)
        noise = random.random() if random.random() < self.current_epsilon else 0.0
        return base + noise, features

    @property
    def current_epsilon(self) -> float:
        decayed = self.epsilon * (self.epsilon_decay**self.step)
        return max(decayed, self.min_epsilon)

    def select(self, candidates: list[WikiLink]) -> tuple[WikiLink, frozenset[str]]:
        """Pick the highest-scoring unvisited link."""
        if not candidates:
            raise ValueError("No candidates to select from")

        best_link: WikiLink | None = None
        best_score = float("-inf")
        best_features: frozenset[str] = frozenset()

        for link in candidates:
            score, features = self.score_link(link)
            jitter = random.uniform(0, 1e-6)
            if score + jitter > best_score:
                best_score = score + jitter
                best_link = link
                best_features = features

        assert best_link is not None
        return best_link, best_features

    def update(self, features: frozenset[str], reward: float) -> None:
        """Online weight update from observed reward."""
        if not features:
            return
        for word in features:
            self.weights[word] = self._clamp(
                self.weights.get(word, 0.0) + self.learning_rate * reward
            )
        self.step += 1

    def apply_deltas(self, deltas: dict[str, float], scale: float = 0.5) -> None:
        """Apply explicit keyword weight deltas from LLM distillation."""
        for word, delta in deltas.items():
            word_str = word.lower().strip()
            if word_str:
                self.weights[word_str] = self._clamp(
                    self.weights.get(word_str, 0.0) + delta * scale
                )

    def top_weights(self, n: int = 5) -> list[tuple[str, float]]:
        """Return the top-n positive keyword weights for debugging."""
        ranked = sorted(self.weights.items(), key=lambda kv: kv[1], reverse=True)
        return [(w, v) for w, v in ranked if v > 0][:n]

    def rank_candidates(self, candidates: list[WikiLink], limit: int) -> list[WikiLink]:
        """Return candidates ranked by weight score. limit<=0 sends all links."""
        ranked = sorted(
            candidates,
            key=lambda link: self._score_features(link_features(link.title)),
            reverse=True,
        )
        if limit <= 0:
            return ranked
        return ranked[:limit]
