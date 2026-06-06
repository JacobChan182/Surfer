"""Persistent storage for distilled bandit keyword weights."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_WEIGHTS_PATH = Path(".surfer/weights.json")
WEIGHTS_VERSION = 1


@dataclass
class WeightsStats:
    runs: int = 0
    successes: int = 0
    last_success_steps: int | None = None


@dataclass
class WeightsStore:
    weights: dict[str, float] = field(default_factory=dict)
    stats: WeightsStats = field(default_factory=WeightsStats)

    def to_dict(self) -> dict:
        return {
            "version": WEIGHTS_VERSION,
            "weights": self.weights,
            "stats": {
                "runs": self.stats.runs,
                "successes": self.stats.successes,
                "last_success_steps": self.stats.last_success_steps,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> WeightsStore:
        stats_data = data.get("stats", {})
        return cls(
            weights={k: float(v) for k, v in data.get("weights", {}).items()},
            stats=WeightsStats(
                runs=int(stats_data.get("runs", 0)),
                successes=int(stats_data.get("successes", 0)),
                last_success_steps=stats_data.get("last_success_steps"),
            ),
        )


def load_weights(path: Path | str = DEFAULT_WEIGHTS_PATH) -> WeightsStore | None:
    """Load weights from disk. Returns None if the file does not exist."""
    file_path = Path(path)
    if not file_path.exists():
        return None
    with file_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return WeightsStore.from_dict(data)


def save_weights(
    weights: dict[str, float],
    path: Path | str = DEFAULT_WEIGHTS_PATH,
    *,
    success: bool = False,
    steps: int = 0,
    existing: WeightsStore | None = None,
) -> WeightsStore:
    """Save weights and update run statistics."""
    store = existing or WeightsStore()
    store.weights = dict(weights)
    store.stats.runs += 1
    if success:
        store.stats.successes += 1
        store.stats.last_success_steps = steps

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(store.to_dict(), f, indent=2)

    return store


def reset_weights(path: Path | str = DEFAULT_WEIGHTS_PATH) -> None:
    """Delete the saved weights file if it exists."""
    file_path = Path(path)
    if file_path.exists():
        file_path.unlink()


def get_top_weights(weights: dict[str, float], n: int = 15) -> list[tuple[str, float]]:
    """Return the top-n keyword weights sorted highest first."""
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    return ranked[:n]


def format_top_weights(
    weights: dict[str, float],
    n: int = 15,
    *,
    label: str | None = None,
) -> str:
    """Format the top-n weights for display."""
    top = get_top_weights(weights, n)
    lines = [label or f"Top {n} weights:"]
    if not top:
        lines.append("  (none)")
        return "\n".join(lines)
    width = max(len(word) for word, _ in top)
    for i, (word, value) in enumerate(top, start=1):
        lines.append(f"  {i:2}. {word:<{width}}  {value:+.4f}")
    return "\n".join(lines)


def print_top_weights(path: Path | str = DEFAULT_WEIGHTS_PATH, n: int = 15) -> None:
    """Load saved weights and print the top-n to stdout."""
    file_path = Path(path)
    store = load_weights(file_path)
    if store is None:
        print(f"No weights file at {file_path}")
        return
    stats = store.stats
    header = (
        f"Top {n} weights from {file_path} "
        f"(runs={stats.runs}, successes={stats.successes})"
    )
    print(format_top_weights(store.weights, n, label=header))
