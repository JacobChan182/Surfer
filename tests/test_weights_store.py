"""Tests for persistent weight storage."""

from pathlib import Path

from surfer.weights_store import WeightsStore, format_top_weights, get_top_weights, load_weights, reset_weights, save_weights


def test_save_and_load(tmp_path: Path):
    path = tmp_path / "weights.json"
    save_weights({"valve": 1.2, "steam": 0.8}, path=path, success=True, steps=15)

    loaded = load_weights(path)
    assert loaded is not None
    assert loaded.weights["valve"] == 1.2
    assert loaded.stats.runs == 1
    assert loaded.stats.successes == 1
    assert loaded.stats.last_success_steps == 15


def test_load_missing_returns_none(tmp_path: Path):
    assert load_weights(tmp_path / "missing.json") is None


def test_save_increments_runs(tmp_path: Path):
    path = tmp_path / "weights.json"
    first = save_weights({"game": 0.5}, path=path)
    second = save_weights({"game": 0.7}, path=path, existing=first)

    assert second.stats.runs == 2
    assert second.stats.successes == 0


def test_reset_weights(tmp_path: Path):
    path = tmp_path / "weights.json"
    save_weights({"valve": 1.0}, path=path)
    assert path.exists()

    reset_weights(path)
    assert not path.exists()


def test_from_dict_roundtrip():
    store = WeightsStore.from_dict(
        {
            "version": 1,
            "weights": {"fps": 0.3},
            "stats": {"runs": 5, "successes": 2, "last_success_steps": 20},
        }
    )
    data = store.to_dict()
    assert data["weights"]["fps"] == 0.3
    assert data["stats"]["runs"] == 5


def test_get_top_weights():
    weights = {"valve": 1.2, "steam": 0.9, "dog": -0.1, "game": 1.5}
    top = get_top_weights(weights, n=3)
    assert top == [("game", 1.5), ("valve", 1.2), ("steam", 0.9)]


def test_format_top_weights():
    text = format_top_weights({"valve": 1.0, "steam": 0.5}, n=2)
    assert "valve" in text
    assert "steam" in text
    assert "Top 2 weights" in text
