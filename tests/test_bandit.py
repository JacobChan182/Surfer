"""Tests for the keyword bandit policy."""

import pytest

from surfer.bandit import BanditPolicy
from surfer.wiki import WikiLink


def test_weights_increase_after_reward():
    policy = BanditPolicy(epsilon=0.0, learning_rate=0.5)
    features = frozenset({"valve", "corporation"})
    policy.update(features, reward=1.0)
    assert policy.weights["valve"] > 0
    assert policy.weights["corporation"] > 0


def test_select_prefers_high_weight_link():
    policy = BanditPolicy(epsilon=0.0)
    policy.weights["valve"] = 10.0
    policy.weights["dog"] = -5.0

    candidates = [
        WikiLink(title="Dog"),
        WikiLink(title="Valve Corporation"),
    ]
    chosen, _ = policy.select(candidates)
    assert chosen.title == "Valve Corporation"


def test_negative_priors_applied():
    policy = BanditPolicy(epsilon=0.0)
    assert policy.weights["disambiguation"] < 0


def test_top_weights():
    policy = BanditPolicy()
    policy.weights["valve"] = 5.0
    policy.weights["steam"] = 3.0
    policy.weights["dog"] = -1.0
    top = policy.top_weights(2)
    assert len(top) == 2
    assert top[0][0] == "valve"


def test_apply_deltas():
    policy = BanditPolicy()
    policy.apply_deltas({"valve": 0.4, "steam": 0.2}, scale=0.5)
    assert policy.weights["valve"] == pytest.approx(0.2)
    assert policy.weights["steam"] == pytest.approx(0.1)


def test_apply_deltas_clamped():
    policy = BanditPolicy()
    policy.weights["word"] = 9.5
    policy.apply_deltas({"word": 1.0}, scale=1.0)
    assert policy.weights["word"] == 10.0


def test_initial_weights_loaded():
    policy = BanditPolicy(weights={"valve": 2.0, "game": 1.0})
    assert policy.weights["valve"] == 2.0
    assert policy.weights["disambiguation"] == -0.3


def test_rank_candidates_unlimited():
    policy = BanditPolicy(epsilon=0.0)
    policy.weights["valve"] = 10.0
    candidates = [
        WikiLink(title="Dog"),
        WikiLink(title="Cat"),
        WikiLink(title="Valve Corporation"),
    ]
    ranked = policy.rank_candidates(candidates, limit=0)
    assert len(ranked) == 3
    assert ranked[0].title == "Valve Corporation"


def test_rank_candidates():
    policy = BanditPolicy(epsilon=0.0)
    policy.weights["valve"] = 10.0
    policy.weights["dog"] = -5.0
    candidates = [
        WikiLink(title="Dog"),
        WikiLink(title="Cat"),
        WikiLink(title="Valve Corporation"),
    ]
    ranked = policy.rank_candidates(candidates, limit=2)
    assert len(ranked) == 2
    assert ranked[0].title == "Valve Corporation"
