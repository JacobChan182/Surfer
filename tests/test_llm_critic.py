"""Tests for LLM critic parsing and blending."""

import json

import pytest

from surfer.llm_critic import (
    _parse_critic_response,
    _parse_picker_response,
    blend_rewards,
    match_chosen_link,
)
from surfer.wiki import WikiLink


def test_parse_critic_response():
    raw = json.dumps(
        {
            "progress_score": 0.72,
            "reasoning": "Close to Valve ecosystem",
            "weight_updates": {"valve": 0.2, "politician": -0.1},
        }
    )
    result = _parse_critic_response(raw)
    assert result.progress_score == 0.72
    assert result.reasoning == "Close to Valve ecosystem"
    assert result.weight_updates["valve"] == 0.2
    assert result.weight_updates["politician"] == -0.1
    assert result.chosen_link is None


def test_parse_picker_response():
    raw = json.dumps(
        {
            "progress_score": 0.8,
            "chosen_link": "Valve Corporation",
            "reasoning": "Direct path to CS publisher",
            "weight_updates": {"valve": 0.3},
        }
    )
    result = _parse_picker_response(raw)
    assert result.chosen_link == "Valve Corporation"
    assert result.progress_score == 0.8


def test_parse_critic_response_with_fence():
    raw = '```json\n{"progress_score": 0.5, "reasoning": "ok", "weight_updates": {}}\n```'
    result = _parse_critic_response(raw)
    assert result.progress_score == 0.5


def test_parse_clamps_score():
    raw = json.dumps(
        {"progress_score": 1.5, "reasoning": "x", "weight_updates": {"a": 0.99}}
    )
    result = _parse_critic_response(raw)
    assert result.progress_score == 1.0
    assert result.weight_updates["a"] == 0.5


def test_blend_rewards():
    assert blend_rewards(0.0, 1.0, 0.6) == pytest.approx(0.6)
    assert blend_rewards(0.5, 0.5, 0.6) == pytest.approx(0.5)
    assert blend_rewards(1.0, 0.0, 0.0) == pytest.approx(1.0)


def test_match_chosen_link_exact():
    candidates = [WikiLink(title="Dog"), WikiLink(title="Valve Corporation")]
    matched = match_chosen_link("Valve Corporation", candidates)
    assert matched is not None
    assert matched.title == "Valve Corporation"


def test_match_chosen_link_case_insensitive():
    candidates = [WikiLink(title="Valve Corporation")]
    matched = match_chosen_link("valve corporation", candidates)
    assert matched is not None
