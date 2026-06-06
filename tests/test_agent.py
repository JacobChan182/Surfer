"""Tests for LLM critic integration and cross-run weight persistence."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from surfer.agent import AgentConfig, run_agent
from surfer.llm_critic import CriticResult, LLMCritic
from surfer.weights_store import load_weights, save_weights
from surfer.wiki import WikiLink, WikiPage


def _mock_wiki(pages: dict[str, WikiPage]):
    client = MagicMock()
    client.fetch.side_effect = lambda title: pages.get(title) or pages.get(title.split()[0])
    client.resolve_title.side_effect = lambda title: title
    client.filter_unvisited.side_effect = (
        lambda links, visited: [link for link in links if link.title not in visited]
    )
    return client


def test_run_agent_saves_weights_on_exit(tmp_path: Path):
    dog = WikiPage(title="Dog", extract="An animal.", links=[WikiLink(title="Cat")])
    cat = WikiPage(title="Cat", extract="Also an animal.", links=[])
    client = _mock_wiki({"Dog": dog, "Cat": cat})

    weights_path = tmp_path / "weights.json"
    config = AgentConfig(
        max_steps=2,
        save_weights=True,
        weights_file=weights_path,
    )

    with patch.object(client, "fetch", side_effect=[dog, cat]):
        result = run_agent(
            start_url="https://en.wikipedia.org/wiki/Dog",
            wiki=client,
            config=config,
        )

    assert not result.success
    loaded = load_weights(weights_path)
    assert loaded is not None
    assert loaded.stats.runs == 1


def test_run_agent_with_mocked_critic(tmp_path: Path):
    valve = WikiPage(
        title="Valve Corporation",
        extract="Video game company. Counter-Strike.",
        links=[WikiLink(title="Counter-Strike")],
    )
    cs = WikiPage(
        title="Counter-Strike (video game)",
        extract="A tactical shooter.",
        links=[],
    )
    client = MagicMock()
    client.fetch.side_effect = [valve, cs]
    client.filter_unvisited.side_effect = (
        lambda links, visited: [link for link in links if link.title not in visited]
    )

    mock_llm = MagicMock()
    critic = LLMCritic(llm=mock_llm, verbose=False)
    mock_llm.chat.return_value = json.dumps(
        {
            "progress_score": 0.9,
            "reasoning": "Very close",
            "weight_updates": {"valve": 0.3, "strike": 0.2},
        }
    )

    weights_path = tmp_path / "weights.json"
    config = AgentConfig(max_steps=10, save_weights=True, weights_file=weights_path)

    result = run_agent(
        start_url="https://en.wikipedia.org/wiki/Valve_Corporation",
        wiki=client,
        critic=critic,
        config=config,
    )

    assert result.success
    assert result.history[0].llm_progress_score == 0.9
    assert result.final_weights.get("valve", 0) > 0

    loaded = load_weights(weights_path)
    assert loaded is not None
    assert loaded.stats.successes == 1


def test_cross_run_weight_loading(tmp_path: Path):
    path = tmp_path / "weights.json"
    save_weights({"valve": 1.5, "steam": 1.0}, path=path, success=True, steps=10)

    loaded = load_weights(path)
    policy_weights = loaded.weights if loaded else {}
    assert policy_weights["valve"] == 1.5

    store = load_weights(path)
    store.stats.runs  # noqa: B018
    save_weights({"valve": 2.0}, path=path, existing=store)

    reloaded = load_weights(path)
    assert reloaded is not None
    assert reloaded.stats.runs == 2
    assert reloaded.weights["valve"] == 2.0
