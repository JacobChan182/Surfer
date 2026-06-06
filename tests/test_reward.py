"""Tests for reward scoring and terminal detection."""

from surfer.reward import score_page


def test_terminal_counter_strike():
    result = score_page("Counter-Strike", "A video game.")
    assert result.is_terminal
    assert result.reward == 1.0


def test_terminal_counter_strike_video_game():
    result = score_page("Counter-Strike (video game)", "First game in the series.")
    assert result.is_terminal
    assert result.reward == 1.0


def test_partial_reward_valve():
    result = score_page("Valve Corporation", "Valve is a video game developer known for Steam.")
    assert not result.is_terminal
    assert result.reward > 0
    assert "valve" in result.matched_keywords


def test_partial_reward_fps():
    result = score_page("First-person shooter", "FPS games include multiplayer modes.")
    assert not result.is_terminal
    assert result.reward > 0


def test_zero_reward_unrelated():
    result = score_page("Dog", "The dog is a domesticated animal.")
    assert not result.is_terminal
    assert result.reward == 0.0
    assert result.matched_keywords == []
