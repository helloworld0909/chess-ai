"""Tests for puzzle GRPO reward functions."""

import pytest

from verification.puzzle_rewards import (
    THRESHOLD,
    MAX_TOKENS,
    _extract_uci,
    _length_penalty,
    reward_format,
    reward_correct,
    compute_rewards,
)

# FEN for a simple position: e2 pawn can move to e4
_E4_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# FEN after 1.e4: knight on g1 can go to f3
_NF3_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


# ---------------------------------------------------------------------------
# _extract_uci
# ---------------------------------------------------------------------------


def test_extract_uci_json_uci():
    assert _extract_uci('{"move": "e2e4"}') == "e2e4"


def test_extract_uci_json_san():
    assert _extract_uci('{"move": "Nf3"}') == "Nf3"


def test_extract_uci_last_match():
    # Last JSON is at the end — valid
    text = '{"move": "d2d4"} ... {"move": "e2e4"}'
    assert _extract_uci(text) == "e2e4"


def test_extract_uci_json_in_middle_only():
    # JSON in middle with text after — should NOT match
    text = '{"move": "e2e4"} and then some more text'
    assert _extract_uci(text) is None


def test_extract_uci_json_in_think_not_at_end():
    # JSON only inside <think>, nothing after </think> — should NOT match
    text = '<think>maybe {"move": "e2e4"} is good</think>\nI recommend e4.'
    assert _extract_uci(text) is None


def test_extract_uci_none():
    assert _extract_uci("I don't know") is None


def test_extract_uci_in_think_block():
    text = '<think>Nf3 looks good</think>\n{"move": "Nf3"}'
    assert _extract_uci(text) == "Nf3"


def test_extract_uci_json_at_end_with_trailing_whitespace():
    text = '<think>thinking</think>\n{"move": "e2e4"}\n   '
    assert _extract_uci(text) == "e2e4"


def test_extract_uci_promotion():
    assert _extract_uci('{"move": "e7e8q"}') == "e7e8q"


# ---------------------------------------------------------------------------
# _length_penalty
# ---------------------------------------------------------------------------


def test_length_penalty_below_threshold():
    assert _length_penalty(0) == 0.0
    assert _length_penalty(THRESHOLD) == 0.0


def test_length_penalty_at_max():
    assert abs(_length_penalty(MAX_TOKENS) - (-0.2)) < 1e-9


def test_length_penalty_midpoint():
    mid = (THRESHOLD + MAX_TOKENS) // 2
    p = _length_penalty(mid)
    assert -0.2 < p < 0.0


def test_length_penalty_capped():
    assert _length_penalty(MAX_TOKENS * 10) == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# reward_format
# ---------------------------------------------------------------------------


def test_reward_format_present():
    assert reward_format('{"move": "e2e4"}') == 0.0


def test_reward_format_missing():
    assert reward_format("My move is e2e4") == -1.0


# ---------------------------------------------------------------------------
# reward_correct — UCI
# ---------------------------------------------------------------------------


def test_reward_correct_uci_correct():
    r = reward_correct('{"move": "e2e4"}', "e2e4", fen=_E4_FEN)
    assert r == pytest.approx(1.0)


def test_reward_correct_uci_wrong():
    r = reward_correct('{"move": "d2d4"}', "e2e4", fen=_E4_FEN)
    assert r == pytest.approx(0.0)


def test_reward_correct_no_format():
    r = reward_correct("e2e4 is the best move", "e2e4", fen=_E4_FEN)
    assert r == pytest.approx(-1.0)


def test_reward_correct_uci_with_length_penalty():
    r = reward_correct('{"move": "e2e4"}', "e2e4", completion_tokens=MAX_TOKENS, fen=_E4_FEN)
    assert r == pytest.approx(1.0 - 0.2)


def test_reward_correct_uci_wrong_with_length_penalty():
    r = reward_correct('{"move": "d2d4"}', "e2e4", completion_tokens=MAX_TOKENS, fen=_E4_FEN)
    assert r == pytest.approx(0.0 - 0.2)


def test_reward_correct_no_format_ignores_length():
    r = reward_correct("no json here", "e2e4", completion_tokens=MAX_TOKENS, fen=_E4_FEN)
    assert r == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# reward_correct — SAN
# ---------------------------------------------------------------------------


def test_reward_correct_san_correct():
    # Starting position: e4 is the move, SAN is "e4", UCI is "e2e4"
    r = reward_correct('{"move": "e4"}', "e2e4", fen=_E4_FEN)
    assert r == pytest.approx(1.0)


def test_reward_correct_san_wrong():
    r = reward_correct('{"move": "d4"}', "e2e4", fen=_E4_FEN)
    assert r == pytest.approx(0.0)


def test_reward_correct_san_illegal():
    # "Nf3" is not legal for black in starting pos
    r = reward_correct('{"move": "Nf3"}', "e2e4", fen=_E4_FEN)
    assert r == pytest.approx(0.0)


def test_reward_correct_san_with_check():
    # After 1.e4 e5 2.Bc4, Qh5 is legal for white but solution might differ
    # Use a simpler test: if SAN matches solution uci, reward=1.0
    # NF3_FEN: after 1.e4, it's black to move — black knight f6 (Nf6 = g8f6)
    r = reward_correct('{"move": "Nf6"}', "g8f6", fen=_NF3_FEN)
    assert r == pytest.approx(1.0)


def test_reward_correct_san_no_fen_returns_zero():
    # Without FEN, SAN can't be validated → 0.0 (not -1.0, format is present)
    r = reward_correct('{"move": "Nf3"}', "g1f3")
    assert r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_rewards
# ---------------------------------------------------------------------------


def test_compute_rewards_batch():
    completions = [
        '{"move": "e2e4"}',   # correct UCI
        '{"move": "d2d4"}',   # wrong UCI
        "no answer here",      # no format
    ]
    solutions = ["e2e4", "e2e4", "e2e4"]
    fens = [_E4_FEN, _E4_FEN, _E4_FEN]
    rewards = compute_rewards(completions, solutions, fens=fens)
    assert rewards[0] == pytest.approx(1.0)
    assert rewards[1] == pytest.approx(0.0)
    assert rewards[2] == pytest.approx(-1.0)


def test_compute_rewards_san_batch():
    completions = [
        '{"move": "e4"}',    # correct SAN
        '{"move": "d4"}',    # wrong SAN
    ]
    solutions = ["e2e4", "e2e4"]
    fens = [_E4_FEN, _E4_FEN]
    rewards = compute_rewards(completions, solutions, fens=fens)
    assert rewards[0] == pytest.approx(1.0)
    assert rewards[1] == pytest.approx(0.0)


def test_compute_rewards_with_tokens():
    completions = ['{"move": "e2e4"}', '{"move": "d2d4"}']
    solutions = ["e2e4", "e2e4"]
    tokens = [MAX_TOKENS, MAX_TOKENS]
    fens = [_E4_FEN, _E4_FEN]
    rewards = compute_rewards(completions, solutions, tokens, fens)
    assert rewards[0] == pytest.approx(0.8)
    assert rewards[1] == pytest.approx(-0.2)
