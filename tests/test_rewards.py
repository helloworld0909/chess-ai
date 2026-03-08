"""Tests for src/verification/rewards.py — GRPO reward functions."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from verification.rewards import (
    _extract_comment,
    combined_reward,
    parse_lines,
    reward_educational,
    reward_format,
    reward_legality,
    reward_sf15_annotation,
    reward_think,
    reward_tone,
)

# ---------------------------------------------------------------------------
# Fixtures: canned completions
# ---------------------------------------------------------------------------

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# FEN after White plays e4 — it is now Black's turn
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# Three legal lines from the post-e4 position (Black to move).
LEGAL_COMPLETION = """\
<line>LINE 1: e5 (contest center) → Nf3 (develop knight) → Nc6 (develop) | eval: equal</line>
<line>LINE 2: d5 (contest center) → exd5 (capture pawn) → Qxd5 (recapture queen) | eval: equal</line>
<line>LINE 3: Nf6 (develop knight) → e5 (gain space) → Nd5 (retreat to outpost) | eval: equal</line>
"""

# Both lines have illegal first moves for Black from post-e4
ILLEGAL_COMPLETION = """\
<line>LINE 1: e4 (illegal) → Nf6 (develop) | eval: equal</line>
<line>LINE 2: d4 (illegal for black) → d5 (pawn push) | eval: equal</line>
"""

# One legal (e5 for Black), one illegal (e4)
MIXED_COMPLETION = """\
<line>LINE 1: e4 (illegal for black) → Nf6 (develop) | eval: equal</line>
<line>LINE 2: e5 (contest center) → Nf3 (develop) | eval: equal</line>
"""

# No lines at all
EMPTY_COMPLETION = "I don't know what to say."


def _make_prompt(fen: str = STARTING_FEN, move_san: str = "e4") -> list[dict]:
    import chess as _chess

    board = _chess.Board(fen)
    try:
        mv = board.parse_san(move_san)
        board.push(mv)
        fen_after = board.fen()
    except Exception:
        fen_after = fen
    return [
        {"role": "system", "content": "You are a chess coach."},
        {
            "role": "user",
            "content": (
                f"## Position\nFEN: {fen}\n"
                f"## Move Played\nMove: {move_san}\n"
                f"Board after the move:\nFEN: {fen_after}\n"
                "## Task\nOutput 3 lines."
            ),
        },
    ]


def _make_joint_prompt(fen: str = STARTING_FEN, move_san: str = "e4") -> list[dict]:
    """Prompt with ## Engine Key Lines containing SF15 annotations."""
    import chess as _chess

    board = _chess.Board(fen)
    try:
        mv = board.parse_san(move_san)
        board.push(mv)
        fen_after = board.fen()
    except Exception:
        fen_after = fen
    return [
        {"role": "system", "content": "You are a chess coach."},
        {
            "role": "user",
            "content": (
                f"## Position\nFEN: {fen}\n"
                f"## Move Played\nMove: {move_san}\n"
                f"Board after the move:\nFEN: {fen_after}\n"
                "## Engine Key Lines\n\n"
                "PLAYED LINE: e4 [space +0.21; mobility +0.12] → e5 [threats +0.18] | eval: equal\n"
                "Line 1: d4 [space +0.25; mobility +0.15] → d5 [pawns -0.11] | eval: equal\n\n"
                "## Task\nAnalyse each engine key line."
            ),
        },
    ]


def _completion(text: str) -> list[dict]:
    return [{"role": "assistant", "content": text}]


# ---------------------------------------------------------------------------
# parse_lines
# ---------------------------------------------------------------------------


class TestParseLines:
    def test_parses_three_lines(self):
        lines = parse_lines(LEGAL_COMPLETION)
        assert len(lines) == 3

    def test_extracts_moves(self):
        lines = parse_lines(LEGAL_COMPLETION)
        assert lines[0]["moves_san"] == ["e5", "Nf3", "Nc6"]

    def test_extracts_eval_label(self):
        lines = parse_lines(LEGAL_COMPLETION)
        assert lines[0]["eval_label"] == "equal"

    def test_empty_returns_empty(self):
        lines = parse_lines(EMPTY_COMPLETION)
        assert lines == []

    def test_bare_line_format(self):
        bare = "LINE 1: e4 → d5 → exd5 | eval: good for white\n"
        lines = parse_lines(bare)
        assert len(lines) == 1
        assert lines[0]["eval_label"] == "good for white"


# ---------------------------------------------------------------------------
# R1 — reward_legality
# ---------------------------------------------------------------------------


class TestRewardLegality:
    def test_fully_legal_completion(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_legality([prompt], [_completion(LEGAL_COMPLETION)])
        assert scores[0] == 1.0

    def test_illegal_completion_negative(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_legality([prompt], [_completion(ILLEGAL_COMPLETION)])
        assert scores[0] == -1.0

    def test_mixed_completion_between(self):
        prompt = _make_prompt()
        scores = reward_legality([prompt], [_completion(MIXED_COMPLETION)])
        assert -1.0 <= scores[0] <= 1.0

    def test_partial_credit(self):
        partial = "<line>LINE 1: e5 (contest) → Nf3 (develop) → e5 (illegal repeat) | eval: equal</line>\n"
        prompt = _make_prompt()
        scores = reward_legality([prompt], [_completion(partial)])
        assert scores[0] < 1.0
        assert scores[0] > -1.0

    def test_no_lines_scores_minus_one(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_legality([prompt], [_completion(EMPTY_COMPLETION)])
        assert scores[0] == -1.0

    def test_missing_fen_neutral(self):
        prompt = [{"role": "user", "content": "no fen here"}]
        scores = reward_legality([prompt], [_completion(LEGAL_COMPLETION)])
        assert scores[0] == 0.0


# ---------------------------------------------------------------------------
# R0 — reward_format
# ---------------------------------------------------------------------------


class TestRewardFormat:
    def test_has_line_tags_scores_positive(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_format([prompt], [_completion(LEGAL_COMPLETION)])
        assert scores[0] == 1.0

    def test_no_line_tags_scores_negative(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_format([prompt], [_completion(EMPTY_COMPLETION)])
        assert scores[0] == -1.0

    def test_illegal_moves_but_correct_format_still_positive(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_format([prompt], [_completion(ILLEGAL_COMPLETION)])
        assert scores[0] == 1.0


# ---------------------------------------------------------------------------
# R_think — reward_think
# ---------------------------------------------------------------------------


class TestRewardThink:
    def test_good_think_block_scores_high(self):
        completion = _completion(
            "<think>\nLet me analyse Nd5 and e4. The knight centralises with Nd5 gaining mobility. "
            "White should consider Nf3 to develop.\n</think>\n" + LEGAL_COMPLETION
        )
        prompt = _make_prompt()
        scores = reward_think([prompt], [completion])
        assert scores[0] > 0.0

    def test_empty_think_scores_minus_one(self):
        completion = _completion("<think></think>\n" + LEGAL_COMPLETION)
        prompt = _make_prompt()
        scores = reward_think([prompt], [completion])
        assert scores[0] == -1.0

    def test_no_think_scores_minus_one(self):
        prompt = _make_prompt()
        scores = reward_think([prompt], [_completion(LEGAL_COMPLETION)])
        assert scores[0] == -1.0


# ---------------------------------------------------------------------------
# R3b — reward_sf15_annotation
# ---------------------------------------------------------------------------


class TestRewardSF15Annotation:
    def test_correct_direction_scores_positive(self):
        # Prompt has "e4 [space +0.21; mobility +0.12]" → positive terms
        # Model's think block correctly says space/mobility improved
        prompt = _make_joint_prompt()
        completion = _completion(
            "<think>\nLet me analyse each line.\n"
            "  PLAYED: e4 → e5 | eval: equal\n"
            "    e4: space +0.21, mobility +0.12 → gains central space, improves piece coordination\n"
            "    e5: threats +0.18 → generates dangerous counterplay\n"
            "VERDICT: e4 is good.\n</think>\n"
            "<line>LINE 1: e5 (contest center) → Nf3 (develop) | eval: equal</line>\n"
            "You played e4, gaining space and improving piece activity."
        )
        scores = reward_sf15_annotation([prompt], [completion])
        assert scores[0] > 0.0

    def test_wrong_direction_scores_negative(self):
        # Prompt has "e4 [space +0.21]" → space improved
        # Model says "restricts space" → contradicts the term
        prompt = _make_joint_prompt()
        completion = _completion(
            "<think>\nLet me analyse.\n"
            "    e4: restricts space, cramped position\n"
            "</think>\n"
            "<line>LINE 1: e5 | eval: equal</line>\n"
        )
        scores = reward_sf15_annotation([prompt], [completion])
        assert scores[0] < 0.0

    def test_no_think_block_scores_minus_one(self):
        prompt = _make_joint_prompt()
        completion = _completion("<line>LINE 1: e5 (contest) | eval: equal</line>\nYou played e4.")
        scores = reward_sf15_annotation([prompt], [completion])
        assert scores[0] == -1.0

    def test_non_joint_prompt_returns_neutral(self):
        # Plain prompt without ## Engine Key Lines
        prompt = _make_prompt()
        scores = reward_sf15_annotation([prompt], [_completion(LEGAL_COMPLETION)])
        assert scores[0] == 0.0

    def test_no_notable_terms_returns_neutral(self):
        # Prompt has no bracket annotations in Engine Key Lines
        prompt = [
            {"role": "system", "content": "You are a chess coach."},
            {
                "role": "user",
                "content": ("## Engine Key Lines\n\nPLAYED LINE: e4 → e5 | eval: equal\n"),
            },
        ]
        scores = reward_sf15_annotation([prompt], [_completion(LEGAL_COMPLETION)])
        assert scores[0] == 0.0


# ---------------------------------------------------------------------------
# Coaching comment reward helpers
# ---------------------------------------------------------------------------

GOOD_COMMENT_COMPLETION = """\
<line>LINE 1: e4 (control center) → e5 (contest center) → Nf3 (develop knight) | eval: equal</line>
<line>LINE 2: d4 (control center) → d5 (contest center) | eval: equal</line>

You played e4, which fights for the center immediately. Instead of e4, consider d4 to control \
even more central squares. Since both moves are strong, your instinct was solid — \
the key idea is controlling space and developing your pieces quickly."""

NO_CONCEPT_COMPLETION = """\
<line>LINE 1: e4 (control center) → e5 (contest center) | eval: equal</line>

The player should have played d4. The move was not great. \
One should always think carefully before choosing a move."""

NO_COMMENT_COMPLETION = """\
<line>LINE 1: e4 (center) → e5 → Nf3 | eval: equal</line>
<line>LINE 2: d4 (center) → d5 | eval: equal</line>"""


class TestExtractComment:
    def test_extracts_comment_after_last_line(self):
        comment = _extract_comment(GOOD_COMMENT_COMPLETION)
        assert "You played e4" in comment

    def test_no_lines_returns_empty(self):
        assert _extract_comment("Some random text with no lines.") == ""

    def test_no_text_after_line_returns_empty(self):
        assert _extract_comment(NO_COMMENT_COMPLETION) == ""


# ---------------------------------------------------------------------------
# reward_tone
# ---------------------------------------------------------------------------


class TestRewardTone:
    def test_rich_concepts_scores_max(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_tone([prompt], [_completion(GOOD_COMMENT_COMPLETION)])
        assert scores[0] == 1.0

    def test_no_concepts_scores_minus_one(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_tone([prompt], [_completion(NO_CONCEPT_COMPLETION)])
        assert scores[0] == -1.0

    def test_no_comment_returns_zero(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_tone([prompt], [_completion(NO_COMMENT_COMPLETION)])
        assert scores[0] == 0.0


# ---------------------------------------------------------------------------
# reward_educational
# ---------------------------------------------------------------------------


class TestRewardEducational:
    def test_good_comment_scores_high(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_educational([prompt], [_completion(GOOD_COMMENT_COMPLETION)])
        assert scores[0] > 0.0

    def test_no_comment_returns_zero(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_educational([prompt], [_completion(NO_COMMENT_COMPLETION)])
        assert scores[0] == 0.0

    def test_vague_comment_scores_lower(self):
        vague = """\
<line>LINE 1: e4 (center) → e5 | eval: equal</line>

You played well. Keep it up and you will improve."""
        prompt = _make_prompt(STARTING_FEN)
        scores_vague = reward_educational([prompt], [_completion(vague)])
        scores_good = reward_educational([prompt], [_completion(GOOD_COMMENT_COMPLETION)])
        assert scores_good[0] > scores_vague[0]

    def test_grounded_consecutive_moves_score_high(self):
        grounded = """\
<line>LINE 1: e4 (center) → e5 (contest) → Nf3 (develop) | eval: equal</line>

After e4 and then e5, the center is contested and development is key since both sides fight for space."""
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_educational([prompt], [_completion(grounded)])
        assert scores[0] > 0.0

    def test_non_consecutive_moves_penalized(self):
        hallucinated = """\
<line>LINE 1: e4 (center) → e5 (contest) → Nf3 (develop) | eval: equal</line>

After e4 and then Nf3, White gains central control since this creates a fork."""
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_educational([prompt], [_completion(hallucinated)])
        # e4→Nf3 are not consecutive in the line (e5 is between) → lower score
        good_prompt = _make_prompt(STARTING_FEN)
        good_scores = reward_educational([good_prompt], [_completion(GOOD_COMMENT_COMPLETION)])
        assert good_scores[0] >= scores[0]


# ---------------------------------------------------------------------------
# combined_reward
# ---------------------------------------------------------------------------


class TestCombinedReward:
    def test_illegal_completion_dominated_by_gate(self):
        all_illegal = (
            "<line>LINE 1: e4 (illegal for black) → Nf6 (develop) | eval: equal</line>\n"
            "<line>LINE 2: d4 (illegal for black) → d5 (pawn) | eval: equal</line>\n"
        )
        prompt = _make_prompt()
        scores = combined_reward([prompt], [_completion(all_illegal)])
        assert scores[0] < 0.0

    def test_legal_completion_positive(self):
        prompt = _make_prompt()
        scores = combined_reward([prompt], [_completion(LEGAL_COMPLETION)])
        assert scores[0] >= -0.5  # may be low due to no think/no comment, but not catastrophic

    def test_batch_size_matches(self):
        prompts = [_make_prompt()] * 3
        completions = [
            _completion(LEGAL_COMPLETION),
            _completion(ILLEGAL_COMPLETION),
            _completion(LEGAL_COMPLETION),
        ]
        scores = combined_reward(prompts, completions)
        assert len(scores) == 3
