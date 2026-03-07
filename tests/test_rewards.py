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
    reward_annotation_structural,
    reward_breadth,
    reward_comment_format,
    reward_conciseness,
    reward_depth,
    reward_educational,
    reward_eval_accuracy,
    reward_format,
    reward_legality,
    reward_relevance,
    reward_tone,
)

# ---------------------------------------------------------------------------
# Fixtures: canned completions
# ---------------------------------------------------------------------------

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# FEN after White plays e4 — it is now Black's turn
AFTER_E4_FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# Three legal lines from the post-e4 position (Black to move).
# First moves must be Black's legal replies: e5, d5, Nf6 etc.
LEGAL_COMPLETION = """\
<line>LINE 1: e5 (contest center) → Nf3 (develop knight) → Nc6 (develop) | eval: equal</line>
<line>LINE 2: d5 (contest center) → exd5 (capture pawn) → Qxd5 (recapture queen) | eval: equal</line>
<line>LINE 3: Nf6 (develop knight) → e5 (gain space) → Nd5 (retreat to outpost) | eval: equal</line>
"""

# Both lines have illegal first moves for Black from post-e4 (e4 already played, e7-e4 not legal)
ILLEGAL_COMPLETION = """\
<line>LINE 1: e4 (illegal) → Nf6 (develop) | eval: equal</line>
<line>LINE 2: d4 (illegal for black) → d5 (pawn push) | eval: equal</line>
"""

# One legal (e5 for Black), one illegal (e4 — White pawn square, illegal for Black) — mean = 0.0
MIXED_COMPLETION = """\
<line>LINE 1: e4 (illegal for black) → Nf6 (develop) | eval: equal</line>
<line>LINE 2: e5 (contest center) → Nf3 (develop) | eval: equal</line>
"""

# All lines start with same first move (no breadth) — e5 is legal for Black
SAME_FIRST_MOVE = """\
<line>LINE 1: e5 (contest) → Nf3 (develop) → Nc6 (develop) | eval: equal</line>
<line>LINE 2: e5 (contest) → Nf3 (develop) → d6 (solid) | eval: equal</line>
<line>LINE 3: e5 (contest) → Bc4 (develop bishop) → Nf6 (develop) | eval: equal</line>
"""

# Short lines (1 move each) — should score low on depth
SHORT_COMPLETION = """\
<line>LINE 1: e5 (contest center) | eval: equal</line>
<line>LINE 2: d5 (contest center) | eval: equal</line>
"""

# No lines at all
EMPTY_COMPLETION = "I don't know what to say."


# Prompt with both FENs embedded — matches actual training data format.
# move_san is the White move played; engine lines start from fen_after (Black to move).
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
        # All 3 moves in each line legal → score = 1.0
        assert scores[0] == 1.0

    def test_illegal_completion_negative(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_legality([prompt], [_completion(ILLEGAL_COMPLETION)])
        # Both lines: first move illegal (0/2 legal) → line score = -1.0 → mean = -1.0
        assert scores[0] == -1.0

    def test_mixed_completion_zero(self):
        prompt = _make_prompt()
        scores = reward_legality([prompt], [_completion(MIXED_COMPLETION)])
        # LINE 1: e4 illegal for Black (0/2 legal) → -1.0
        # LINE 2: e5 legal for Black (1/2 legal, Nf3 after is White's move) → varies
        # mean is between -1.0 and +1.0
        assert -1.0 <= scores[0] <= 1.0

    def test_partial_credit(self):
        # A line where first move is legal but second repeats a pawn push illegally
        # From post-e4 (Black to move): e5 is legal, then Nf3 (White), then e5 again illegal
        partial = "<line>LINE 1: e5 (contest) → Nf3 (develop) → e5 (illegal repeat) | eval: equal</line>\n"
        prompt = _make_prompt()
        scores = reward_legality([prompt], [_completion(partial)])
        # 2 legal out of 3 moves → 2*(2/3) - 1 = 0.333...
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
# R4 — reward_depth
# ---------------------------------------------------------------------------


class TestRewardDepth:
    def test_three_move_lines_below_target(self):
        prompt = _make_prompt()
        scores = reward_depth([prompt], [_completion(LEGAL_COMPLETION)])
        # Each line has 3 moves; target is 2 → score = min(3,2)/2 = 1.0
        assert abs(scores[0] - 1.0) < 0.01

    def test_short_lines_low_score(self):
        prompt = _make_prompt()
        scores = reward_depth([prompt], [_completion(SHORT_COMPLETION)])
        # Each line has 1 move (e5/d5 legal for Black); target is 2 → score = 1/2 = 0.5
        assert abs(scores[0] - 0.5) < 0.01

    def test_no_lines_minus_one(self):
        prompt = _make_prompt()
        scores = reward_depth([prompt], [_completion(EMPTY_COMPLETION)])
        assert scores[0] == -1.0

    def test_illegal_lines_excluded(self):
        # ILLEGAL_COMPLETION: both lines start with illegal moves for Black → no legal lines → -1.0
        prompt = _make_prompt()
        scores = reward_depth([prompt], [_completion(ILLEGAL_COMPLETION)])
        assert scores[0] == -1.0

    def test_long_line_capped_at_one(self):
        # Build a very long (12-move) line starting from post-e4 (Black to move) — score = 1.0
        # Black plays e5, White Nf3, Black Nc6, White Bb5 (Ruy Lopez), etc.
        long_line = (
            "<line>LINE 1: e5 (contest) → Nf3 (develop) → Nc6 (develop)"
            " → Bb5 (pin) → a6 (challenge) → Ba4 (retreat) → Nf6 (develop)"
            " → O-O (castle) → Be7 (develop) → Re1 (centralize) → b5 (expand)"
            " → Bb3 (retreat bishop) | eval: equal</line>"
        )
        prompt = _make_prompt()
        scores = reward_depth([prompt], [_completion(long_line)])
        assert scores[0] == 1.0


# ---------------------------------------------------------------------------
# R5 — reward_breadth
# ---------------------------------------------------------------------------


class TestRewardBreadth:
    def test_all_different_first_moves_score_one(self):
        prompt = _make_prompt()
        scores = reward_breadth([prompt], [_completion(LEGAL_COMPLETION)])
        # e5, d5, Nf6 — all different legal Black moves from post-e4
        assert scores[0] == 1.0

    def test_same_first_move_score_low(self):
        prompt = _make_prompt()
        scores = reward_breadth([prompt], [_completion(SAME_FIRST_MOVE)])
        # All three lines start with e5 (legal for Black) → unique_ratio = 1/3
        assert abs(scores[0] - 1 / 3) < 0.01

    def test_illegal_lines_excluded(self):
        # ILLEGAL_COMPLETION: e4/d4 both illegal for Black from post-e4 → no legal lines → -1.0
        prompt = _make_prompt()
        scores = reward_breadth([prompt], [_completion(ILLEGAL_COMPLETION)])
        assert scores[0] == -1.0

    def test_no_lines_minus_one(self):
        prompt = _make_prompt()
        scores = reward_breadth([prompt], [_completion(EMPTY_COMPLETION)])
        assert scores[0] == -1.0


# ---------------------------------------------------------------------------
# R6 — reward_relevance
# ---------------------------------------------------------------------------


class TestRewardRelevance:
    def test_legal_first_moves_score_positive(self):
        prompt = _make_prompt()
        scores = reward_relevance([prompt], [_completion(LEGAL_COMPLETION)])
        assert scores[0] > 0

    def test_illegal_first_move_score_negative(self):
        prompt = _make_prompt()
        scores = reward_relevance([prompt], [_completion(ILLEGAL_COMPLETION)])
        # e4/d4 not legal for Black from post-e4 position
        assert scores[0] < 1.0  # at least one bad first move

    def test_no_lines_minus_one(self):
        prompt = _make_prompt()
        scores = reward_relevance([prompt], [_completion(EMPTY_COMPLETION)])
        assert scores[0] == -1.0


# ---------------------------------------------------------------------------
# combined_reward — hard gate behaviour
# ---------------------------------------------------------------------------


class TestCombinedReward:
    def test_illegal_completion_dominated_by_gate(self):
        # All lines illegal for Black from post-e4 + no comment → combined near -1.0.
        # e4/d4 are not legal Black moves from post-e4 position.
        all_illegal = (
            "<line>LINE 1: e4 (illegal for black) → Nf6 (develop) | eval: equal</line>\n"
            "<line>LINE 2: d4 (illegal for black) → d5 (pawn) | eval: equal</line>\n"
        )
        prompt = _make_prompt()
        with patch("verification.rewards._eval_fen", return_value=0):
            scores = combined_reward([prompt], [_completion(all_illegal)])
        # Gate fires: legality heavily negative; combined < 0
        assert scores[0] < 0.0

    def test_legal_completion_positive(self):
        prompt = _make_prompt()
        # Stockfish returns 0 cp (equal) — R2 exact match → 1.0
        # Combined should be positive.
        with patch("verification.rewards._eval_fen", return_value=0):
            scores = combined_reward([prompt], [_completion(LEGAL_COMPLETION)])
        assert scores[0] >= 0.0

    def test_batch_size_matches(self):
        prompts = [_make_prompt()] * 3
        completions = [
            _completion(LEGAL_COMPLETION),
            _completion(ILLEGAL_COMPLETION),
            _completion(LEGAL_COMPLETION),
        ]
        with patch("verification.rewards._eval_fen", return_value=0):
            scores = combined_reward(prompts, completions)
        assert len(scores) == 3


# ---------------------------------------------------------------------------
# Coaching comment reward helpers
# ---------------------------------------------------------------------------

# A completion with lines + a good coaching comment
GOOD_COMMENT_COMPLETION = """\
<line>LINE 1: e4 (control center) → e5 (contest center) → Nf3 (develop knight) | eval: equal</line>
<line>LINE 2: d4 (control center) → d5 (contest center) | eval: equal</line>

You played e4, which fights for the center immediately. Instead of e4, consider d4 to control \
even more central squares. Since both moves are strong, your instinct was solid — \
the key idea is controlling space and developing your pieces quickly."""

# A completion with no chess concept vocabulary
NO_CONCEPT_COMPLETION = """\
<line>LINE 1: e4 (control center) → e5 (contest center) | eval: equal</line>

The player should have played d4. The move was not great. \
One should always think carefully before choosing a move."""

# A completion with 7 sentences (over-long/padding)
LONG_COMMENT_COMPLETION = """\
<line>LINE 1: e4 (center) → e5 → Nf3 | eval: equal</line>

You played e4. This is a good move. It controls the center. You should consider d4 too. \
Both moves are common. The knight on f3 is well-placed. You did a great job here."""

# A completion with no comment after the lines
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
# reward_comment_format
# ---------------------------------------------------------------------------


class TestRewardCommentFormat:
    def test_with_comment_scores_positive(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_comment_format([prompt], [_completion(GOOD_COMMENT_COMPLETION)])
        assert scores[0] == 1.0

    def test_no_comment_scores_zero(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_comment_format([prompt], [_completion(NO_COMMENT_COMPLETION)])
        assert scores[0] == 0.0

    def test_no_lines_scores_minus_one(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_comment_format([prompt], [_completion(EMPTY_COMPLETION)])
        assert scores[0] == -1.0


# ---------------------------------------------------------------------------
# reward_tone
# ---------------------------------------------------------------------------


class TestRewardTone:
    def test_rich_concepts_scores_max(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_tone([prompt], [_completion(GOOD_COMMENT_COMPLETION)])
        # "center", "space", "development" → 3+ concepts → +1.0
        assert scores[0] == 1.0

    def test_no_concepts_scores_minus_one(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_tone([prompt], [_completion(NO_CONCEPT_COMPLETION)])
        # No chess terminology → -1.0
        assert scores[0] == -1.0

    def test_no_comment_returns_zero(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_tone([prompt], [_completion(NO_COMMENT_COMPLETION)])
        assert scores[0] == 0.0


# ---------------------------------------------------------------------------
# reward_conciseness
# ---------------------------------------------------------------------------


class TestRewardConciseness:
    def test_three_sentences_scores_one(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_conciseness([prompt], [_completion(GOOD_COMMENT_COMPLETION)])
        assert scores[0] == 1.0

    def test_seven_sentences_scores_minus_one(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_conciseness([prompt], [_completion(LONG_COMMENT_COMPLETION)])
        assert scores[0] == -1.0

    def test_no_comment_returns_zero(self):
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_conciseness([prompt], [_completion(NO_COMMENT_COMPLETION)])
        assert scores[0] == 0.0


# ---------------------------------------------------------------------------
# reward_educational
# ---------------------------------------------------------------------------


class TestRewardEducational:
    def test_good_comment_scores_high(self):
        # GOOD_COMMENT_COMPLETION references e4 (from lines) + "space"/"center" + "since"
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
        # Comment mentions e4 then e5 — these are consecutive in LINE 1
        grounded = """\
<line>LINE 1: e4 (center) → e5 (contest) → Nf3 (develop) | eval: equal</line>

After e4 and then e5, the center is contested and development is key since both sides fight for space."""
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_educational([prompt], [_completion(grounded)])
        assert scores[0] > 0.0

    def test_non_consecutive_moves_penalized(self):
        # Comment mentions e4 then Nf3 — these are NOT consecutive (e5 is between them)
        hallucinated = """\
<line>LINE 1: e4 (center) → e5 (contest) → Nf3 (develop) | eval: equal</line>

After e4 and then Nf3, White gains central control since this creates a fork."""
        prompt = _make_prompt(STARTING_FEN)
        scores = reward_educational([prompt], [_completion(hallucinated)])
        # e4→Nf3 are not consecutive in the line (e5 is between) → grounded_score = -1.0
        assert scores[0] < 0.5  # penalized vs fully grounded comment
