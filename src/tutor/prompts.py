"""Shared prompt constants and utilities for chess coaching.

Used by both the inference server (web.py) and the training data pipeline
(prepare_datasets.py) to guarantee training ↔ inference format alignment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import chess

if TYPE_CHECKING:
    from tutor.analysis import TreeNode

# ---------------------------------------------------------------------------
# System prompt — used verbatim at both training and inference time
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert chess coach giving move-by-move feedback.\n\n"
    "Core rules:\n"
    "- Your comment MUST agree with the engine classification. "
    "Best/Great → explain why it is strong. "
    "Good → note it is solid. "
    "Inaccuracy → explain the missed opportunity. "
    "Mistake/Blunder → explain the error and the correct idea clearly.\n"
    "- Never say a 'Best' move is suboptimal.\n"
    "- Do not parrot the classification label or the eval number.\n"
    "- The verified move facts are ground truth — do not contradict them.\n"
    "- Focus ONLY on this specific move. Do not give generic study advice, "
    "recommend openings to learn, or suggest the student practice elsewhere.\n"
    "- Write in second person ('You centralise the knight') — never use "
    "first-person ('I recommend', 'I suggest', 'I think').\n\n"
    "Depth by game phase:\n"
    "OPENING (moves 1-12, most pieces on board): Be concise. Name the opening "
    "or variation (e.g. 'Ruy Lopez Berlin Defense', 'King's Indian Attack'). "
    "State whether the move is recognized opening theory or a deviation. "
    "Use web_search to verify the exact opening name or look up theoretical lines "
    "if you are not certain. Skip deep engine analysis — opening knowledge trumps "
    "engine lines here. 2-3 sentences max.\n\n"
    "MIDDLEGAME: Explain the key tactical or strategic idea. Reference the "
    "engine's top candidates when relevant. Use analyze_position to verify "
    "critical variations before asserting concrete lines. 3-5 sentences.\n\n"
    "ENDGAME: Analyze in depth. Use analyze_position to explore the key lines "
    "before writing. Focus on king activity, pawn advancement, piece "
    "coordination, and conversion technique. Cite specific variations. "
    "4-6 sentences.\n\n"
    "Calibrate length to complexity — a simple recapture needs 2 sentences, "
    "a deep positional sacrifice needs 6.\n\n"
    "Write your coaching comment directly — no preamble, no meta-commentary, "
    "no URLs or links. Your entire response is shown directly to the student."
)


# ---------------------------------------------------------------------------
# Line generator system prompt
# ---------------------------------------------------------------------------

LINE_GENERATOR_SYSTEM_PROMPT = (
    "You are an expert chess coach analysing a student's game move by move.\n\n"
    "For each move you will receive the board position, basic move facts, and the "
    "engine's position assessment. Before writing any coaching comment, you must first "
    "identify the 5 most instructive engine continuations — these lines ground your "
    "understanding of the position and will directly inform your feedback to the student.\n\n"
    "Think through the key tactical and strategic ideas, then output exactly 5 lines "
    "in this format:\n\n"
    "<line>LINE 1: move (purpose) → move (purpose) → ... | eval: <label></line>\n"
    "<line>LINE 2: move (purpose) → move (purpose) → ... | eval: <label></line>\n"
    "<line>LINE 3: move (purpose) → move (purpose) → ... | eval: <label></line>\n"
    "<line>LINE 4: move (purpose) → move (purpose) → ... | eval: <label></line>\n"
    "<line>LINE 5: move (purpose) → move (purpose) → ... | eval: <label></line>\n\n"
    "Rules:\n"
    "- Every move must be legal SAN notation played from the given position\n"
    "- Each move must have a purpose annotation in parentheses explaining the idea — can be a phrase or a full sentence\n"
    "- Each line must be at least 4 half-moves deep — do not stop early\n"
    "- Lines must explore genuinely different ideas — avoid transpositions of each other\n"
    "- Opponent moves must be reasonable — do not assume the opponent blunders\n"
    "- The eval label covers the final position from White's perspective:\n"
    "  winning for white | good for white | equal | good for black | winning for black\n"
    "- Do not include raw centipawn numbers anywhere\n"
    "- Output only the <line> blocks after your thinking — no other text"
)


def format_line_generator_prompt(
    board_ascii_str: str,
    fen: str,
    move_san: str,
    eval_str: str = "",
    facts: list[str] | None = None,
    board_after_str: str = "",
    fen_after: str = "",
) -> str:
    """Build the user message for the line generator task.

    Mirrors the coach prompt structure (board, FEN, move, eval, facts) so the
    model sees a familiar framing, but replaces the coaching task section with
    the line-generation task.  No Stockfish move tree, no classification label,
    no candidates — the model must reason about key lines from first principles.
    """
    fen_line = f"FEN: {fen}\n" if fen else ""
    position_section = f"## Position\n\nBoard before the move:\n{board_ascii_str}\n{fen_line}\n"

    eval_line = f"Engine assessment: {eval_str}\n" if eval_str else ""
    move_section = f"## Move Played\n\nMove: {move_san}\n{eval_line}\n"
    if board_after_str:
        fen_after_line = f"FEN: {fen_after}\n" if fen_after else ""
        move_section += f"\nBoard after the move:\n{board_after_str}\n{fen_after_line}\n"

    facts_section = ""
    if facts:
        facts_section = "## Verified Move Facts\n\n" + "\n".join(f"- {f}" for f in facts) + "\n\n"

    task_section = (
        "## Task\n\n"
        "Think through the key continuations from this position, then output the "
        "5 most instructive engine lines using the <line> format. "
        "Each line must be at least 4 half-moves deep and explore a genuinely different idea."
    )

    return f"{position_section}{move_section}{facts_section}{task_section}"


# ---------------------------------------------------------------------------
# Board / move utilities
# ---------------------------------------------------------------------------


def board_ascii(board: chess.Board) -> str:
    """Return a labelled ASCII board (rank/file labels) for LLM context."""
    rows = str(board).split("\n")  # rank 8 at top
    lines = ["  a b c d e f g h"]
    for i, row in enumerate(rows):
        lines.append(f"{8 - i} {row}")
    lines.append(f"  ({'White' if board.turn == chess.WHITE else 'Black'} to move)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Joint task system prompt (Phase 2 SFT)
# ---------------------------------------------------------------------------

JOINT_SYSTEM_PROMPT = (
    "You are an expert chess coach analysing a student's game move by move.\n\n"
    "You will receive the board position, the student's move, and a set of engine key lines. "
    "Each move in the key lines is shown as a board embedding (representing that board state) "
    "followed by the SAN notation and SF15 classical eval term changes in brackets, e.g.:\n"
    "  Nd5 [mobility +0.32; king safety −0.15]\n"
    "The first line (PLAYED LINE) shows what happens when the student's move is played; "
    "the remaining lines are the engine's top alternatives.\n\n"
    "Your task:\n\n"
    "THINKING — analyse every line:\n"
    "In your thinking block, work through each line systematically:\n"
    "  POSITION: <phase> | <side> to move | eval: <label> (<N> cp)\n"
    "  CLASSIFICATION: <san> is <class> (<N> cp loss)\n"
    "  LINE analysis:\n"
    "    PLAYED: <moves> | eval: <label>\n"
    "      <san>: <term> <sign><val>, <term> <sign><val> → <one-line interpretation>\n"
    "      ...\n"
    "      net: <overall impact in one phrase>\n"
    "    LINE 1: <moves> | eval: <label>\n"
    "      <san>: <terms> → <interpretation>\n"
    "      net: <impact>\n"
    "    ... (one entry per line provided)\n"
    "  VERDICT: <san> <class> (<N> cp loss). <alt_san> was <description> (−<N> cp).\n"
    "  COACHING FOCUS: <one sentence on what to emphasise to the student>\n\n"
    "COACHING COMMENT — after your thinking:\n"
    "Write one paragraph (3-5 sentences) as a coaching comment directly to the student. The comment must:\n"
    "- Be grounded in the line analysis above — reference specific moves or ideas\n"
    "- Address the quality of the student's move relative to the engine alternatives\n"
    "- Explain the key strategic or tactical idea in this position\n"
    "- Be written in second person ('You play Nd5…') — never use first person\n"
    "- Not repeat raw numbers from the eval terms\n\n"
    "Output format:\n"
    "<think>\n"
    "[structured line analysis]\n"
    "</think>\n"
    "[coaching comment]"
)


def format_joint_user_prompt(
    board_ascii_str: str,
    fen: str,
    move_san: str,
    eval_str: str = "",
    facts: list[str] | None = None,
    board_after_str: str = "",
    fen_after: str = "",
    played_line: str | None = None,
    key_lines: list[str] | None = None,
) -> str:
    """Build the user message for the joint thinking + coaching task (Phase 2).

    played_line: annotated line string starting from the student's move, e.g.:
        "Ne5 [mobility +0.32; threats +0.18] → d4 [space +0.21] → ..."
    key_lines: list of annotated alternative line strings with the same format.
    Sentinels are injected later by _inject_move_tokens in the train script
    (one sentinel per SAN, positioned before the SAN + bracket annotation).
    """
    fen_line = f"FEN: {fen}\n" if fen else ""
    position_section = f"## Position\n\nBoard before the move:\n{board_ascii_str}\n{fen_line}\n"

    eval_line = f"Engine assessment: {eval_str}\n" if eval_str else ""
    move_section = f"## Move Played\n\nMove: {move_san}\n{eval_line}\n"
    if board_after_str:
        fen_after_line = f"FEN: {fen_after}\n" if fen_after else ""
        move_section += f"\nBoard after the move:\n{board_after_str}\n{fen_after_line}\n"

    facts_section = ""
    if facts:
        facts_section = "## Verified Move Facts\n\n" + "\n".join(f"- {f}" for f in facts) + "\n\n"

    lines_section = ""
    all_lines = []
    if played_line:
        all_lines.append(f"PLAYED LINE: {played_line}")
    if key_lines:
        for i, line in enumerate(key_lines):
            all_lines.append(f"Line {i + 1}: {line}")
    if all_lines:
        lines_section = "## Engine Key Lines\n\n" + "\n".join(all_lines) + "\n\n"

    task_section = (
        "## Task\n\n"
        "Analyse each engine key line using the SF15 eval term changes shown, "
        "then give a coaching comment to the student about their move."
    )

    return f"{position_section}{move_section}{facts_section}{lines_section}{task_section}"


# ---------------------------------------------------------------------------
# Textbook rewrite system prompt
# ---------------------------------------------------------------------------

TEXTBOOK_SYSTEM_PROMPT = (
    "You are an expert chess coach reformatting a human expert's game annotation "
    "into a standard coaching comment.\n\n"
    "Your ONLY task is to rewrite the provided expert annotation in clear coaching language. "
    "The expert annotation is ground truth — it comes from a qualified chess instructor "
    "and takes priority over engine classifications.\n\n"
    "You MUST rewrite to assume that the game is currently played by your student, "
    "even if the expert annotation says the game is played by named players. "
    "Do NOT mention the name of the original player like Akopian or Atkins.\n\n"
    "Rules:\n"
    "- PRESERVE every chess insight, claim, plan, and explanation from the expert annotation. "
    "Do not drop, weaken, or contradict any point the expert makes.\n"
    "- Do NOT introduce new chess claims or analysis beyond what the expert wrote or "
    "what Stockfish directly supports.\n"
    "- Rewrite in direct, concise coaching language (3-6 sentences).\n"
    "- If the expert's wording is already clear and well-structured, a light rewrite is fine; "
    "do not pad or dilute it.\n\n"
    "FILTERING RULE — respond with exactly the word SKIP (nothing else) when the annotation:\n"
    "- Contains no chess instruction (e.g. 'Forced.', 'Nice!', 'Good move.', '0-1')\n"
    "- Is purely a game-result or time-forfeit note\n"
    "- Is only a quiz/question with no answer or explanation\n"
    "- Is too vague to convey any chess idea (e.g. 'Black continues their plan.')\n\n"
    "OUTPUT FORMAT: End your response with your coaching comment in this exact format:\\n"
    "<comment>The knight leaps to its outpost on d5, eyeing the weak c7-pawn.</comment>\\n"
    "The text between those tags is the only part shown to the student. "
    "Do NOT include URLs, links, or references to external resources in the comment. "
    "For SKIP responses, end with exactly: <comment>SKIP</comment>"
)

# Few-shot examples inserted into the message list before the real query.
# Each tuple is (user_content, assistant_content).
TEXTBOOK_FEW_SHOT: list[tuple[str, str]] = [
    # --- Negative: pure result ---
    (
        "Move played: Rd1\nClassification: best\nEngine evaluation before move: +3.2\n\n"
        "Expert annotation to preserve and reformat:\n0-1 White resigns.\n\n"
        "Rewrite the expert annotation above into coaching style. "
        "Every insight from the expert must appear in your response. "
        "Do not omit or contradict any point.",
        "<comment>SKIP</comment>",
    ),
    # --- Negative: single-word / no insight ---
    (
        "Move played: Nf3\nClassification: best\nEngine evaluation before move: +0.1\n\n"
        "Expert annotation to preserve and reformat:\nForced.\n\n"
        "Rewrite the expert annotation above into coaching style. "
        "Every insight from the expert must appear in your response. "
        "Do not omit or contradict any point.",
        "<comment>SKIP</comment>",
    ),
    # --- Negative: quiz with no answer ---
    (
        "Move played: e4\nClassification: best\nEngine evaluation before move: +0.3\n\n"
        "Expert annotation to preserve and reformat:\n"
        "What would you play here? Find the best continuation.\n\n"
        "Rewrite the expert annotation above into coaching style. "
        "Every insight from the expert must appear in your response. "
        "Do not omit or contradict any point.",
        "<comment>SKIP</comment>",
    ),
    # --- Negative: vague with no chess idea ---
    (
        "Move played: Rb8\nClassification: good\nEngine evaluation before move: -0.5\n\n"
        "Expert annotation to preserve and reformat:\nBlack continues their plan.\n\n"
        "Rewrite the expert annotation above into coaching style. "
        "Every insight from the expert must appear in your response. "
        "Do not omit or contradict any point.",
        "<comment>SKIP</comment>",
    ),
    # --- Positive: genuine instructive annotation ---
    (
        "Move played: Nd5\nClassification: best\nEngine evaluation before move: +1.8\n\n"
        "Expert annotation to preserve and reformat:\n"
        "The knight leaps to its dream square. From d5 it cannot be challenged by a pawn, "
        "attacks the c7 weakness, and eyes the f6 entry — all at once. "
        "This is the outpost that the entire opening was played for.\n\n"
        "Rewrite the expert annotation above into coaching style. "
        "Every insight from the expert must appear in your response. "
        "Do not omit or contradict any point.",
        "<comment>The knight jumps to d5 — a permanent outpost that no enemy pawn can challenge. "
        "From this dominant square it simultaneously pressures the weak c7-pawn and "
        "threatens to penetrate via f6, combining attack and restriction in one move. "
        "The entire opening strategy has been building toward this moment.</comment>",
    ),
]


def format_textbook_prompt(
    board_ascii_str: str,
    san: str,
    classification: str,
    eval_str: str,
    expert_annotation: str,
    fen: str = "",
) -> str:
    """Build the user message for textbook annotation rewriting.

    The expert annotation is the primary content — the LLM's job is to reformat
    it faithfully, not to generate fresh analysis.
    """
    fen_line = f"FEN: {fen}\n" if fen else ""
    board_section = (
        f"Board position before the move:\n{board_ascii_str}\n{fen_line}\n"
        if board_ascii_str
        else ""
    )

    return (
        f"{board_section}"
        f"Move played: {san}\n"
        f"Classification: {classification}\n"
        f"Engine evaluation before move: {eval_str}\n"
        f"\nExpert annotation to preserve and reformat:\n{expert_annotation}\n"
        "\nRewrite the expert annotation above into coaching style. "
        "Every insight from the expert must appear in your response. "
        "Do not omit or contradict any point."
    )


# ---------------------------------------------------------------------------
# Prompt formatting — shared between training and inference
# ---------------------------------------------------------------------------


def _cct_inline(cct: dict[str, list[str]]) -> str:
    """Format a CCT dict as a compact inline string."""
    parts = []
    if cct.get("checks"):
        parts.append(f"Checks: {', '.join(cct['checks'])}")
    if cct.get("captures"):
        parts.append(f"Captures: {', '.join(cct['captures'])}")
    if cct.get("threats"):
        parts.append(f"Threats: {', '.join(cct['threats'])}")
    return " | ".join(parts) if parts else "none"


def _render_move_tree(nodes: list[TreeNode], played_san: str) -> str:
    """Render a 3-level move tree as a numbered, LLM-readable block.

    Format:
      1) Opponent: Nc6 [+0.28]
         Your options (CCT): Threats: Nxe4
         1a) You: d4 [+0.32]
             Opp options (CCT): Captures: dxe5
             Opp replies: exd4 [+0.20] · d6 [+0.28] · Bc5 [+0.25]
         1b) You: Bb5 [+0.31]
             ...
    """
    if not nodes:
        return ""

    _LETTERS = "abc"
    lines = [f"## Continuation Tree\n\nAfter {played_san}, opponent's top replies:\n"]

    for i, l1 in enumerate(nodes, start=1):
        lines.append(f"{i}) Opponent: {l1.move_san} [{l1.eval_str}]")
        cct1 = _cct_inline(l1.cct)
        if cct1 != "none":
            lines.append(f"   Your options (CCT): {cct1}")

        for j, l2 in enumerate(l1.children):
            letter = _LETTERS[j] if j < len(_LETTERS) else str(j + 1)
            lines.append(f"   {i}{letter}) You: {l2.move_san} [{l2.eval_str}]")
            cct2 = _cct_inline(l2.cct)
            if cct2 != "none":
                lines.append(f"       Opp options (CCT): {cct2}")
            if l2.children:
                l3_str = " · ".join(f"{l3.move_san} [{l3.eval_str}]" for l3 in l2.children)
                lines.append(f"       Opp replies: {l3_str}")

        if i < len(nodes):
            lines.append("")  # blank line between L1 entries

    return "\n".join(lines)


def format_position_context(ctx: dict[str, Any]) -> str:
    """Format the position context dict as a readable ## section for the LLM prompt."""
    if not ctx:
        return ""

    lines = ["## Position Context\n"]

    # Game phase & material
    phase = ctx.get("game_phase", "")
    move_num = ctx.get("move_number", "")
    mat_w = ctx.get("material_white", 0)
    mat_b = ctx.get("material_black", 0)
    balance = ctx.get("material_balance", 0)
    bal_str = f"+{balance}" if balance > 0 else str(balance)
    lines.append(
        f"Phase: {phase}  |  Move: {move_num}  |  Material — White: {mat_w}  Black: {mat_b}  Balance: {bal_str}"
    )

    # Pawn structure
    def _pawn_facts(passed: list[str], doubled: list[str], isolated: list[str]) -> str:
        parts = []
        if passed:
            parts.append(f"passed on {', '.join(passed)}")
        if doubled:
            files_str = ", ".join(f"{f}-file" for f in doubled)
            parts.append(f"doubled on {files_str}")
        if isolated:
            parts.append(f"isolated on {', '.join(isolated)}")
        return ", ".join(parts) if parts else "none"

    pw = _pawn_facts(
        ctx.get("passed_pawns_white", []),
        ctx.get("doubled_files_white", []),
        ctx.get("isolated_pawns_white", []),
    )
    pb = _pawn_facts(
        ctx.get("passed_pawns_black", []),
        ctx.get("doubled_files_black", []),
        ctx.get("isolated_pawns_black", []),
    )
    lines.append(f"Pawn structure — White: {pw}  |  Black: {pb}")

    # Open files
    open_files = ctx.get("open_files", [])
    hw = ctx.get("half_open_white", [])
    hb = ctx.get("half_open_black", [])
    file_parts = []
    if open_files:
        file_parts.append(f"Open: {', '.join(open_files)}")
    if hw:
        file_parts.append(f"Half-open (White): {', '.join(hw)}")
    if hb:
        file_parts.append(f"Half-open (Black): {', '.join(hb)}")
    if file_parts:
        lines.append("Files — " + "  |  ".join(file_parts))

    # King safety
    ks_w = ctx.get("king_shield_white", 0)
    ks_b = ctx.get("king_shield_black", 0)
    on_w = ctx.get("open_near_king_white", [])
    on_b = ctx.get("open_near_king_black", [])
    w_king = f"{ks_w} shield pawn{'s' if ks_w != 1 else ''}"
    if on_w:
        w_king += f", open file{'s' if len(on_w) > 1 else ''} {', '.join(on_w)} near king"
    b_king = f"{ks_b} shield pawn{'s' if ks_b != 1 else ''}"
    if on_b:
        b_king += f", open file{'s' if len(on_b) > 1 else ''} {', '.join(on_b)} near king"
    lines.append(f"King safety — White: {w_king}  |  Black: {b_king}")

    # Piece mobility
    mob_w = ctx.get("mobility_white", 0)
    mob_b = ctx.get("mobility_black", 0)
    lines.append(f"Piece mobility — White: {mob_w}  |  Black: {mob_b}")

    return "\n".join(lines) + "\n"


def move_facts(board: chess.Board, move: chess.Move) -> list[str]:
    """Return a list of concise, verified facts about a move.

    These are deterministic, engine-free observations (captures, checks,
    castling, promotion, etc.) that ground the LLM prompt and prevent
    hallucination about basic move properties.

    Args:
        board: Position *before* the move is played.
        move:  The move to describe.

    Returns:
        List of short fact strings, e.g. ["captures pawn on e5", "gives check"].
    """
    facts: list[str] = []

    piece = board.piece_at(move.from_square)
    if piece is None:
        return facts

    piece_name = chess.piece_name(piece.piece_type)

    # Capture
    captured = board.piece_at(move.to_square)
    if board.is_en_passant(move):
        ep_square = chess.square_name(move.to_square)
        facts.append(f"captures pawn en passant on {ep_square}")
    elif captured:
        captured_name = chess.piece_name(captured.piece_type)
        to_sq = chess.square_name(move.to_square)
        facts.append(f"captures {captured_name} on {to_sq}")

    # Castling
    if board.is_castling(move):
        if board.is_kingside_castling(move):
            facts.append("castles kingside")
        else:
            facts.append("castles queenside")

    # Promotion
    if move.promotion:
        promo_name = chess.piece_name(move.promotion)
        facts.append(f"promotes pawn to {promo_name}")

    # Check / checkmate — apply the move and test
    board_copy = board.copy()
    board_copy.push(move)
    if board_copy.is_checkmate():
        facts.append("delivers checkmate")
    elif board_copy.is_check():
        facts.append("gives check")

    # Piece movement description (if not a simple pawn push or capture already described)
    to_sq = chess.square_name(move.to_square)
    from_sq = chess.square_name(move.from_square)
    if piece.piece_type != chess.PAWN and not board.is_castling(move):
        if captured:
            pass  # already described above
        else:
            facts.append(f"moves {piece_name} from {from_sq} to {to_sq}")
    elif piece.piece_type == chess.PAWN and not captured and not board.is_en_passant(move):
        # Pawn push — note if it's a two-square advance
        rank_diff = abs(chess.square_rank(move.to_square) - chess.square_rank(move.from_square))
        if rank_diff == 2:
            facts.append(f"advances pawn two squares to {to_sq}")
        else:
            facts.append(f"advances pawn to {to_sq}")

    return facts


def format_user_prompt(
    board_ascii_str: str,
    san: str,
    classification: str,
    eval_str: str,
    best_move: str = "",
    cp_loss: int = 0,
    candidates: list[str] | None = None,
    opponent_threats: list[str] | None = None,
    facts: list[str] | None = None,
    fen: str = "",
    cct: dict[str, list[str]] | None = None,
    move_tree: list[TreeNode] | None = None,
    position_context: dict[str, Any] | None = None,
) -> str:
    """Build the user message for the chess coaching prompt.

    Format matches the training data exactly — section headers, field labels,
    and ordering must not change without retraining.
    """
    # --- Section 1: Position ---
    position_section = ""
    if board_ascii_str:
        fen_line = f"FEN: {fen}\n" if fen else ""
        position_section = (
            f"## Position\n\nBoard before your move:\n{board_ascii_str}\n{fen_line}\n"
        )

    # --- Section 2: Position Context ---
    ctx_section = ""
    if position_context:
        ctx_section = format_position_context(position_context) + "\n"

    # --- Section 3: Move Played ---
    move_header = (
        f"## Move Played\n\nMove: {san}  |  Classification: {classification}  |  Eval: {eval_str}\n"
    )
    best_line = f"Engine best was: {best_move} (\u2212{cp_loss} cp)\n" if cp_loss > 0 else ""
    candidates_line = f"Engine top candidates: {', '.join(candidates)}\n" if candidates else ""
    threats_line = (
        f"Opponent threats if you passed: {', '.join(opponent_threats)}\n"
        if opponent_threats
        else ""
    )
    cct_line = ""
    if cct:
        cct_str = _cct_inline(cct)
        if cct_str != "none":
            cct_line = f"Your tactical options (CCT): {cct_str}\n"

    move_section = move_header + best_line + candidates_line + threats_line + cct_line

    # --- Section 4: Verified Move Facts ---
    facts_section = ""
    if facts:
        facts_section = "## Verified Move Facts\n\n" + "\n".join(f"- {f}" for f in facts) + "\n\n"

    # --- Section 5: Continuation Tree ---
    tree_section = ""
    if move_tree:
        tree_section = _render_move_tree(move_tree, san) + "\n\n"

    # --- Section 6: Coaching Task ---
    task_section = (
        "## Coaching Task\n\n"
        "Explain the chess idea behind this move in 4-6 sentences. "
        "Cover the immediate tactical or positional purpose, the strategic plan it supports, "
        "and any relevant opening theory, piece activity, pawn structure, or king safety considerations. "
        "If verified move facts are listed, do not contradict them."
    )

    return (
        f"{position_section}{ctx_section}{move_section}\n"
        f"{facts_section}{tree_section}{task_section}"
    )
