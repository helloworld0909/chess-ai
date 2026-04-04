import importlib.util as _ilu
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import chess

from encoder import BOARD_TOKENS_PER_POSITION, MOVE_TOKEN, MOVE_TOKEN_ID
from encoder.board_tensor import board_to_tensor
from encoder.cnn import ChessEncoder, ResidualBlock

_REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _REPO)
from src.model.encoder_collator import EncoderDataCollator
from src.model.encoder_model import ChessLMWithEncoder


def _load_module(path: str, name: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sft = _load_module(
    os.path.join(_REPO, "recipes-train", "qwen3.5-4b-encoder-phase2-grpo", "train.py"),
    "encoder_sft_train",
)
_inject_move_tokens = _sft._inject_move_tokens
_extract_line_sans = _sft._extract_line_sans
_inject_move_tokens_p2 = _sft._inject_move_tokens

# ---------------------------------------------------------------------------
# Minimal fake tokenizer — no network calls, no model weights
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 250000  # large enough to include MOVE_TOKEN_ID (248055 for Qwen3.5-4B)
_PAD_ID = 0
_EOS_ID = 1


class _FakeTokenizer:
    """Minimal tokenizer for encoder tests.

    Tokenises by splitting on whitespace, mapping each token to a stable
    integer via hash.  The MOVE_TOKEN sentinel gets its correct ID (151654).
    The collator only needs: __call__, pad(), apply_chat_template().
    """

    pad_token_id = _PAD_ID
    eos_token_id = _EOS_ID
    pad_token = "[PAD]"
    eos_token = "[EOS]"
    model_input_names = ["input_ids", "attention_mask"]

    # ------------------------------------------------------------------
    def _tok_id(self, word: str) -> int:
        if word == MOVE_TOKEN:
            return MOVE_TOKEN_ID
        return max(2, abs(hash(word)) % (_VOCAB_SIZE - 2))

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors=None,
        **kwargs,
    ) -> Dict[str, Any]:
        if isinstance(text, list):
            results = [self._encode_single(t) for t in text]
            ids = [r["input_ids"] for r in results]
            masks = [r["attention_mask"] for r in results]
            if return_tensors == "pt":
                max_len = max(len(i) for i in ids)
                ids_padded = [i + [_PAD_ID] * (max_len - len(i)) for i in ids]
                masks_padded = [m + [0] * (max_len - len(m)) for m in masks]
                return {
                    "input_ids": torch.tensor(ids_padded),
                    "attention_mask": torch.tensor(masks_padded),
                }
            return {"input_ids": ids, "attention_mask": masks}
        return self._encode_single(text)

    def _encode_single(self, text: str) -> Dict[str, List[int]]:
        words = text.split()
        ids = [self._tok_id(w) for w in words] if words else [_EOS_ID]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(
        self,
        features: List[Dict[str, Any]],
        padding: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        ids_list = [f["input_ids"] for f in features]
        masks_list = [f["attention_mask"] for f in features]
        target_len = max_length or max(len(i) for i in ids_list)
        padded_ids = [i + [_PAD_ID] * (target_len - len(i)) for i in ids_list]
        padded_masks = [m + [0] * (target_len - len(m)) for m in masks_list]
        result = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
        }
        return result

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> str:
        """Concatenate role+content pairs into a single string."""
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>{content}<|end|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return "".join(parts)


# ---------------------------------------------------------------------------
# Minimal fake LLM — pure nn.Module, no pretrained weights
# ---------------------------------------------------------------------------


class _FakeEmbedding(nn.Embedding):
    pass


class _FakeLLM(nn.Module):
    """Tiny causal LM stub with the interface ChessLMWithEncoder expects."""

    def __init__(self, hidden_size: int = 256, vocab_size: int = _VOCAB_SIZE):
        super().__init__()
        self.config = type("cfg", (), {"hidden_size": hidden_size})()
        self._embed = _FakeEmbedding(vocab_size, hidden_size)
        # Expose the same attribute name as Qwen models
        self.model = type("model_stub", (), {"embed_tokens": self._embed})()
        # A single linear head (B, L, H) -> (B, L, V)
        self._lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self._embed

    def forward(self, inputs_embeds=None, input_ids=None, attention_mask=None, labels=None, **kw):
        if inputs_embeds is None:
            inputs_embeds = self._embed(input_ids)
        logits = self._lm_head(inputs_embeds)
        loss = None
        if labels is not None:
            # Simple cross-entropy on last dim
            shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            mask = shift_labels != -100
            if mask.any():
                loss = nn.functional.cross_entropy(shift_logits[mask], shift_labels[mask])
            else:
                loss = torch.tensor(0.0, requires_grad=True)
        return type("Out", (), {"logits": logits, "loss": loss})()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_tokenizer():
    return _FakeTokenizer()


@pytest.fixture(scope="module")
def mock_llm():
    return _FakeLLM(hidden_size=256)


@pytest.fixture(scope="module")
def move_token_id():
    return MOVE_TOKEN_ID


@pytest.fixture(scope="module")
def wrapper(mock_llm):
    return ChessLMWithEncoder(
        llm=mock_llm,
        hidden_size=256,
        cnn_in_channels=19,
        cnn_hidden_size=32,
        cnn_num_blocks=2,
        move_token_id=MOVE_TOKEN_ID,
    )


# ---------------------------------------------------------------------------
# CNN / board tensor tests (unchanged)
# ---------------------------------------------------------------------------


def test_residual_block():
    block = ResidualBlock(channels=64)
    x = torch.randn(2, 64, 8, 8)
    out = block(x)
    assert out.shape == (2, 64, 8, 8)


def test_chess_encoder_shape():
    encoder = ChessEncoder(in_channels=19, hidden_size=64, num_blocks=3, out_dim=256)
    x = torch.randn(4, 19, 8, 8)
    out = encoder(x)
    assert out.shape == (4, 65, 256)  # 64 per-square tokens + 1 global token

    grid, glob = encoder.forward_clip(x)
    assert grid.shape == (4, 64, 256)
    assert glob.shape == (4, 1, 256)


def test_chess_encoder_legacy_19ch():
    encoder = ChessEncoder(in_channels=19, hidden_size=64, num_blocks=3, out_dim=256)
    x = torch.randn(4, 19, 8, 8)
    out = encoder(x)
    assert out.shape == (4, 65, 256)  # 64 per-square tokens + 1 global token


def test_board_to_tensor():
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert tensor.shape == (19, 8, 8)
    assert tensor[3, 0, 0] == 1.0  # Ra1
    assert tensor[3, 0, 7] == 1.0  # Rh1
    assert torch.all(tensor[12] == 1.0)  # white to move
    assert tensor[6 + 3, 7, 0] == 1.0  # Ra8
    assert torch.allclose(tensor[18], torch.tensor(0.005))


# ---------------------------------------------------------------------------
# _extract_line_sans (phase 2) / _inject_move_tokens
# ---------------------------------------------------------------------------


def test_extract_line_sans_phase2_basic():
    """_extract_line_sans parses ## Engine Key Lines section in user prompt."""
    content = (
        "## Engine Key Lines\n\n"
        "Line 1: Nd5 → Kh7 → Be3\n"
        "Line 2: Be3 → Ne6\n\n"
        "## Task\n\nAnnotate lines."
    )
    result = _extract_line_sans(content)
    assert result == [["Nd5", "Kh7", "Be3"], ["Be3", "Ne6"]]


def test_extract_line_sans_phase2_no_section():
    """Returns [] when no ## Engine Key Lines section present (phase 1 format)."""
    content = "Move: Ne5\nBoard: ...\n## Task\nDo something."
    result = _extract_line_sans(content)
    assert result == []


def test_inject_move_tokens_user_message_phase1():
    """Phase 1: only student move in user turn gets a sentinel; assistant untouched."""
    msgs = [
        {"role": "user", "content": "Move: Ne5\nsome other text Ne5"},
        {
            "role": "assistant",
            "content": "<line>LINE 1: Nd5 (knight) → Kh7 (king) | eval: equal</line>",
        },
    ]
    new_msgs, line_sans = _inject_move_tokens(msgs, "Ne5")
    # User turn: 'Move: SAN' replaced, other occurrences untouched
    assert f"Move: {MOVE_TOKEN}" in new_msgs[0]["content"]
    assert "some other text Ne5" in new_msgs[0]["content"]
    # Assistant turn: NOT modified — pure text output
    assert f"LINE 1: Nd5" in new_msgs[1]["content"]
    assert MOVE_TOKEN not in new_msgs[1]["content"]
    # line_sans is always [] for phase 1
    assert line_sans == []


def test_inject_move_tokens_phase1_count(mock_tokenizer):
    """Phase 1: exactly 1 MOVE_TOKEN total (only student move)."""
    msgs = [
        {"role": "system", "content": "You are a coach."},
        {"role": "user", "content": "Move: e4\nBoard: ..."},
        {
            "role": "assistant",
            "content": (
                "<line>LINE 1: d4 (pawn) → e5 (pawn) → Nf3 (knight) | eval: equal</line>"
                "<line>LINE 2: c4 (pawn) → Nc6 (knight) | eval: equal</line>"
            ),
        },
    ]
    new_msgs, line_sans = _inject_move_tokens(msgs, "e4")
    text = mock_tokenizer.apply_chat_template(new_msgs, tokenize=False, add_generation_prompt=False)
    # Phase 1: only the student move sentinel, no line sentinels
    assert text.count(MOVE_TOKEN) == 1
    assert line_sans == []


def test_inject_move_tokens_phase2_key_lines(mock_tokenizer):
    """Phase 2: sentinels injected for student move + all key line moves."""
    user_content = (
        "Move: e4\nBoard: ...\n\n"
        "## Engine Key Lines\n\n"
        "Line 1: d4 → e5 → Nf3\n"
        "Line 2: c4 → Nc6\n\n"
        "## Task\n\nAnnotate."
    )
    msgs = [
        {"role": "system", "content": "You are a coach."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "<line>LINE 1: d4 (pawn) | eval: equal</line>"},
    ]
    new_msgs, line_sans = _inject_move_tokens_p2(msgs, "e4")
    text = mock_tokenizer.apply_chat_template(new_msgs, tokenize=False, add_generation_prompt=False)
    # 1 student + 3 (line 1) + 2 (line 2) = 6
    expected = 1 + sum(len(ls) for ls in line_sans)
    assert expected == 6
    assert text.count(MOVE_TOKEN) == 6
    # Assistant untouched
    assert MOVE_TOKEN not in new_msgs[2]["content"]


def test_extract_line_sans_with_played_line():
    """_extract_line_sans includes PLAYED LINE as the first entry."""
    content = (
        "## Engine Key Lines\n\n"
        "PLAYED LINE: Ne5 → d4 → Nc6\n"
        "Line 1: Nd5 → e4\n"
        "Line 2: Nf6 → g5\n\n"
        "## Task\n\nAnnotate."
    )
    result = _extract_line_sans(content)
    assert result == [["Ne5", "d4", "Nc6"], ["Nd5", "e4"], ["Nf6", "g5"]]


def test_inject_move_tokens_played_line(mock_tokenizer):
    """PLAYED LINE: sentinels are injected for all moves in the played line."""
    user_content = (
        "Move: Ne5\nBoard: ...\n\n"
        "## Engine Key Lines\n\n"
        "PLAYED LINE: Ne5 → d4 → Nc6\n"
        "Line 1: Nd5 → e4\n\n"
        "## Task\n\nAnnotate."
    )
    msgs = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": "<line>LINE 1: Nd5 (knight) | eval: equal</line>"},
    ]
    new_msgs, line_sans = _inject_move_tokens(msgs, "Ne5")
    text = mock_tokenizer.apply_chat_template(new_msgs, tokenize=False, add_generation_prompt=False)
    # 1 (student "Move:") + 3 (PLAYED LINE) + 2 (Line 1) = 6
    assert line_sans == [["Ne5", "d4", "Nc6"], ["Nd5", "e4"]]
    expected_count = 1 + sum(len(ls) for ls in line_sans)
    assert text.count(MOVE_TOKEN) == expected_count


# ---------------------------------------------------------------------------
# EncoderDataCollator
# ---------------------------------------------------------------------------


def _make_feature(tokenizer, text: str, fen: str, move_san: str, line_sans: list) -> dict:
    """Tokenize text and attach metadata fields."""
    encoded = tokenizer(text, return_tensors=None)
    feat = dict(encoded)
    feat["fen"] = fen
    feat["move_san"] = move_san
    feat["line_sans_json"] = json.dumps(line_sans)
    return feat


_SENTINEL_BLOCK = " ".join([MOVE_TOKEN] * BOARD_TOKENS_PER_POSITION)  # 64 sentinels per position


def test_collator_single_move(mock_tokenizer, move_token_id):
    """Collator with no LINE moves produces 1 board tensor per example."""
    text = f"Move: {_SENTINEL_BLOCK} board info"
    feat = _make_feature(
        mock_tokenizer,
        text,
        fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        move_san="e4",
        line_sans=[],
    )
    collator = EncoderDataCollator(tokenizer=mock_tokenizer)
    batch = collator([feat])
    assert batch["board_tensors_flat"].shape == (1, 19, 8, 8)
    assert batch["move_counts"].tolist() == [1]


def test_collator_multi_move(mock_tokenizer, move_token_id):
    """Collator with LINE moves produces sum(move_counts) board tensors."""
    line_sans = [["d4", "e5", "Nf3"], ["c4", "Nc6"]]
    n_boards = 1 + sum(len(ls) for ls in line_sans)
    # Build text with exactly n_boards sentinel blocks (64 sentinels each)
    text = " ".join([_SENTINEL_BLOCK] * n_boards)
    feat = _make_feature(
        mock_tokenizer,
        text,
        fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        move_san="e4",
        line_sans=line_sans,
    )
    collator = EncoderDataCollator(tokenizer=mock_tokenizer)
    batch = collator([feat])
    assert batch["move_counts"].sum().item() == n_boards
    assert batch["board_tensors_flat"].shape == (n_boards, 19, 8, 8)


def test_collator_counts_match_input_ids(mock_tokenizer, move_token_id):
    """sum(move_counts) must equal number of sentinel groups (64 tokens each) in input_ids."""
    line_sans = [["Nf3", "d5"]]
    n_boards = 1 + sum(len(ls) for ls in line_sans)
    text = " ".join([_SENTINEL_BLOCK] * n_boards)
    feat = _make_feature(
        mock_tokenizer,
        text,
        fen=chess.STARTING_FEN,
        move_san="e4",
        line_sans=line_sans,
    )
    collator = EncoderDataCollator(tokenizer=mock_tokenizer)
    batch = collator([feat])
    n_sentinel_groups = (
        batch["input_ids"] == move_token_id
    ).sum().item() // BOARD_TOKENS_PER_POSITION
    assert n_sentinel_groups == batch["move_counts"].sum().item()


# ---------------------------------------------------------------------------
# ChessLMWithEncoder — forward pass
# ---------------------------------------------------------------------------


def test_chess_lm_forward_shape(wrapper, move_token_id):
    """Forward pass output shape is (B, L, vocab) — sequence length unchanged."""
    B, L = 2, 10
    n_moves = [2, 3]
    input_ids = torch.randint(100, 1000, (B, L))
    # Plant move tokens at known positions
    input_ids[0, 2] = move_token_id
    input_ids[0, 5] = move_token_id
    input_ids[1, 1] = move_token_id
    input_ids[1, 4] = move_token_id
    input_ids[1, 7] = move_token_id

    board_tensors_flat = torch.randn(sum(n_moves), 19, 8, 8)
    move_counts = torch.tensor(n_moves)

    out = wrapper(
        board_tensors_flat=board_tensors_flat,
        move_counts=move_counts,
        input_ids=input_ids,
        attention_mask=torch.ones(B, L),
    )
    assert hasattr(out, "logits")
    assert out.logits.shape[:2] == (B, L)


def test_chess_lm_labels_move_positions_masked(wrapper, move_token_id):
    """<|move|> positions in labels must be -100 after forward."""
    B, L = 1, 6
    input_ids = torch.randint(100, 1000, (B, L))
    input_ids[0, 2] = move_token_id
    labels = torch.randint(100, 1000, (B, L))

    board_tensors_flat = torch.randn(1, 19, 8, 8)
    move_counts = torch.tensor([1])

    out = wrapper(
        board_tensors_flat=board_tensors_flat,
        move_counts=move_counts,
        input_ids=input_ids,
        labels=labels,
    )
    assert out.loss is not None


def test_chess_lm_mismatch_pads(wrapper, move_token_id):
    """board_tensors_flat count mismatch is handled gracefully (pad/trim, no error)."""
    B, L = 1, 5
    input_ids = torch.randint(100, 1000, (B, L))
    # Plant one full sentinel block (64 tokens) — needs L >= 64; use a longer seq
    long_ids = torch.randint(100, 1000, (1, 70))
    long_ids[0, :64] = move_token_id  # 64 sentinels = 1 board group

    board_tensors_flat = torch.randn(3, 19, 8, 8)  # 3 tensors but only 1 group needed
    move_counts = torch.tensor([1])

    # Should not raise — model trims to n_boards
    out = wrapper(
        board_tensors_flat=board_tensors_flat,
        move_counts=move_counts,
        input_ids=long_ids,
        attention_mask=torch.ones(1, 70),
    )
    assert hasattr(out, "logits")


def test_chess_lm_load_pretrained_weights(mock_llm, tmp_path):
    """encoder_weights.pt can be saved and reloaded into a fresh ChessLMWithEncoder."""
    w1 = ChessLMWithEncoder(
        llm=mock_llm,
        hidden_size=256,
        cnn_hidden_size=32,
        cnn_num_blocks=2,
    )
    weights_path = tmp_path / "encoder_weights.pt"
    torch.save(w1.cnn.state_dict(), weights_path)

    w2 = ChessLMWithEncoder(
        llm=mock_llm,
        hidden_size=256,
        cnn_hidden_size=32,
        cnn_num_blocks=2,
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = w2.cnn.load_state_dict(state, strict=True)
    assert missing == []
    assert unexpected == []

    x = torch.randn(2, 19, 8, 8)
    with torch.no_grad():
        assert torch.equal(w1.cnn(x), w2.cnn(x))


# ---------------------------------------------------------------------------
# _pawn_structure_parts tests
# ---------------------------------------------------------------------------

import importlib.util as _ilu


def _load_clip_train():
    spec = _ilu.spec_from_file_location(
        "encoder_clip_train",
        str(Path(__file__).parent.parent / "recipes-train/encoder-phase0/train.py"),
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_clip = _load_clip_train()
_pawn_structure_parts = _clip._pawn_structure_parts


def _board(fen: str) -> chess.Board:
    return chess.Board(fen)


def test_pawn_structure_start_no_flags():
    # Starting position: no pawn qualifies as passed, isolated, or doubled.
    board = _board(chess.STARTING_FEN)
    for sq in chess.SquareSet(board.pieces(chess.PAWN, chess.WHITE)):
        parts = _pawn_structure_parts(board, sq, chess.WHITE)
        assert parts == [], f"e2-h2 pawn {chess.square_name(sq)} should have no flags, got {parts}"


def test_pawn_structure_passed_white():
    # White pawn on e5, no black pawns on d/e/f files ahead of rank 4.
    board = _board("8/8/8/4P3/8/8/8/8 w - - 0 1")
    parts = _pawn_structure_parts(board, chess.E5, chess.WHITE)
    assert "passed pawn" in parts


def test_pawn_structure_not_passed_blocked():
    # Black pawn on e7 directly ahead of white pawn on e5 → not passed.
    board = _board("8/4p3/8/4P3/8/8/8/8 w - - 0 1")
    parts = _pawn_structure_parts(board, chess.E5, chess.WHITE)
    assert "passed pawn" not in parts


def test_pawn_structure_not_passed_adjacent_file():
    # Black pawn on d6 (adjacent file, ahead) blocks white pawn on e5.
    board = _board("8/8/3p4/4P3/8/8/8/8 w - - 0 1")
    parts = _pawn_structure_parts(board, chess.E5, chess.WHITE)
    assert "passed pawn" not in parts


def test_pawn_structure_passed_black():
    # Black pawn on d4, no white pawns on c/d/e files ahead (rank < 3).
    board = _board("8/8/8/8/3p4/8/8/8 b - - 0 1")
    parts = _pawn_structure_parts(board, chess.D4, chess.BLACK)
    assert "passed pawn" in parts


def test_pawn_structure_isolated():
    # White pawn on h4, no white pawns on g-file → isolated.
    board = _board("8/8/8/8/7P/8/8/8 w - - 0 1")
    parts = _pawn_structure_parts(board, chess.H4, chess.WHITE)
    assert "isolated pawn" in parts


def test_pawn_structure_not_isolated_has_neighbor():
    # White pawns on e4 and f2 → e4 pawn not isolated (f-file neighbor).
    board = _board("8/8/8/8/4P3/8/5P2/8 w - - 0 1")
    parts = _pawn_structure_parts(board, chess.E4, chess.WHITE)
    assert "isolated pawn" not in parts


def test_pawn_structure_isolated_rook_pawn():
    # White pawn on a4, no white pawns on b-file → isolated (a-file has only one neighbor).
    board = _board("8/8/8/8/P7/8/8/8 w - - 0 1")
    parts = _pawn_structure_parts(board, chess.A4, chess.WHITE)
    assert "isolated pawn" in parts


def test_pawn_structure_doubled():
    # White pawns on e4 and e2 → both are doubled.
    board = _board("8/8/8/8/4P3/8/4P3/8 w - - 0 1")
    assert "doubled pawn" in _pawn_structure_parts(board, chess.E4, chess.WHITE)
    assert "doubled pawn" in _pawn_structure_parts(board, chess.E2, chess.WHITE)


def test_pawn_structure_not_doubled_solo():
    # Single white pawn on e4 → not doubled.
    board = _board("8/8/8/8/4P3/8/8/8 w - - 0 1")
    parts = _pawn_structure_parts(board, chess.E4, chess.WHITE)
    assert "doubled pawn" not in parts


def test_pawn_structure_passed_isolated_combined():
    # White pawn on a5, no black pawns on a/b files → passed AND isolated.
    board = _board("8/8/8/P7/8/8/8/8 w - - 0 1")
    parts = _pawn_structure_parts(board, chess.A5, chess.WHITE)
    assert "passed pawn" in parts
    assert "isolated pawn" in parts
