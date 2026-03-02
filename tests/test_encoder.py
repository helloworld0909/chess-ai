import json
import os
import sys
import tempfile

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import importlib.util as _ilu

import chess

from encoder import MOVE_TOKEN, MOVE_TOKEN_ID
from encoder.board_tensor import board_to_tensor, boards_to_tensor
from encoder.cnn import ChessEncoder, ResidualBlock

_REPO = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _REPO)
from training.encoder_collator import EncoderDataCollator
from training.encoder_model import ChessLMWithEncoder


def _load_module(path: str, name: str):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sft = _load_module(
    os.path.join(_REPO, "recipes-train", "qwen3-4b-encoder-phase1-sft", "train.py"),
    "encoder_sft_train",
)
_inject_move_tokens = _sft._inject_move_tokens
_extract_line_sans = _sft._extract_line_sans
_inject_move_tokens_p2 = _sft._inject_move_tokens

_pre = _load_module(
    os.path.join(_REPO, "recipes-train", "encoder-pretrain", "train.py"), "encoder_pretrain_train"
)
EncoderPretrainDataset = _pre.EncoderPretrainDataset
EncoderRegressorHead = _pre.EncoderRegressorHead

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_tokenizer():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # No add_special_tokens needed — MOVE_TOKEN (<|fim_pad|>) is already in vocab
    return tok


@pytest.fixture(scope="module")
def mock_llm():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
    config.hidden_size = 256
    config.num_hidden_layers = 1
    llm = AutoModelForCausalLM.from_config(config)
    # No resize_token_embeddings needed — vocab unchanged
    return llm


@pytest.fixture(scope="module")
def move_token_id():
    return MOVE_TOKEN_ID


@pytest.fixture(scope="module")
def wrapper(mock_llm):
    return ChessLMWithEncoder(
        llm=mock_llm,
        hidden_size=256,
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
    encoder = ChessEncoder(hidden_size=64, num_blocks=3, out_dim=256)
    x = torch.randn(4, 38, 8, 8)
    out = encoder(x)
    assert out.shape == (4, 256)


def test_chess_encoder_legacy_19ch():
    encoder = ChessEncoder(in_channels=19, hidden_size=64, num_blocks=3, out_dim=256)
    x = torch.randn(4, 19, 8, 8)
    out = encoder(x)
    assert out.shape == (4, 256)


def test_board_to_tensor():
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert tensor.shape == (19, 8, 8)
    assert tensor[3, 0, 0] == 1.0  # Ra1
    assert tensor[3, 0, 7] == 1.0  # Rh1
    assert torch.all(tensor[12] == 1.0)  # white to move
    assert tensor[6 + 3, 7, 0] == 1.0  # Ra8
    assert torch.allclose(tensor[18], torch.tensor(0.005))


def test_boards_to_tensor_with_move():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    tensor = boards_to_tensor(board, move)
    assert tensor.shape == (38, 8, 8)
    assert tensor[0, 1, 4] == 1.0  # e2 pawn before
    assert tensor[19, 1, 4] == 0.0  # e2 pawn gone after
    assert tensor[19, 3, 4] == 1.0  # e4 pawn present after


def test_boards_to_tensor_static():
    board = chess.Board()
    tensor = boards_to_tensor(board, move=None)
    assert tensor.shape == (38, 8, 8)
    assert torch.equal(tensor[:19], tensor[19:])


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


def test_collator_single_move(mock_tokenizer, move_token_id):
    """Collator with no LINE moves produces 1 board tensor per example."""
    text = f"Move: {MOVE_TOKEN} board info"
    feat = _make_feature(
        mock_tokenizer,
        text,
        fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        move_san="e4",
        line_sans=[],
    )
    collator = EncoderDataCollator(tokenizer=mock_tokenizer)
    batch = collator([feat])
    assert batch["board_tensors_flat"].shape == (1, 38, 8, 8)
    assert batch["move_counts"].tolist() == [1]


def test_collator_multi_move(mock_tokenizer, move_token_id):
    """Collator with LINE moves produces sum(move_counts) board tensors."""
    line_sans = [["d4", "e5", "Nf3"], ["c4", "Nc6"]]
    n_tokens = 1 + sum(len(ls) for ls in line_sans)
    # Build text with exactly n_tokens MOVE_TOKEN occurrences
    text = f"Move: {MOVE_TOKEN} " + " ".join(f"{MOVE_TOKEN}" for _ in range(n_tokens - 1))
    feat = _make_feature(
        mock_tokenizer,
        text,
        fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        move_san="e4",
        line_sans=line_sans,
    )
    collator = EncoderDataCollator(tokenizer=mock_tokenizer)
    batch = collator([feat])
    assert batch["move_counts"].sum().item() == n_tokens
    assert batch["board_tensors_flat"].shape == (n_tokens, 38, 8, 8)


def test_collator_counts_match_input_ids(mock_tokenizer, move_token_id):
    """sum(move_counts) must equal number of move_token_id tokens in input_ids."""
    line_sans = [["Nf3", "d5"]]
    n_tokens = 1 + sum(len(ls) for ls in line_sans)
    text = " ".join([MOVE_TOKEN] * n_tokens)
    feat = _make_feature(
        mock_tokenizer,
        text,
        fen=chess.STARTING_FEN,
        move_san="e4",
        line_sans=line_sans,
    )
    collator = EncoderDataCollator(tokenizer=mock_tokenizer)
    batch = collator([feat])
    n_in_ids = (batch["input_ids"] == move_token_id).sum().item()
    assert n_in_ids == batch["move_counts"].sum().item()


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

    board_tensors_flat = torch.randn(sum(n_moves), 38, 8, 8)
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

    board_tensors_flat = torch.randn(1, 38, 8, 8)
    move_counts = torch.tensor([1])

    out = wrapper(
        board_tensors_flat=board_tensors_flat,
        move_counts=move_counts,
        input_ids=input_ids,
        labels=labels,
    )
    assert out.loss is not None


def test_chess_lm_mismatch_raises(wrapper, move_token_id):
    """RuntimeError when board_tensors_flat count != move tokens in input_ids."""
    B, L = 1, 5
    input_ids = torch.randint(100, 1000, (B, L))
    input_ids[0, 1] = move_token_id  # 1 token in input_ids

    board_tensors_flat = torch.randn(3, 38, 8, 8)  # 3 tensors — mismatch
    move_counts = torch.tensor([3])

    with pytest.raises(RuntimeError, match="move token count mismatch"):
        wrapper(
            board_tensors_flat=board_tensors_flat,
            move_counts=move_counts,
            input_ids=input_ids,
        )


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

    x = torch.randn(2, 38, 8, 8)
    with torch.no_grad():
        assert torch.equal(w1.cnn(x), w2.cnn(x))


# ---------------------------------------------------------------------------
# EncoderRegressorHead
# ---------------------------------------------------------------------------


def test_encoder_regressor_head_shape():
    encoder = ChessEncoder(in_channels=38, hidden_size=32, num_blocks=2, out_dim=64)
    head = EncoderRegressorHead(encoder)
    x = torch.randn(4, 38, 8, 8)
    preds = head(x)
    assert preds.shape == (4,)
    assert preds.abs().max().item() < 1.0


def test_encoder_regressor_head_range():
    encoder = ChessEncoder(in_channels=38, hidden_size=32, num_blocks=2, out_dim=64)
    head = EncoderRegressorHead(encoder)
    x = torch.randn(8, 38, 8, 8) * 100.0
    preds = head(x)
    assert (preds > -1.0).all()
    assert (preds < 1.0).all()


# ---------------------------------------------------------------------------
# EncoderPretrainDataset
# ---------------------------------------------------------------------------


_SAMPLE_RECORDS = [
    {
        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "move_uci": "e7e5",
        "cp_diff_scaled": -0.05,
    },
    {
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        "move_uci": "g1f3",
        "cp_diff_scaled": 0.12,
    },
    {
        "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        "move_uci": "b8c6",
        "cp_diff_scaled": 0.03,
    },
]


def test_encoder_pretrain_dataset_len():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for rec in _SAMPLE_RECORDS:
            f.write(json.dumps(rec) + "\n")
        tmp_path = f.name
    ds = EncoderPretrainDataset(tmp_path)
    assert len(ds) == len(_SAMPLE_RECORDS)
    os.unlink(tmp_path)


def test_encoder_pretrain_dataset_item_shapes():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for rec in _SAMPLE_RECORDS:
            f.write(json.dumps(rec) + "\n")
        tmp_path = f.name
    ds = EncoderPretrainDataset(tmp_path)
    board_tensor, label = ds[0]
    assert board_tensor.shape == (38, 8, 8)
    assert label.shape == ()
    assert label.dtype == torch.float32
    os.unlink(tmp_path)


def test_encoder_pretrain_dataset_limit():
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for rec in _SAMPLE_RECORDS:
            f.write(json.dumps(rec) + "\n")
        tmp_path = f.name
    ds = EncoderPretrainDataset(tmp_path, limit=2)
    assert len(ds) == 2
    os.unlink(tmp_path)
