#!/usr/bin/env python3
"""Probe encoder checkpoint token vectors against Qwen input embeddings.

Prints an ASCII board plus nearest-neighbor vocab tokens for one opening,
one middlegame, and one endgame position.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import chess
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.encoder.board_tensor import board_to_tensor
from src.encoder.cnn import ChessEncoder
from training.lib import load_config

DEFAULT_OPENING = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
DEFAULT_MIDDLEGAME = "r2q1rk1/pp2bppp/2npbn2/2p1p3/2P1P3/2NPBN1P/PPQ2PP1/R3KB1R w KQ - 0 10"
DEFAULT_ENDGAME = "8/5pk1/3p2p1/3P4/4PK2/6P1/7P/6R1 w - - 0 1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="recipes-train/encoder-clip/config.yaml",
        help="Training config path.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path, for example checkpoints/encoder-clip/checkpoint-600/checkpoint.pt",
    )
    parser.add_argument("--opening", default=DEFAULT_OPENING, help="Opening FEN.")
    parser.add_argument("--middlegame", default=DEFAULT_MIDDLEGAME, help="Middlegame FEN.")
    parser.add_argument("--endgame", default=DEFAULT_ENDGAME, help="Endgame FEN.")
    parser.add_argument(
        "--neighbors",
        type=int,
        default=6,
        help="Number of nearest vocab neighbors to show per square.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for encoder/Qwen model.",
    )
    return parser.parse_args()


def build_encoder(enc_cfg: dict, checkpoint_path: str, device: torch.device) -> ChessEncoder:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = ChessEncoder(
        in_channels=enc_cfg.get("in_channels", 19),
        hidden_size=enc_cfg.get("hidden_size", 256),
        num_blocks=enc_cfg.get("num_blocks", 10),
        out_dim=enc_cfg.get("out_dim", 2560),
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder.to(device)
    encoder.eval()
    return encoder


def build_qwen(model_name: str, device: torch.device) -> tuple[AutoTokenizer, AutoModel, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if device.type == "cuda" else "eager",
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    emb = F.normalize(model.get_input_embeddings().weight.detach().float(), dim=-1)
    return tokenizer, model, emb


def board_ascii(board: chess.Board) -> str:
    rows = []
    for rank in range(7, -1, -1):
        cells = []
        for file_idx in range(8):
            piece = board.piece_at(chess.square(file_idx, rank))
            cells.append(piece.symbol() if piece else ".")
        rows.append(f"{rank + 1}  {' '.join(cells)}")
    rows.append("   a b c d e f g h")
    return "\n".join(rows)


def piece_desc(board: chess.Board, sq: int) -> str:
    piece = board.piece_at(sq)
    if piece is None:
        return "empty"
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color} {chess.piece_name(piece.piece_type)}"


def top_neighbors(
    vec: torch.Tensor,
    *,
    vocab_emb: torch.Tensor,
    tokenizer,
    k: int,
) -> list[str]:
    sims = vocab_emb @ F.normalize(vec.float().cpu(), dim=-1)
    vals, ids = torch.topk(sims, k=min(k * 4, sims.numel()))
    out: list[str] = []
    for score, token_id in zip(vals.tolist(), ids.tolist()):
        text = tokenizer.decode([token_id]).replace("\n", "\\n")
        if not text.strip():
            continue
        out.append(f"{text!r} ({score:.3f})")
        if len(out) >= k:
            break
    return out


def probe_position(
    *,
    name: str,
    fen: str,
    encoder: ChessEncoder,
    tokenizer,
    vocab_emb: torch.Tensor,
    device: torch.device,
    neighbors: int,
) -> None:
    board = chess.Board(fen)
    with torch.no_grad():
        tokens = encoder(board_to_tensor(board).unsqueeze(0).to(device))[0].cpu()

    print(f"=== {name} ===")
    print(f"FEN: {fen}")
    print(board_ascii(board))
    print()
    for sq in chess.SQUARES:
        label = chess.square_name(sq)
        near = ", ".join(top_neighbors(tokens[sq], vocab_emb=vocab_emb, tokenizer=tokenizer, k=neighbors))
        print(f"{label:>2} | {piece_desc(board, sq):<13} | {near}")
    global_near = ", ".join(
        top_neighbors(tokens[64], vocab_emb=vocab_emb, tokenizer=tokenizer, k=max(neighbors, 8))
    )
    print(f"global | {global_near}")
    print()


def main() -> None:
    logging.getLogger().setLevel(logging.ERROR)
    hf_logging.set_verbosity_error()
    args = parse_args()

    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    encoder = build_encoder(config["encoder"], str(checkpoint_path), device)
    tokenizer, _, vocab_emb = build_qwen(config["clip"]["qwen_model"], device)

    positions = [
        ("Opening", args.opening),
        ("Middlegame", args.middlegame),
        ("Endgame", args.endgame),
    ]
    for name, fen in positions:
        probe_position(
            name=name,
            fen=fen,
            encoder=encoder,
            tokenizer=tokenizer,
            vocab_emb=vocab_emb,
            device=device,
            neighbors=args.neighbors,
        )


if __name__ == "__main__":
    main()
