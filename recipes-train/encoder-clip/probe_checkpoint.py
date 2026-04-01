#!/usr/bin/env python3
"""Probe encoder checkpoint alignment quality — CPU-only, pure PyTorch + matplotlib.

Three probes:
  1. Linear piece-identity probe — single linear layer (SGD, CPU) on frozen encoder
     tokens predicts (color × piece_type): 13 classes. No sklearn.

  2. Spatial geometry heatmap — mean cos_sim(anchor_sq, every_sq) across N positions.
     Reveals whether CNN spatial inductive bias survived CLIP training.

  3. PCA clustering — torch.pca_lowrank projects all 64×N tokens to 2D, colored by
     piece type. Clean clusters = structured embedding space.

All compute is CPU. GPU is left entirely for training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import chess
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers.utils import logging as hf_logging

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.encoder.board_tensor import board_to_tensor
from src.encoder.cnn import ChessEncoder
from training.lib import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 13 classes: empty=0, wP=1..wK=6, bP=7..bK=12
_PIECE_CLASS = {
    (chess.WHITE, chess.PAWN): 1,
    (chess.WHITE, chess.KNIGHT): 2,
    (chess.WHITE, chess.BISHOP): 3,
    (chess.WHITE, chess.ROOK): 4,
    (chess.WHITE, chess.QUEEN): 5,
    (chess.WHITE, chess.KING): 6,
    (chess.BLACK, chess.PAWN): 7,
    (chess.BLACK, chess.KNIGHT): 8,
    (chess.BLACK, chess.BISHOP): 9,
    (chess.BLACK, chess.ROOK): 10,
    (chess.BLACK, chess.QUEEN): 11,
    (chess.BLACK, chess.KING): 12,
}
_CLASS_NAMES = ["empty", "wP", "wN", "wB", "wR", "wQ", "wK", "bP", "bN", "bB", "bR", "bQ", "bK"]
_COLORS = [
    "#aaaaaa",  # empty
    "#ffe066",
    "#ffd700",
    "#ffb300",
    "#ff8c00",
    "#ff4500",
    "#cc0000",  # white P N B R Q K
    "#a8d8f0",
    "#74b9ff",
    "#0984e3",
    "#636e72",
    "#6c5ce7",
    "#1a1a2e",  # black P N B R Q K
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="recipes-train/encoder-clip/config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--data", default="data/processed/encoder_pretrain_sf15_eval.jsonl")
    parser.add_argument("--n-positions", type=int, default=2000)
    parser.add_argument("--out-dir", default="checkpoints/encoder-clip/probe")
    return parser.parse_args()


def load_encoder(enc_cfg: dict, checkpoint_path: str) -> ChessEncoder:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    encoder = ChessEncoder(
        in_channels=enc_cfg.get("in_channels", 19),
        hidden_size=enc_cfg.get("hidden_size", 256),
        num_blocks=enc_cfg.get("num_blocks", 10),
        out_dim=enc_cfg.get("out_dim", 2560),
    )
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()
    return encoder


def encode_positions(encoder: ChessEncoder, fens: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode FENs on CPU. Returns (tokens, labels).

    tokens: (N*64, D) — L2-normalized grid tokens
    labels: (N*64,)  — piece class 0–12
    """
    all_tokens: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for fen in fens:
            board = chess.Board(fen)
            out = encoder(board_to_tensor(board).unsqueeze(0))  # (1, 65, D)
            grid = F.normalize(out[0, :64].float(), dim=-1)  # (64, D)
            labels = torch.tensor(
                [
                    _PIECE_CLASS.get((p.color, p.piece_type), 0) if (p := board.piece_at(sq)) else 0
                    for sq in chess.SQUARES
                ],
                dtype=torch.long,
            )
            all_tokens.append(grid)
            all_labels.append(labels)
    return torch.cat(all_tokens, dim=0), torch.cat(all_labels, dim=0)


# ---------------------------------------------------------------------------
# Probe 1: Linear piece-identity probe (pure PyTorch SGD)
# ---------------------------------------------------------------------------


def probe_linear(tokens: torch.Tensor, labels: torch.Tensor) -> None:
    """Single linear layer trained with SGD on CPU — no sklearn."""
    n = len(tokens)
    split = int(n * 0.8)
    perm = torch.randperm(n)
    X_train, X_test = tokens[perm[:split]], tokens[perm[split:]]
    y_train, y_test = labels[perm[:split]], labels[perm[split:]]

    n_cls = 13
    D = tokens.shape[1]
    linear = torch.nn.Linear(D, n_cls, bias=True)
    opt = torch.optim.SGD(linear.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    batch = 4096
    for epoch in range(20):
        perm2 = torch.randperm(len(X_train))
        total_loss = 0.0
        for i in range(0, len(X_train), batch):
            idx = perm2[i : i + batch]
            logits = linear(X_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            logger.info("  epoch %d  loss=%.4f", epoch + 1, total_loss)

    with torch.no_grad():
        preds = linear(X_test).argmax(dim=1)
    acc = (preds == y_test).float().mean().item()

    # Per-class accuracy
    print("\n=== Probe 1: Linear Piece-Identity (PyTorch SGD) ===")
    print(f"Overall accuracy: {acc:.3f}  (random baseline: {1 / 13:.3f})")
    print(f"{'Class':<8} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    for cls_id, name in enumerate(_CLASS_NAMES):
        mask = y_test == cls_id
        if mask.sum() == 0:
            continue
        cls_acc = (preds[mask] == cls_id).float().mean().item()
        print(
            f"{name:<8} {(preds[mask] == cls_id).sum().item():>8} {mask.sum().item():>8} {cls_acc:>8.3f}"
        )


# ---------------------------------------------------------------------------
# Probe 2: Spatial geometry heatmap
# ---------------------------------------------------------------------------


def probe_spatial_geometry(encoder: ChessEncoder, fens: list[str], out_dir: Path) -> None:
    """Mean cos_sim(anchor, sq) across positions — saved as PNG heatmap."""
    logger.info("Probe 2: Spatial geometry (%d positions)", len(fens))

    all_grid: list[torch.Tensor] = []
    with torch.no_grad():
        for fen in fens:
            board = chess.Board(fen)
            out = encoder(board_to_tensor(board).unsqueeze(0))
            grid = F.normalize(out[0, :64].float(), dim=-1)  # (64, D)
            all_grid.append(grid)
    grid_stack = torch.stack(all_grid, dim=0)  # (N, 64, D)

    anchor_squares = [chess.A1, chess.D4, chess.E5, chess.H8]
    anchor_names = ["a1", "d4", "e5", "h8"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Spatial Geometry: mean cos_sim(anchor, sq) across positions", fontsize=12)

    for ax, anc_sq, anc_name in zip(axes, anchor_squares, anchor_names):
        anc = grid_stack[:, anc_sq, :]  # (N, D)
        sims = torch.einsum("nd,nsd->ns", anc, grid_stack).mean(dim=0)  # (64,)

        heat = np.zeros((8, 8))
        for sq in range(64):
            heat[chess.square_rank(sq), chess.square_file(sq)] = sims[sq].item()

        im = ax.imshow(heat[::-1], vmin=-0.1, vmax=1.0, cmap="RdYlGn", aspect="equal")
        ax.set_title(f"anchor={anc_name}")
        ax.set_xticks(range(8))
        ax.set_xticklabels(list("abcdefgh"))
        ax.set_yticks(range(8))
        ax.set_yticklabels(list("87654321"))
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    out_path = out_dir / "spatial_geometry.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n=== Probe 2: Spatial Geometry ===")
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Probe 3: PCA clustering
# ---------------------------------------------------------------------------


def probe_pca(tokens: torch.Tensor, labels: torch.Tensor, out_dir: Path) -> None:
    """PCA (torch.pca_lowrank) to 2D, colored by piece type — saved as PNG."""
    logger.info("Probe 3: PCA clustering (%d tokens)", len(tokens))

    # Subsample to 20k max for speed
    max_pts = 20000
    if len(tokens) > max_pts:
        idx = torch.randperm(len(tokens))[:max_pts]
        tokens, labels = tokens[idx], labels[idx]

    # Center before PCA
    X = tokens - tokens.mean(dim=0, keepdim=True)
    _, _, V = torch.pca_lowrank(X, q=2, niter=4)
    emb = (X @ V).numpy()  # (N, 2)
    labels_np = labels.numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_id, name in enumerate(_CLASS_NAMES):
        mask = labels_np == cls_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            emb[mask, 0], emb[mask, 1], c=_COLORS[cls_id], label=name, s=4, alpha=0.6, linewidths=0
        )

    ax.legend(markerscale=4, loc="best", fontsize=8)
    ax.set_title("PCA of encoder grid tokens, colored by piece type")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    out_path = out_dir / "pca_clusters.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n=== Probe 3: PCA Clustering ===")
    print(f"Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.getLogger("transformers").setLevel(logging.ERROR)
    hf_logging.set_verbosity_error()
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    logger.info("Loading encoder from %s (CPU)", checkpoint_path)
    encoder = load_encoder(config["encoder"], str(checkpoint_path))

    logger.info("Loading %d FENs from %s", args.n_positions, args.data)
    fens: list[str] = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            if i >= args.n_positions:
                break
            fens.append(json.loads(line)["fen"])

    logger.info("Encoding %d positions on CPU...", len(fens))
    tokens, labels = encode_positions(encoder, fens)
    logger.info("tokens: %s  labels: %s", tuple(tokens.shape), tuple(labels.shape))

    probe_linear(tokens, labels)
    probe_spatial_geometry(encoder, fens[:500], out_dir)
    probe_pca(tokens, labels, out_dir)

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
