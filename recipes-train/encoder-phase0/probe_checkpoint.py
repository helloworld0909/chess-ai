#!/usr/bin/env python3
"""Probe encoder checkpoint alignment quality — CPU-only, pure PyTorch + matplotlib.

Three probes:
  1. Linear piece-identity probe — single linear layer (Adam, CPU) on frozen encoder
     tokens predicts (color × piece_type): 13 classes.

  2. Spatial geometry heatmap — mean cos_sim(anchor_sq, every_sq) across N positions.
     Reveals whether CNN spatial inductive bias survived CLIP training.

  3. Confusion matrix — row-normalised heatmap of piece-type confusions.

All compute is CPU. GPU is left entirely for training.
Note: retrieval metrics (Top-1, margin, fixed-tau loss) are logged live during
training via top1=grid/global in the step log — no need to re-run Qwen here.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="recipes-train/encoder-phase0/config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--data", default="data/processed/encoder_pretrain_sf15_eval.jsonl")
    parser.add_argument("--n-positions", type=int, default=2000)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for PNGs. Defaults to the checkpoint's parent folder.",
    )
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


def probe_linear(tokens: torch.Tensor, labels: torch.Tensor, out_dir: Path) -> None:
    """Single linear layer trained with Adam on CPU — no sklearn."""
    n = len(tokens)
    split = int(n * 0.8)
    perm = torch.randperm(n)
    X_train, X_test = tokens[perm[:split]], tokens[perm[split:]]
    y_train, y_test = labels[perm[:split]], labels[perm[split:]]

    n_cls = 13
    D = tokens.shape[1]
    linear = torch.nn.Linear(D, n_cls, bias=True)
    opt = torch.optim.Adam(linear.parameters(), lr=1e-2, weight_decay=1e-4)

    # Inverse-frequency class weights so minority piece types get equal gradient weight
    counts = torch.bincount(y_train, minlength=n_cls).float()
    class_weights = 1.0 / counts.clamp(min=1)
    class_weights = class_weights / class_weights.sum() * n_cls  # normalize to mean=1

    batch = 4096
    for epoch in range(50):
        perm2 = torch.randperm(len(X_train))
        total_loss = 0.0
        for i in range(0, len(X_train), batch):
            idx = perm2[i : i + batch]
            logits = linear(X_train[idx])
            loss = F.cross_entropy(logits, y_train[idx], weight=class_weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.info("  epoch %d  loss=%.4f", epoch + 1, total_loss)

    with torch.no_grad():
        preds = linear(X_test).argmax(dim=1)
    acc = (preds == y_test).float().mean().item()

    # Per-class accuracy
    print("\n=== Probe 1: Linear Piece-Identity (PyTorch Adam) ===")
    print(f"Overall accuracy: {acc:.3f}  (random baseline: {1 / 13:.3f})")
    print(f"{'Class':<8} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    class_accs: list[tuple[str, float]] = []
    for cls_id, name in enumerate(_CLASS_NAMES):
        mask = y_test == cls_id
        if mask.sum() == 0:
            continue
        cls_acc = (preds[mask] == cls_id).float().mean().item()
        class_accs.append((name, cls_acc))
        print(
            f"{name:<8} {(preds[mask] == cls_id).sum().item():>8} {mask.sum().item():>8} {cls_acc:>8.3f}"
        )

    # Bar chart
    names_plot = [c[0] for c in class_accs]
    accs_plot = [c[1] for c in class_accs]
    colors = [
        "#4e79a7" if n.startswith("w") else "#f28e2b" if n.startswith("b") else "#76b7b2"
        for n in names_plot
    ]
    _, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(names_plot, accs_plot, color=colors)
    ax.axhline(acc, color="black", linestyle="--", linewidth=1, label=f"overall {acc:.3f}")
    ax.axhline(1 / 13, color="red", linestyle=":", linewidth=1, label="random 0.077")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Probe 1: Linear piece-identity accuracy per class")
    ax.legend()
    for bar, v in zip(bars, accs_plot):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.01,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    plt.tight_layout()
    out_path = out_dir / "linear_probe.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_path}")


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
# Probe 3: Confusion matrix (text)
# ---------------------------------------------------------------------------


def probe_confusion(tokens: torch.Tensor, labels: torch.Tensor, out_dir: Path) -> None:
    """Confusion matrix heatmap — row-normalized so each row sums to 100%.

    Diagonal = correct predictions (want high). Off-diagonal = confusions.
    """
    n = len(tokens)
    split = int(n * 0.8)
    perm = torch.randperm(n)
    X_train, X_test = tokens[perm[:split]], tokens[perm[split:]]
    y_train, y_test = labels[perm[:split]], labels[perm[split:]]

    n_cls = 13
    D = tokens.shape[1]
    linear = torch.nn.Linear(D, n_cls, bias=True)
    opt = torch.optim.Adam(linear.parameters(), lr=1e-2, weight_decay=1e-4)

    counts = torch.bincount(y_train, minlength=n_cls).float()
    class_weights = 1.0 / counts.clamp(min=1)
    class_weights = class_weights / class_weights.sum() * n_cls

    for _ in range(50):
        perm2 = torch.randperm(len(X_train))
        for i in range(0, len(X_train), 4096):
            idx = perm2[i : i + 4096]
            loss = F.cross_entropy(linear(X_train[idx]), y_train[idx], weight=class_weights)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        preds = linear(X_test).argmax(dim=1)

    # Build row-normalized confusion matrix
    conf = torch.zeros(n_cls, n_cls, dtype=torch.float)
    for t, p in zip(y_test.tolist(), preds.tolist()):
        conf[t, p] += 1
    row_sums = conf.sum(dim=1, keepdim=True).clamp(min=1)
    conf_pct = (conf / row_sums * 100).numpy()

    present = [i for i in range(n_cls) if conf[i].sum() > 0]
    names = [_CLASS_NAMES[i] for i in present]
    mat = conf_pct[np.ix_(present, present)]

    _, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(mat, vmin=0, vmax=100, cmap="Blues", aspect="equal")
    plt.colorbar(im, ax=ax, label="%")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized %)")

    # Annotate cells with value if ≥ 1%
    for i in range(len(present)):
        for j in range(len(present)):
            val = mat[i, j]
            if val >= 1:
                color = "white" if val > 60 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color)

    plt.tight_layout()
    out_path = out_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n=== Probe 3: Confusion Matrix ===")
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

    out_dir = Path(args.out_dir) if args.out_dir else checkpoint_path.parent
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

    probe_linear(tokens, labels, out_dir)
    probe_spatial_geometry(encoder, fens[:500], out_dir)
    probe_confusion(tokens, labels, out_dir)
    print(f"\nOutput saved to {out_dir}/")


if __name__ == "__main__":
    main()
