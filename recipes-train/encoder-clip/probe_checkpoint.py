#!/usr/bin/env python3
"""Probe encoder checkpoint alignment quality — CPU-only, pure PyTorch + matplotlib.

Four probes:
  1. Linear piece-identity probe — single linear layer (SGD, CPU) on frozen encoder
     tokens predicts (color × piece_type): 13 classes.

  2. Spatial geometry heatmap — mean cos_sim(anchor_sq, every_sq) across N positions.
     Reveals whether CNN spatial inductive bias survived CLIP training.

  3. Confusion matrix — row-normalised heatmap of piece-type confusions.

  4. Retrieval metrics (tau-agnostic lie detectors) — computed from raw cosine
     similarities between encoder grid tokens and Qwen text anchors, with no
     temperature scaling:
       - Top-1 retrieval accuracy (random baseline ≈ 1/N)
       - Positive sim S_pos, hard-negative sim S_hard_neg, margin
       - Fixed-tau (0.07) InfoNCE loss — decoupled from learned tau
       - Top-1 accuracy heatmap per board square (8×8 PNG)

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
RECIPE_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(RECIPE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from train import (  # noqa: E402  (train.py is in the same directory)
    EmbeddingCache,
    build_global_description,
    describe_square,
    embed_texts,
    get_anchor_embeddings,
)

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
    parser.add_argument("--config", default="recipes-train/encoder-clip/config.yaml")
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

    # Inverse-frequency class weights so minority piece types get equal gradient weight
    counts = torch.bincount(y_train, minlength=n_cls).float()
    class_weights = 1.0 / counts.clamp(min=1)
    class_weights = class_weights / class_weights.sum() * n_cls  # normalize to mean=1

    batch = 4096
    for epoch in range(20):
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
    opt = torch.optim.SGD(linear.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    counts = torch.bincount(y_train, minlength=n_cls).float()
    class_weights = 1.0 / counts.clamp(min=1)
    class_weights = class_weights / class_weights.sum() * n_cls

    for _ in range(20):
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
# Probe 4: Retrieval metrics — tau-agnostic lie detectors
# ---------------------------------------------------------------------------

_FIXED_TAU = 0.07  # OpenAI CLIP default — used for fixed-tau eval loss


def _compute_qwen_anchors(
    fens: list[str],
    sf15_rows: list[list[float]],
    eval_scores: list[float],
    qwen_model_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load Qwen on CPU and compute (N, 64, D) grid anchors + (N, D) global anchors.

    Uses embed_texts + get_anchor_embeddings from train.py — single source of truth
    for label generation so probed anchors exactly match training targets.
    """
    from transformers import AutoModel, AutoTokenizer

    logger.info("Probe 4: loading Qwen %s on CPU for anchor embeddings...", qwen_model_name)
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModel.from_pretrained(
        qwen_model_name,
        dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=True,
        local_files_only=True,
    )
    qwen.eval()

    cache = EmbeddingCache(maxsize=len(fens) * 65 + 100, dtype=torch.float32)
    device = torch.device("cpu")

    N = len(fens)
    grid_descs: list[str] = []
    global_descs: list[str] = []
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        for sq in range(64):
            grid_descs.append(describe_square(board, sq))
        global_descs.append(build_global_description(sf15_rows[i], eval_scores[i], board))

    logger.info(
        "  embedding %d grid + %d global descriptions...", len(grid_descs), len(global_descs)
    )
    all_embs = get_anchor_embeddings(grid_descs + global_descs, cache, qwen, tokenizer, device)

    del qwen  # free ~16GB CPU RAM
    grid_anchors = F.normalize(all_embs[: N * 64].reshape(N, 64, -1).float(), dim=-1)
    global_anchors = F.normalize(all_embs[N * 64 :].float(), dim=-1)
    return grid_anchors, global_anchors


def probe_retrieval(
    encoder: ChessEncoder,
    fens: list[str],
    sf15_rows: list[list[float]],
    eval_scores: list[float],
    qwen_model_name: str,
    out_dir: Path,
) -> None:
    """Tau-agnostic retrieval metrics between CNN grid tokens and Qwen text anchors.

    For each of 64 square positions independently:
      - Top-1 retrieval accuracy: does argmax(sim[i, :]) == i?
      - Positive sim S_pos: mean diagonal cosine similarity
      - Hard-negative sim S_hard: mean of max off-diagonal per row
      - Margin: S_pos - S_hard
      - Fixed-tau InfoNCE at tau=0.07 (decoupled from learned tau)

    Saves a per-square Top-1 accuracy heatmap (8×8 PNG).
    """
    N = len(fens)

    # Encode all positions — (N, 64, D) normalised grid tokens + (N, D) global
    logger.info("Probe 4: encoding %d positions...", N)
    grid_tokens_list: list[torch.Tensor] = []
    global_tokens_list: list[torch.Tensor] = []
    with torch.no_grad():
        for fen in fens:
            out = encoder(board_to_tensor(chess.Board(fen)).unsqueeze(0))  # (1, 65, D)
            grid_tokens_list.append(F.normalize(out[0, :64].float(), dim=-1))
            global_tokens_list.append(F.normalize(out[0, 64].float(), dim=0))
    grid_tokens = torch.stack(grid_tokens_list, dim=0)  # (N, 64, D)
    global_tokens = torch.stack(global_tokens_list, dim=0)  # (N, D)

    # Compute Qwen anchors (loads Qwen, slow)
    grid_anchors, global_anchors = _compute_qwen_anchors(
        fens, sf15_rows, eval_scores, qwen_model_name
    )

    # ── Per-square retrieval metrics ──────────────────────────────────────────
    sq_top1 = np.zeros(64)
    sq_pos_sim = np.zeros(64)
    sq_hard_sim = np.zeros(64)

    labels = torch.arange(N)
    with torch.no_grad():
        for sq in range(64):
            v = grid_tokens[:, sq, :]  # (N, D) CNN
            t = grid_anchors[:, sq, :]  # (N, D) Qwen
            sim = v @ t.T  # (N, N) raw cosine (both L2-normalised)

            preds = sim.argmax(dim=1)
            sq_top1[sq] = (preds == labels).float().mean().item()

            pos = sim.diagonal().mean().item()
            mask = torch.eye(N, dtype=torch.bool)
            hard = sim.masked_fill(mask, -1.0).max(dim=1).values.mean().item()
            sq_pos_sim[sq] = pos
            sq_hard_sim[sq] = hard

    # ── Global token retrieval ────────────────────────────────────────────────
    with torch.no_grad():
        sim_global = global_tokens @ global_anchors.T  # (N, N)
        global_top1 = (sim_global.argmax(dim=1) == labels).float().mean().item()
        global_pos = sim_global.diagonal().mean().item()
        mask = torch.eye(N, dtype=torch.bool)
        global_hard = sim_global.masked_fill(mask, -1.0).max(dim=1).values.mean().item()

    # ── Fixed-tau InfoNCE ─────────────────────────────────────────────────────
    with torch.no_grad():
        # Mean over all 64 squares
        fixed_tau_losses = []
        for sq in range(64):
            v = grid_tokens[:, sq, :]
            t = grid_anchors[:, sq, :]
            logits = (v @ t.T) / _FIXED_TAU
            lbl = torch.arange(N)
            loss = (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.T, lbl)) / 2
            fixed_tau_losses.append(loss.item())
        fixed_tau_grid = float(np.mean(fixed_tau_losses))

        logits_g = sim_global / _FIXED_TAU
        fixed_tau_global = (
            (F.cross_entropy(logits_g, labels) + F.cross_entropy(logits_g.T, labels)) / 2
        ).item()

    # ── Print summary ─────────────────────────────────────────────────────────
    mean_top1 = sq_top1.mean()
    mean_pos = sq_pos_sim.mean()
    mean_hard = sq_hard_sim.mean()
    mean_margin = mean_pos - mean_hard
    random_baseline = 1.0 / N

    print(f"\n=== Probe 4: Retrieval Metrics (tau-agnostic) ===")
    print(f"N={N}  random_baseline={random_baseline:.4f}")
    print(f"                   Grid (mean/64sq)   Global")
    print(
        f"  Top-1 accuracy:  {mean_top1:.4f}            {global_top1:.4f}  (random: {random_baseline:.4f})"
    )
    print(f"  S_pos  (diag):   {mean_pos:.4f}            {global_pos:.4f}")
    print(f"  S_hard (max neg):{mean_hard:.4f}            {global_hard:.4f}")
    print(f"  Margin:          {mean_margin:.4f}            {global_pos - global_hard:.4f}")
    print(
        f"  Fixed-tau loss (tau={_FIXED_TAU}): grid={fixed_tau_grid:.4f}  global={fixed_tau_global:.4f}"
    )

    # ── Top-1 accuracy heatmap (8×8) ─────────────────────────────────────────
    heat = np.zeros((8, 8))
    for sq in range(64):
        heat[chess.square_rank(sq), chess.square_file(sq)] = sq_top1[sq]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        heat[::-1],
        vmin=0,
        vmax=max(sq_top1.max(), random_baseline * 3),
        cmap="RdYlGn",
        aspect="equal",
    )
    ax.set_xticks(range(8))
    ax.set_xticklabels(list("abcdefgh"))
    ax.set_yticks(range(8))
    ax.set_yticklabels(list("87654321"))
    ax.set_title(f"Top-1 retrieval accuracy per square (N={N}, random={random_baseline:.4f})")
    for sq in range(64):
        r, f = chess.square_rank(sq), chess.square_file(sq)
        ax.text(f, 7 - r, f"{sq_top1[sq]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out_path = out_dir / "retrieval_heatmap.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


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
    sf15_rows: list[list[float]] = []
    eval_scores: list[float] = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            if i >= args.n_positions:
                break
            rec = json.loads(line)
            fens.append(rec["fen"])
            sf15_rows.append(rec.get("sf15_terms", [0.0] * 11))
            eval_scores.append(float(rec.get("eval_score", 0.0)))

    logger.info("Encoding %d positions on CPU...", len(fens))
    tokens, labels = encode_positions(encoder, fens)
    logger.info("tokens: %s  labels: %s", tuple(tokens.shape), tuple(labels.shape))

    probe_linear(tokens, labels)
    probe_spatial_geometry(encoder, fens[:500], out_dir)
    probe_confusion(tokens, labels, out_dir)
    probe_retrieval(
        encoder,
        fens,
        sf15_rows,
        eval_scores,
        config["clip"]["qwen_model"],
        out_dir,
    )
    print(f"\nOutput saved to {out_dir}/")


if __name__ == "__main__":
    main()
