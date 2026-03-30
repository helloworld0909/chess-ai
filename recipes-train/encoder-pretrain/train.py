"""Encoder pre-training — SF15 classical eval terms + total eval score + piece classification.

Trains ONLY the CNN trunk + cross-attention SF15 head + piece head. No LLM involved.

Architecture:
    ChessEncoder(19-ch) → spatial_features() → (B, 64, 256)
    EncoderSF15Head:
        14 learned queries (13 SF15 terms + 1 total eval score)
        cross-attention over 64 spatial tokens in 256-dim space
        → (B, 14) predictions
    piece_head: Linear(256, 13) per token → (B, 64, 13)

Loss:
    term_weight(1.0) × MSE(pred[:, :13], sf15_terms)
    + eval_weight(3.0) × MSE(pred[:, 13], eval_score)
    + piece_weight(1.0) × CE(piece_logits, piece_labels)

Usage (2-GPU DDP):
    torchrun --nproc_per_node=2 recipes-train/encoder-pretrain/train.py

Usage (single GPU):
    uv run python recipes-train/encoder-pretrain/train.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import mmap
import os
import sys
import time
from pathlib import Path

import chess
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.encoder.board_tensor import board_to_tensor
from src.encoder.cnn import ChessEncoder
from training.lib import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# SF15 term names in canonical order — must match generate_encoder_pretrain_sf15.py
SF15_TERMS = [
    "Material",
    "Imbalance",
    "Pawns",
    "Knights",
    "Bishops",
    "Rooks",
    "Queens",
    "Mobility",
    "King safety",
    "Threats",
    "Passed",
    "Space",
    "Winnable",
]
N_TERMS = len(SF15_TERMS)  # 13

# Piece label classes: 0=empty, 1-6=wPNBRQK, 7-12=bpnbrqk
N_PIECE_CLASSES = 13
_PIECE_TO_CLASS = {
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


def board_to_piece_labels(board: chess.Board) -> torch.Tensor:
    """Return (64,) int64 tensor of piece class labels, row-major a1..h8."""
    labels = torch.zeros(64, dtype=torch.long)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            labels[sq] = _PIECE_TO_CLASS[(piece.color, piece.piece_type)]
    return labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EncoderSF15Dataset(Dataset):
    """Binary (.bin) dataset: 210 bytes/record, O(1) random access.

    Binary record layout:
        fen:          90 bytes (null-padded UTF-8)
        sf15_terms:   13 × float32
        eval_score:   1  × float32
        piece_labels: 64 × uint8
    """

    _FEN_BYTES = 90
    _RECORD_SIZE = _FEN_BYTES + N_TERMS * 4 + 4 + 64  # 210 bytes

    def __init__(self, path: str, limit: int = 0) -> None:
        self.path = path
        self.mm = None

        file_size = Path(path).stat().st_size
        assert file_size % self._RECORD_SIZE == 0, (
            f"Binary size {file_size} not divisible by {self._RECORD_SIZE}"
        )
        total = file_size // self._RECORD_SIZE
        self.length = min(total, limit) if limit else total
        logger.info("Binary dataset: %d records from %s", self.length, path)

    def _init_mmap(self) -> None:
        f = open(self.path, "rb")
        self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return self.length

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        import struct

        if self.mm is None:
            self._init_mmap()

        self.mm.seek(idx * self._RECORD_SIZE)
        raw = self.mm.read(self._RECORD_SIZE)
        fen = raw[: self._FEN_BYTES].rstrip(b"\x00").decode("utf-8")
        sf15_vals = struct.unpack_from(f"{N_TERMS}f", raw, self._FEN_BYTES)
        eval_val = struct.unpack_from("f", raw, self._FEN_BYTES + N_TERMS * 4)[0]
        piece_vals = raw[self._FEN_BYTES + N_TERMS * 4 + 4 :]
        sf15 = torch.tensor(sf15_vals, dtype=torch.float32)
        eval_score = torch.tensor(eval_val, dtype=torch.float32)
        piece_labels = torch.frombuffer(bytearray(piece_vals), dtype=torch.uint8).to(torch.long)

        board = chess.Board(fen)
        board_tensor = board_to_tensor(board)  # (19, 8, 8)
        return board_tensor, sf15, eval_score, piece_labels


class EncoderSF15JsonlDataset(torch.utils.data.IterableDataset):
    """Streaming JSONL dataset — no offset index, zero extra RAM.

    Each DataLoader worker opens the file once and reads its assigned stride of
    lines sequentially. With num_workers=W and DDP world_size=N, the global
    stride is W*N: worker w on rank r reads lines where
        line_idx % (W * N) == r * W + w

    Epoch ordering is deterministic by (epoch, rank, worker_id) seed.
    """

    def __init__(self, path: str, limit: int = 0, start_line: int = 0) -> None:
        self.path = path
        self.limit = limit
        self.start_line = start_line  # absolute file line to start from (0 = beginning)
        self._rank: int = 0
        self._world_size: int = 1
        self._len: int | None = None

    def _count_lines(self) -> int:
        count = 0
        with open(self.path, "rb") as f:
            for _ in f:
                count += 1
                if self.limit and count >= self.limit:
                    break
        return count

    def __len__(self) -> int:
        if self._len is None:
            self._len = self._count_lines()
        return self._len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        # DDP rank info passed via dataset attribute set in main()
        rank = getattr(self, "_rank", 0)
        world_size = getattr(self, "_world_size", 1)

        global_stride = num_workers * world_size
        my_slot = rank * num_workers + worker_id

        import itertools

        # Each worker's first line = start_line + my_slot, then every global_stride lines.
        # islice(f, start, stop, step) reads the file at C speed up to `start`.
        first_line = self.start_line + my_slot

        emitted = 0
        with open(self.path, "rb") as f:
            it = itertools.islice(f, first_line, None, global_stride)
            for raw in it:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                fen = rec.get("fen", "")
                if not fen:
                    continue
                try:
                    sf15 = torch.tensor(rec["sf15_terms"], dtype=torch.float32)
                    eval_score = torch.tensor(rec["eval_score"], dtype=torch.float32)
                    piece_labels = board_to_piece_labels(chess.Board(fen))
                    board = chess.Board(fen)
                    board_tensor = board_to_tensor(board)
                    yield board_tensor, sf15, eval_score, piece_labels
                    emitted += 1
                    if self.limit and emitted * global_stride >= self.limit:
                        break
                except Exception:
                    continue


def _make_dataset(path: str, limit: int = 0, start_line: int = 0):
    """Return the right dataset class based on file extension."""
    if path.endswith(".bin"):
        return EncoderSF15Dataset(path, limit=limit)
    return EncoderSF15JsonlDataset(path, limit=limit, start_line=start_line)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class EncoderSF15Head(nn.Module):
    """Cross-attention readout head operating in CNN's native 256-dim space.

    14 learned queries attend over 64 spatial tokens → (B, 14) scalar predictions.
    All in hidden_size (256) — tiny (~138K params).
    """

    def __init__(self, cnn_hidden_size: int, n_queries: int = N_TERMS + 1) -> None:
        super().__init__()
        self.scale = cnn_hidden_size**-0.5
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.empty(n_queries, cnn_hidden_size))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.key_proj = nn.Linear(cnn_hidden_size, cnn_hidden_size, bias=False)
        self.val_proj = nn.Linear(cnn_hidden_size, cnn_hidden_size, bias=False)
        self.scalar_heads = nn.Linear(cnn_hidden_size, n_queries)

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_features: (B, 64, cnn_hidden_size)
        Returns:
            (B, n_queries)
        """
        B = spatial_features.shape[0]
        K = self.key_proj(spatial_features)
        V = self.val_proj(spatial_features)
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)
        attn = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
        ctx = torch.bmm(attn, V)  # (B, n_queries, H)
        W = self.scalar_heads.weight  # (n_queries, H)
        b = self.scalar_heads.bias
        return torch.einsum("bqh,qh->bq", ctx, W) + b  # (B, n_queries)


class EncoderWithSF15Head(nn.Module):
    """ChessEncoder + SF15 cross-attention head + per-square piece classification head."""

    def __init__(self, encoder: ChessEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        cnn_hidden_size = encoder.proj.in_features  # 256
        self.sf15_head = EncoderSF15Head(cnn_hidden_size=cnn_hidden_size)
        self.piece_head = nn.Linear(cnn_hidden_size, N_PIECE_CLASSES)

    def forward(self, board_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            board_tensor: (B, 19, 8, 8)
        Returns:
            sf15_preds:   (B, 14)
            piece_logits: (B, 64, 13)
        """
        features = self.encoder.spatial_features(board_tensor)  # (B, 64, 256)
        sf15_preds = self.sf15_head(features)  # (B, 14)
        piece_logits = self.piece_head(features)  # (B, 64, 13)
        return sf15_preds, piece_logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def setup_ddp() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default="recipes-train/encoder-pretrain/config.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--finetune", action="store_true")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    config = load_config(args.config)
    train_cfg = config.get("training", {})
    encoder_cfg = config.get("encoder", {})
    loss_cfg = config.get("loss", {})
    wandb_cfg = config.get("wandb", {})
    output_dir = Path(config.get("output_dir", "checkpoints/encoder-pretrain"))

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        if wandb_cfg.get("enabled"):
            import wandb

            wandb.init(
                project=wandb_cfg.get("project", "chess-tutor"),
                name=wandb_cfg.get("name", "encoder-pretrain"),
                tags=wandb_cfg.get("tags", ["encoder-pretrain"]),
                config=config,
            )

    term_weight: float = loss_cfg.get("term_weight", 1.0)
    eval_weight: float = loss_cfg.get("eval_weight", 3.0)
    piece_weight: float = loss_cfg.get("piece_weight", 1.0)

    # Peek at checkpoint before building dataset — compute start_line from step
    resume_step = 0
    resume_start_line = 0
    if args.resume and not args.finetune:
        _peek = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_step = _peek.get("step", 0)
        # start_line = records already consumed = step * batch_per_step
        # batch_per_step = per_device_batch * world_size (world_size known after DDP init)
        # Use world_size from torchrun env if available, else 1
        _ws = int(os.environ.get("WORLD_SIZE", 1))
        _bs = train_cfg.get("per_device_train_batch_size", 256)
        resume_start_line = resume_step * _bs * _ws
        if is_main:
            logger.info(
                "Resuming from step %d, skipping first %d lines", resume_step, resume_start_line
            )
        del _peek

    # ── Dataset ──────────────────────────────────────────────────────────────
    num_workers = train_cfg.get("dataloader_num_workers", 4)
    train_ds = _make_dataset(
        train_cfg["train_file"],
        limit=train_cfg.get("train_limit", 0),
        start_line=resume_start_line,
    )
    eval_ds = None
    if train_cfg.get("eval_file"):
        eval_ds = _make_dataset(train_cfg["eval_file"], limit=train_cfg.get("eval_limit", 0))

    # Inject rank/world_size into IterableDatasets so they can stride correctly.
    # Map-style datasets use DistributedSampler instead.
    is_iterable = isinstance(train_ds, torch.utils.data.IterableDataset)
    if is_iterable:
        train_ds._rank = rank
        train_ds._world_size = world_size
        train_sampler = None
        train_shuffle = False
    else:
        train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
        train_shuffle = train_sampler is None

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("per_device_train_batch_size", 256),
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_loader = None
    if eval_ds is not None:
        is_eval_iterable = isinstance(eval_ds, torch.utils.data.IterableDataset)
        if is_eval_iterable:
            eval_ds._rank = rank
            eval_ds._world_size = world_size
            eval_sampler = None
        else:
            eval_sampler = DistributedSampler(eval_ds, shuffle=False) if world_size > 1 else None
        eval_loader = DataLoader(
            eval_ds,
            batch_size=train_cfg.get("per_device_eval_batch_size", 512),
            sampler=eval_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    encoder = ChessEncoder(
        in_channels=encoder_cfg.get("in_channels", 19),
        hidden_size=encoder_cfg.get("hidden_size", 256),
        num_blocks=encoder_cfg.get("num_blocks", 10),
        out_dim=encoder_cfg.get("out_dim", 2560),
    )
    model = EncoderWithSF15Head(encoder).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=not args.finetune)
        if is_main:
            logger.info(
                "%s from %s (step %d)",
                "Fine-tuning" if args.finetune else "Resumed",
                args.resume,
                resume_step,
            )

    if world_size > 1:
        # find_unused_parameters=True: encoder.proj (256→2560) unused during pretrain
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    lr = train_cfg.get("learning_rate", 3e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=train_cfg.get("weight_decay", 0.01)
    )

    num_epochs = train_cfg.get("num_train_epochs", 3)
    bs = train_cfg.get("per_device_train_batch_size", 256)
    # For IterableDataset, len() triggers a full line scan — use a config override if set.
    # Fall back to len() only for map-style datasets (no scan needed).
    if isinstance(train_ds, torch.utils.data.IterableDataset):
        # Approximate: file_size / avg_bytes_per_record, divided across ranks
        file_size = Path(train_cfg["train_file"]).stat().st_size
        approx_records = file_size // 175  # ~175 bytes/record empirically
        steps_per_epoch = math.ceil(approx_records / (bs * world_size))
    else:
        steps_per_epoch = math.ceil(len(train_ds) / (bs * world_size))
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.03))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Restore optimizer + scheduler state after creating them (needs model on device first)
    if args.resume and not args.finetune:
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    logging_steps = train_cfg.get("logging_steps", 200)
    save_steps = train_cfg.get("save_steps", 10000)
    eval_steps = train_cfg.get("eval_steps", 10000)
    save_total_limit = train_cfg.get("save_total_limit", 5)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
    bf16 = train_cfg.get("bf16", True)
    amp_dtype = torch.bfloat16 if bf16 else torch.float16

    global_step = resume_step
    saved_checkpoints: list[Path] = []

    def save_checkpoint(step: int, epoch: int, eval_loss: float | None = None) -> None:
        nonlocal saved_checkpoints
        ckpt_dir = output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        raw_model = model.module if isinstance(model, DDP) else model
        torch.save(
            {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "epoch": epoch,
                "eval_loss": eval_loss,
            },
            ckpt_dir / "checkpoint.pt",
        )
        encoder_state = {
            k.removeprefix("encoder."): v
            for k, v in raw_model.state_dict().items()
            if k.startswith("encoder.")
        }
        torch.save(encoder_state, ckpt_dir / "encoder_weights.pt")
        if is_main:
            logger.info("Saved checkpoint → %s (eval_loss=%.4f)", ckpt_dir, eval_loss or 0)
        saved_checkpoints.append(ckpt_dir)
        if save_total_limit and len(saved_checkpoints) > save_total_limit:
            import shutil

            oldest = saved_checkpoints.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)

    def run_eval() -> dict[str, float]:
        if eval_loader is None:
            return {}
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.eval()
        total = {"loss": 0.0, "term": 0.0, "eval": 0.0, "piece": 0.0, "n": 0.0}
        with torch.no_grad():
            for boards, sf15, eval_score, piece_labels in eval_loader:
                boards = boards.to(device, non_blocking=True)
                sf15 = sf15.to(device, non_blocking=True)
                eval_score = eval_score.to(device, non_blocking=True)
                piece_labels = piece_labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    sf15_preds, piece_logits = raw_model(boards)
                    term_loss = mse(sf15_preds[:, :N_TERMS], sf15)
                    ev_loss = mse(sf15_preds[:, N_TERMS], eval_score)
                    p_loss = ce(piece_logits.reshape(-1, N_PIECE_CLASSES), piece_labels.reshape(-1))
                    loss = term_weight * term_loss + eval_weight * ev_loss + piece_weight * p_loss
                bs = boards.size(0)
                total["loss"] += loss.item() * bs
                total["term"] += term_loss.item() * bs
                total["eval"] += ev_loss.item() * bs
                total["piece"] += p_loss.item() * bs
                total["n"] += bs
        raw_model.train()
        t = torch.tensor(list(total.values()), dtype=torch.float64, device=device)
        if world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        n = t[4].item()
        if n > 0:
            return {k: (t[i] / n).item() for i, k in enumerate(["loss", "term", "eval", "piece"])}
        return {k: 0.0 for k in ["loss", "term", "eval", "piece"]}

    # ── Training loop ──────────────────────────────────────────────────────────
    if is_main:
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            "Encoder pretrain | params=%s | steps/epoch=%d | total_steps=%d"
            " | term_weight=%.1f eval_weight=%.1f piece_weight=%.1f",
            f"{param_count:,}",
            steps_per_epoch,
            total_steps,
            term_weight,
            eval_weight,
            piece_weight,
        )

    model.train()
    t0 = time.time()
    running = {"loss": 0.0, "term": 0.0, "eval": 0.0, "piece": 0.0, "n": 0}

    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for boards, sf15, eval_score, piece_labels in train_loader:
            boards = boards.to(device, non_blocking=True)
            sf15 = sf15.to(device, non_blocking=True)
            eval_score = eval_score.to(device, non_blocking=True)
            piece_labels = piece_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                sf15_preds, piece_logits = model(boards)
                term_loss = mse(sf15_preds[:, :N_TERMS], sf15)
                ev_loss = mse(sf15_preds[:, N_TERMS], eval_score)
                p_loss = ce(piece_logits.reshape(-1, N_PIECE_CLASSES), piece_labels.reshape(-1))
                loss = term_weight * term_loss + eval_weight * ev_loss + piece_weight * p_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

            bs = boards.size(0)
            running["loss"] += loss.item() * bs
            running["term"] += term_loss.item() * bs
            running["eval"] += ev_loss.item() * bs
            running["piece"] += p_loss.item() * bs
            running["n"] += bs

            if is_main and global_step % logging_steps == 0:
                n = running["n"]
                avg = {k: running[k] / n for k in ("loss", "term", "eval", "piece")}
                lr_now = scheduler.get_last_lr()[0]
                logger.info(
                    "step=%d epoch=%d  loss=%.4f term=%.4f eval=%.4f piece=%.4f  lr=%.2e  %.1fs",
                    global_step,
                    epoch + 1,
                    avg["loss"],
                    avg["term"],
                    avg["eval"],
                    avg["piece"],
                    lr_now,
                    time.time() - t0,
                )
                if wandb_cfg.get("enabled"):
                    import wandb

                    wandb.log(
                        {
                            "train/loss": avg["loss"],
                            "train/term_loss": avg["term"],
                            "train/eval_loss": avg["eval"],
                            "train/piece_loss": avg["piece"],
                            "train/lr": lr_now,
                        },
                        step=global_step,
                    )
                running = {"loss": 0.0, "term": 0.0, "eval": 0.0, "piece": 0.0, "n": 0}

            do_eval = global_step % eval_steps == 0
            do_save = is_main and global_step % save_steps == 0
            if do_eval or do_save:
                metrics = run_eval()
                if is_main and metrics and do_eval:
                    logger.info(
                        "step=%d  eval loss=%.4f term=%.4f eval_score=%.4f piece=%.4f",
                        global_step,
                        metrics["loss"],
                        metrics["term"],
                        metrics["eval"],
                        metrics["piece"],
                    )
                    if wandb_cfg.get("enabled"):
                        import wandb

                        wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)
                if do_save:
                    save_checkpoint(global_step, epoch + 1, metrics.get("loss"))

    if is_main:
        metrics = run_eval()
        logger.info("Training done. Final eval: %s", metrics)
        save_checkpoint(global_step, num_epochs, metrics.get("loss"))

    cleanup_ddp()


if __name__ == "__main__":
    main()
