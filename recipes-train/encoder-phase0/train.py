"""Chess-CLIP: Contrastive encoder alignment against frozen Qwen embeddings.

Trains the ChessEncoder CNN (grid tokens + global cross-attention token) to align its
65-token output with frozen Qwen3.5-4B text embeddings via InfoNCE loss. After training,
CNN features lie on Qwen's language manifold — enabling effective LLM integration without
the phase0 bootstrap-deadlock failure mode.

Loss:
  L_grid   = per-position InfoNCE: for each of 64 square positions, B×B CLIP loss
              between CNN grid tokens and Qwen embeddings of per-square descriptions.
  L_global = InfoNCE over B board pairs: CNN global token vs Qwen embedding of
              eval-tier + top-3 SF15 imbalance description.
  loss = L_grid + global_loss_weight × L_global

Anchor text (online, per step):
  Grid   — "white knight on d4, attacked by black pawn on e5, controls 6 squares"
  Global — "White has a clear advantage, driven by: White leads in Mobility, ..."

Dataset: binary .bin (map-style, O(1) access) or streaming .jsonl.
  Binary record layout (210 bytes):
      fen: 90 bytes | sf15_terms: 13×f32 | eval_score: f32 | piece_labels: 64×u8

Usage (2-GPU DDP):
    torchrun --nproc_per_node=2 recipes-train/encoder-phase0/train.py

Usage (single GPU):
    uv run python recipes-train/encoder-phase0/train.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import mmap
import os
import struct
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import chess
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.encoder.board_tensor import board_to_tensor
from src.encoder.cnn import ChessEncoder
from training.lib import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("kernels").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

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

_EVAL_TIERS = [
    (-1e9, -2.5, "Black is winning decisively"),
    (-2.5, -1.0, "Black has a clear advantage"),
    (-1.0, -0.3, "Black has a slight advantage"),
    (-0.3, +0.3, "The position is approximately equal"),
    (+0.3, +1.0, "White has a slight advantage"),
    (+1.0, +2.5, "White has a clear advantage"),
    (+2.5, +1e9, "White is winning decisively"),
]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class ChessClipBinDataset(Dataset):
    """Binary map-style dataset (210 bytes/record, O(1) random access).

    Yields: (board_tensor, sf15, eval_score, fen)
    """

    _FEN_BYTES = 90
    _RECORD_SIZE = _FEN_BYTES + N_TERMS * 4 + 4 + 64  # 210 bytes

    def __init__(self, path: str, limit: int = 0) -> None:
        self.path = path
        self.mm: mmap.mmap | None = None
        file_size = Path(path).stat().st_size
        assert file_size % self._RECORD_SIZE == 0, (
            f"Binary size {file_size} not divisible by {self._RECORD_SIZE}"
        )
        total = file_size // self._RECORD_SIZE
        self.length = min(total, limit) if limit else total
        logger.info("ChessClipBinDataset: %d records from %s", self.length, path)

    def _init_mmap(self) -> None:
        f = open(self.path, "rb")
        self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        if self.mm is None:
            self._init_mmap()
        assert self.mm is not None
        self.mm.seek(idx * self._RECORD_SIZE)
        raw = self.mm.read(self._RECORD_SIZE)
        fen = raw[: self._FEN_BYTES].rstrip(b"\x00").decode("utf-8")
        sf15_vals = struct.unpack_from(f"{N_TERMS}f", raw, self._FEN_BYTES)
        eval_val = struct.unpack_from("f", raw, self._FEN_BYTES + N_TERMS * 4)[0]
        sf15 = torch.tensor(sf15_vals, dtype=torch.float32)
        eval_score = torch.tensor(eval_val, dtype=torch.float32)
        board = chess.Board(fen)
        return board_to_tensor(board), sf15, eval_score, fen


class ChessClipJsonlDataset(torch.utils.data.IterableDataset):
    """Streaming JSONL dataset. Yields: (board_tensor, sf15, eval_score, fen)

    DDP sharding: inject _rank / _world_size attributes before creating DataLoader.
    """

    def __init__(self, path: str, limit: int = 0) -> None:
        self.path = path
        self.limit = limit
        self._rank = 0
        self._world_size = 1
        self._resume_examples = 0

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        global_stride = num_workers * self._world_size
        my_slot = self._rank * num_workers + worker_id
        start_example = self._resume_examples
        if my_slot >= start_example:
            start_idx = my_slot
        else:
            steps = (start_example - my_slot + global_stride - 1) // global_stride
            start_idx = my_slot + steps * global_stride

        emitted = 0
        with open(self.path, "rb") as f:
            for raw in itertools.islice(f, start_idx, None, global_stride):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                    fen = rec["fen"]
                    sf15 = torch.tensor(rec["sf15_terms"], dtype=torch.float32)
                    eval_score = torch.tensor(rec["eval_score"], dtype=torch.float32)
                    board = chess.Board(fen)
                    yield board_to_tensor(board), sf15, eval_score, fen
                    emitted += 1
                    if self.limit and emitted * global_stride >= self.limit:
                        break
                except Exception:
                    continue


def _make_dataset(path: str, limit: int = 0) -> Dataset | torch.utils.data.IterableDataset:
    if path.endswith(".bin"):
        return ChessClipBinDataset(path, limit=limit)
    return ChessClipJsonlDataset(path, limit=limit)


def _collate(batch):
    boards = torch.stack([b[0] for b in batch])
    sf15 = torch.stack([b[1] for b in batch])
    eval_scores = torch.stack([b[2] for b in batch])
    fens = [b[3] for b in batch]
    return boards, sf15, eval_scores, fens


# ---------------------------------------------------------------------------
# Text description builders
# ---------------------------------------------------------------------------


def _eval_tier_prefix(eval_score: float) -> str:
    for lo, hi, label in _EVAL_TIERS:
        if lo <= eval_score < hi:
            return label
    return _EVAL_TIERS[-1][2]


def _term_qual_label(term_name: str, val: float) -> str | None:
    abs_val = abs(val)
    if abs_val < 0.1:
        return None
    side = "White" if val > 0 else "Black"
    if abs_val > 2.0:
        verb = "dominates in"
    elif abs_val > 0.5:
        verb = "leads in"
    else:
        verb = "is slightly better in"
    return f"{side} {verb} {term_name}"


_MATERIAL_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def _material_summary(board: chess.Board) -> str:
    syms = {
        chess.QUEEN: "Q",
        chess.ROOK: "R",
        chess.BISHOP: "B",
        chess.KNIGHT: "N",
        chess.PAWN: "P",
    }
    imbalances = []
    for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
        diff = len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))
        if diff > 0:
            imbalances.append(f"white +{diff}{syms[pt]}")
        elif diff < 0:
            imbalances.append(f"black +{-diff}{syms[pt]}")
    return "material: " + (", ".join(imbalances) if imbalances else "equal")


def build_global_description(sf15_terms: list[float], eval_score: float, board: chess.Board) -> str:
    """Position-specific global description: eval tier + SF15 imbalances + board state facts.

    Board state facts (side to move, move number, castling, en passant, material) make
    each description near-unique, preventing Qwen embeddings from clustering by template
    and ensuring the global InfoNCE loss has well-separated negatives.
    """
    stm = "White" if board.turn == chess.WHITE else "Black"
    header = f"{stm} to move"

    # Castling rights
    castling_parts = []
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_parts.append("white kingside")
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_parts.append("white queenside")
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_parts.append("black kingside")
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_parts.append("black queenside")
    castling = (
        f"castling available: {', '.join(castling_parts)}" if castling_parts else "no castling"
    )

    # En passant
    ep = f"en passant: {chess.square_name(board.ep_square)}" if board.ep_square else None

    # Eval tier + top-3 SF15 imbalances
    prefix = _eval_tier_prefix(eval_score)
    ranked = sorted(zip(SF15_TERMS, sf15_terms), key=lambda x: abs(x[1]), reverse=True)
    labels = [_term_qual_label(name, val) for name, val in ranked[:3]]
    labels = [lb for lb in labels if lb is not None]
    eval_part = f"{prefix}, driven by: {', '.join(labels)}." if labels else f"{prefix}."
    material = _material_summary(board)

    parts = [f"{header}.", material + ".", castling + ".", eval_part]
    if ep:
        parts.insert(2, ep + ".")
    if board.is_checkmate():
        parts.insert(1, "Checkmate!")
    elif board.is_check():
        parts.insert(1, "King is in check!")
    elif board.is_stalemate():
        parts.insert(1, "Stalemate!")
    return " ".join(parts)


def _pawn_structure_parts(board: chess.Board, sq: int, color: chess.Color) -> list[str]:
    file_idx = chess.square_file(sq)
    rank_idx = chess.square_rank(sq)
    friendly_pawns = board.pieces(chess.PAWN, color)
    enemy_pawns = board.pieces(chess.PAWN, not color)
    parts: list[str] = []

    ahead = range(rank_idx + 1, 8) if color == chess.WHITE else range(rank_idx - 1, -1, -1)
    if not any(
        chess.square(f, r) in enemy_pawns
        for f in [file_idx - 1, file_idx, file_idx + 1]
        if 0 <= f <= 7
        for r in ahead
    ):
        parts.append("passed pawn")

    if not any(
        chess.square(f, r) in friendly_pawns
        for f in [file_idx - 1, file_idx + 1]
        if 0 <= f <= 7
        for r in range(8)
    ):
        parts.append("isolated pawn")

    if any(chess.square(file_idx, r) in friendly_pawns for r in range(8) if r != rank_idx):
        parts.append("doubled pawn")

    return parts


_PIECE_VAL = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def _xray_target(board: chess.Board, sq: int) -> str | None:
    """Return description if an enemy sliding piece x-rays through `sq` to a higher-value piece."""
    piece = board.piece_at(sq)
    if piece is None:
        return None
    enemy = not piece.color
    piece_val = _PIECE_VAL.get(piece.piece_type, 0)
    dist_sq = chess.square_distance  # alias

    best: tuple[int, str] | None = None  # (target_value, description)
    for atk_sq in board.attackers(enemy, sq):
        attacker = board.piece_at(atk_sq)
        if attacker is None or attacker.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue
        ray = chess.BB_RAYS[atk_sq][sq]
        if not ray:
            continue
        d_to_sq = dist_sq(atk_sq, sq)
        # Walk squares on the ray beyond sq (farther from attacker than sq)
        for target_sq in sorted(
            (
                s
                for s in chess.SquareSet(ray)
                if s != atk_sq and s != sq and dist_sq(atk_sq, s) > d_to_sq
            ),
            key=lambda s: dist_sq(sq, s),
        ):
            target = board.piece_at(target_sq)
            if target is not None:
                tval = _PIECE_VAL.get(target.piece_type, 0)
                if tval > piece_val and (best is None or tval > best[0]):
                    tcol = "white" if target.color == chess.WHITE else "black"
                    best = (
                        tval,
                        f"x-rays {tcol} {chess.piece_name(target.piece_type)} on {chess.square_name(target_sq)}",
                    )
                break  # first piece behind sq along this ray

    return best[1] if best else None


def _file_type(board: chess.Board, sq: int) -> str:
    fi = chess.square_file(sq)
    wp = any(chess.square_file(p) == fi for p in board.pieces(chess.PAWN, chess.WHITE))
    bp = any(chess.square_file(p) == fi for p in board.pieces(chess.PAWN, chess.BLACK))
    if not wp and not bp:
        return "open file"
    if not wp:
        return "half-open file for white"
    if not bp:
        return "half-open file for black"
    return "closed file"


def _rank_type(board: chess.Board, sq: int) -> str:
    ri = chess.square_rank(sq)
    wp = any(chess.square_rank(p) == ri for p in board.pieces(chess.PAWN, chess.WHITE))
    bp = any(chess.square_rank(p) == ri for p in board.pieces(chess.PAWN, chess.BLACK))
    if not wp and not bp:
        return "open rank"
    if not wp:
        return "half-open rank for white"
    if not bp:
        return "half-open rank for black"
    return "closed rank"


def _nearest_pawn_on_file(board: chess.Board, sq: int) -> str | None:
    """Distance to nearest pawn (either color) on the same file. None if no pawns."""
    fi = chess.square_file(sq)
    ri = chess.square_rank(sq)
    min_dist = None
    for color in (chess.WHITE, chess.BLACK):
        for p in board.pieces(chess.PAWN, color):
            if chess.square_file(p) == fi:
                d = abs(chess.square_rank(p) - ri)
                if min_dist is None or d < min_dist:
                    min_dist = d
    if min_dist is None:
        return None
    return f"{min_dist} square{'s' if min_dist != 1 else ''} from nearest pawn on file"


def _is_outpost(board: chess.Board, sq: int, color: chess.Color) -> bool:
    """True if no enemy pawn can ever attack sq (outpost for `color`)."""
    fi = chess.square_file(sq)
    ri = chess.square_rank(sq)
    enemy = not color
    for p in board.pieces(chess.PAWN, enemy):
        pf = chess.square_file(p)
        pr = chess.square_rank(p)
        if abs(pf - fi) == 1:
            # Enemy pawn on adjacent file: can it reach a rank where it attacks sq?
            # White outpost: black pawns advance by decreasing rank; attack sq at rank ri
            # from rank ri+1. Black pawn at pr can reach ri+1 if pr >= ri+1 (black moves down).
            # Black outpost: white pawns advance by increasing rank; attack sq at rank ri
            # from rank ri-1. White pawn at pr can reach ri-1 if pr <= ri-1 (white moves up).
            if color == chess.WHITE and pr >= ri + 1:
                return False
            if color == chess.BLACK and pr <= ri - 1:
                return False
    return True


_PIECE_SYM = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


def describe_square(board: chess.Board, sq: int) -> str:
    """Rich natural-language description for one square.

    Includes algebraic notation (e.g. "Bc1") alongside full piece name so the
    model learns to associate symbols with names.
    """
    piece = board.piece_at(sq)
    sq_name = chess.square_name(sq)
    col = lambda c: "white" if c == chess.WHITE else "black"  # noqa: E731

    white_attackers = sorted(board.attackers(chess.WHITE, sq))
    black_attackers = sorted(board.attackers(chess.BLACK, sq))

    _PIECE_VALUE = {
        chess.QUEEN: 9,
        chess.ROOK: 5,
        chess.BISHOP: 3,
        chess.KNIGHT: 3,
        chess.PAWN: 1,
        chess.KING: 0,
    }

    def _by_value(sq: int) -> int:
        p = board.piece_at(sq)
        return -_PIECE_VALUE.get(p.piece_type, 0) if p else 0

    def _summarize_attackers(squares: list[int], color: chess.Color, limit: int = 2) -> str:
        if not squares:
            return "none"
        names = []
        for attacker_sq in sorted(squares, key=_by_value)[:limit]:
            attacker = board.piece_at(attacker_sq)
            if attacker is not None:
                names.append(
                    f"{'white' if color == chess.WHITE else 'black'} "
                    f"{chess.piece_name(attacker.piece_type)} on {chess.square_name(attacker_sq)}"
                )
        return "; ".join(names)

    def _control_summary(squares: list[int], color: chess.Color) -> str:
        count = len(squares)
        side = "white" if color == chess.WHITE else "black"
        piece_word = "piece" if count == 1 else "pieces"
        if count == 0:
            return ""
        return (
            f"{count} {side} {piece_word} including {_summarize_attackers(squares, color, limit=1)}"
        )

    sq_color = "light" if chess.BB_LIGHT_SQUARES & chess.BB_SQUARES[sq] else "dark"

    if piece is None:
        desc = (
            f"square {sq_name}. occupancy: empty, {sq_color} square"
            f", {_file_type(board, sq)}, {_rank_type(board, sq)}"
        )
        pawn_dist = _nearest_pawn_on_file(board, sq)
        if pawn_dist:
            desc += f", {pawn_dist}"
        # Outpost for the side to move — can a friendly piece sit here unchallenged?
        if _is_outpost(board, sq, board.turn):
            stm_str = "white" if board.turn == chess.WHITE else "black"
            desc += f", outpost for {stm_str}"
    else:
        sym = _PIECE_SYM[piece.piece_type]
        # Bishop: prefix with square color complex (light-squared / dark-squared)
        piece_name = (
            f"{sq_color}-squared {chess.piece_name(piece.piece_type)}"
            if piece.piece_type == chess.BISHOP
            else chess.piece_name(piece.piece_type)
        )
        desc = (
            f"square {sq_name}. occupancy: {col(piece.color)} "
            f"{piece_name} ({sym}{sq_name}), {sq_color} square"
        )
        # board.attacks(sq) is turn-independent — correct for both sides
        mobility = len(board.attacks(sq))
        if mobility > 0:
            desc += f", controls {mobility} squares"
        if piece.piece_type == chess.PAWN:
            parts = _pawn_structure_parts(board, sq, piece.color)
            if parts:
                desc += f", {', '.join(parts)}"
        if piece.piece_type in (chess.ROOK, chess.QUEEN):
            desc += f", on {_file_type(board, sq)}, {_rank_type(board, sq)}"
        if board.is_pinned(piece.color, sq):
            desc += ", pinned to king"

    if piece is None:
        control_parts = [
            part
            for part in (
                _control_summary(white_attackers, chess.WHITE),
                _control_summary(black_attackers, chess.BLACK),
            )
            if part
        ]
        desc += (
            f", controlled by {' and '.join(control_parts)}"
            if control_parts
            else ", not controlled"
        )
    else:
        enemy = not piece.color
        friendly = piece.color
        enemy_atk = black_attackers if enemy == chess.BLACK else white_attackers
        friendly_def = white_attackers if friendly == chess.WHITE else black_attackers
        if enemy_atk:
            desc += f", attacked by {_summarize_attackers(enemy_atk, enemy)}"
        if friendly_def:
            desc += f", defended by {_summarize_attackers(friendly_def, friendly)}"
        xray = _xray_target(board, sq)
        if xray:
            desc += f", {xray}"

    return desc


# ---------------------------------------------------------------------------
# Embedding cache + Qwen inference
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """LRU dict: text → (D,) CPU tensor matching the model embedding dtype."""

    def __init__(self, maxsize: int = 50_000, dtype: torch.dtype = torch.bfloat16) -> None:
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._maxsize = maxsize
        self._dtype = dtype
        self.hits = 0
        self.misses = 0
        self._save_thread: threading.Thread | None = None

    def get(self, key: str) -> torch.Tensor | None:
        val = self._cache.get(key)
        if val is not None:
            self._cache.move_to_end(key)  # promote to MRU end
            self.hits += 1
            return val
        self.misses += 1
        return None

    def put(self, key: str, value: torch.Tensor) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)  # promote existing entry
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)  # evict LRU (oldest) end
            self._cache[key] = value.to(dtype=self._dtype, device="cpu")

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._cache)

    def save(self, path: str | Path) -> None:
        """Save cache to disk in a background thread (non-blocking).

        Joins any in-flight save first so we never hold two full snapshots
        simultaneously (each snapshot is ~20 GB at 4 M entries).
        """
        if self._save_thread and self._save_thread.is_alive():
            self._save_thread.join()  # wait for previous save — don't double-buffer
        # Take a lightweight view of current keys/values without copying tensors.
        # list() snapshots the item pairs at this point in time; the tensors
        # themselves are not duplicated (bfloat16 CPU, referenced not copied).
        snapshot = list(self._cache.items())

        def _write() -> None:
            tmp = Path(str(path) + ".tmp")
            torch.save(snapshot, tmp)
            tmp.replace(path)

        self._save_thread = threading.Thread(target=_write, daemon=True)
        self._save_thread.start()

    def load(self, path: str | Path) -> int:
        """Load cache from disk, merging into existing entries.

        Safe to call multiple times — new entries are inserted via put() so
        maxsize is respected and LRU order is maintained. If text labels change
        between runs, old keys simply become dead weight and are evicted
        naturally by the LRU; new descriptions get fresh embeddings.

        Returns number of entries loaded.
        """
        path = Path(path)
        if not path.exists():
            return 0
        raw = torch.load(path, map_location="cpu", weights_only=True)
        # Support both old dict format and new list-of-pairs format.
        items: list[tuple[str, torch.Tensor]] = raw.items() if isinstance(raw, dict) else raw
        for k, v in items:
            self.put(k, v)  # respects maxsize + LRU order
        logger.info("EmbeddingCache: loaded %d entries from %s", len(self), path)
        return len(self)


@torch.no_grad()
def embed_texts(
    qwen: nn.Module,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    """Embed texts using Qwen last-layer last-token hidden state.

    Returns:
        (N, D) float32 tensor on device.
    """
    all_embs: list[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=96,
            # 96 covers grid (max=58) and global (max=89 for imbalance-only material + all castling + ep).
        ).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = qwen(**enc, use_cache=False)
        last_layer = out.last_hidden_state  # (B, L, D)
        seq_lens = enc.attention_mask.sum(dim=1) - 1  # last non-padding token index
        embs = last_layer[torch.arange(len(chunk), device=device), seq_lens].detach()
        all_embs.append(embs)
        del out, last_layer, enc  # free full hidden states immediately; don't wait for loop end
    return torch.cat(all_embs, dim=0)


def _all_gather_cat(tensor: torch.Tensor, *, require_grad: bool) -> torch.Tensor:
    """Gather first-dimension batches across ranks, preserving gradients when requested."""
    if not dist.is_initialized():
        return tensor
    if require_grad:
        return torch.cat(dist_nn.all_gather(tensor), dim=0)

    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def maybe_gather_negatives(
    tensor: torch.Tensor,
    *,
    require_grad: bool,
    enabled: bool,
) -> torch.Tensor:
    """Optionally gather negatives across ranks while keeping local tensors unchanged otherwise."""
    if not enabled:
        return tensor
    return _all_gather_cat(tensor, require_grad=require_grad)


def get_anchor_embeddings(
    descriptions: list[str],
    cache: EmbeddingCache,
    qwen: nn.Module,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Retrieve embeddings with dedup + LRU caching. Returns (N, D) on device."""
    unique_descs: list[str] = []
    desc_to_uid: dict[str, int] = {}
    for d in descriptions:
        if d not in desc_to_uid:
            desc_to_uid[d] = len(unique_descs)
            unique_descs.append(d)

    unique_embs: dict[str, torch.Tensor] = {}
    uncached = []
    for d in unique_descs:
        hit = cache.get(d)
        if hit is not None:
            unique_embs[d] = hit.to(device)
        else:
            uncached.append(d)

    if uncached:
        new_embs = embed_texts(qwen, tokenizer, uncached, device)
        for d, emb in zip(uncached, new_embs):
            cache.put(d, emb)
            unique_embs[d] = emb

    D = next(iter(unique_embs.values())).shape[-1]
    result = torch.zeros(len(descriptions), D, device=device)
    for i, d in enumerate(descriptions):
        result[i] = unique_embs[d]
    return result


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _infonce(v: torch.Tensor, t: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE. v, t: (B, D) L2-normalised. tau: scalar."""
    logits = (v @ t.T) / tau
    labels = torch.arange(v.size(0), device=v.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def clip_loss(
    grid_tokens: torch.Tensor,
    grid_anchors: torch.Tensor,
    global_token: torch.Tensor,
    global_anchor: torch.Tensor,
    tau: torch.Tensor,
    global_weight: float,
    grid_chunk_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-position InfoNCE (64 × B×B) + global InfoNCE (B×B).

    Args:
        grid_tokens:   (B, 64, D)
        grid_anchors:  (B, 64, D)
        global_token:  (B, D)
        global_anchor: (B, D)
    Returns:
        total_loss, L_grid, L_global
    """
    grid_tokens = F.normalize(grid_tokens, dim=-1)
    grid_anchors = F.normalize(grid_anchors, dim=-1)
    global_token = F.normalize(global_token, dim=-1)
    global_anchor = F.normalize(global_anchor, dim=-1)

    n_squares = grid_tokens.size(1)
    labels = torch.arange(grid_tokens.size(0), device=grid_tokens.device)
    L_grid = grid_tokens.new_zeros(())
    for start in range(0, n_squares, grid_chunk_size):
        end = min(start + grid_chunk_size, n_squares)
        tok_chunk = grid_tokens[:, start:end, :]
        anc_chunk = grid_anchors[:, start:end, :]
        logits = torch.einsum("bsd,csd->sbc", tok_chunk, anc_chunk) / tau
        chunk_labels = labels.unsqueeze(0).expand(end - start, -1).reshape(-1)
        v2t = F.cross_entropy(logits.reshape(-1, logits.size(-1)), chunk_labels)
        t2v = F.cross_entropy(logits.transpose(1, 2).reshape(-1, logits.size(-1)), chunk_labels)
        L_grid = L_grid + (v2t + t2v) * (end - start) / 2
    L_grid = L_grid / n_squares

    L_global = _infonce(global_token, global_anchor, tau)
    return L_grid + global_weight * L_global, L_grid, L_global


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def setup_ddp() -> tuple[int, int, int]:
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
    else:
        rank, local_rank, world_size = 0, 0, 1
    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "-c", default="recipes-train/encoder-phase0/config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint.pt")
    parser.add_argument(
        "--reset-scheduler",
        action="store_true",
        help="Do not restore scheduler state on resume (resets LR to peak).",
    )
    args = parser.parse_args()

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    config = load_config(args.config)
    train_cfg = config.get("training", {})
    encoder_cfg = config.get("encoder", {})
    clip_cfg = config.get("clip", {})
    wandb_cfg = config.get("wandb", {})
    output_dir = Path(config.get("output_dir", "checkpoints/encoder-clip"))

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        if wandb_cfg.get("enabled"):
            import wandb

            wandb.init(
                project=wandb_cfg.get("project", "chess-tutor"),
                name=wandb_cfg.get("name", "encoder-phase0"),
                tags=wandb_cfg.get("tags", ["encoder-phase0"]),
                config=config,
            )

    global_loss_weight: float = clip_cfg.get("global_loss_weight", 0.5)
    cache_maxsize: int = clip_cfg.get("cache_maxsize", 50_000)
    grid_loss_chunk_size: int = clip_cfg.get("grid_loss_chunk_size", 8)
    cross_rank_negatives: bool = clip_cfg.get("cross_rank_negatives", False)
    qwen_model_name: str = clip_cfg.get("qwen_model", "Qwen/Qwen3.5-4B")

    # ── Dataset ───────────────────────────────────────────────────────────────
    num_workers = train_cfg.get("dataloader_num_workers", 4)
    train_ds = _make_dataset(train_cfg["train_file"], limit=train_cfg.get("train_limit", 0))
    eval_ds = None
    if train_cfg.get("eval_file"):
        eval_ds = _make_dataset(train_cfg["eval_file"], limit=train_cfg.get("eval_limit", 2_000))

    is_iterable = isinstance(train_ds, ChessClipJsonlDataset)
    if is_iterable:
        train_ds._rank = rank
        train_ds._world_size = world_size
        train_batch_size = train_cfg.get("per_device_train_batch_size", 256)
        train_sampler = None
        train_shuffle = False
    else:
        train_batch_size = train_cfg.get("per_device_train_batch_size", 256)
        train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
        train_shuffle = train_sampler is None

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        sampler=train_sampler,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=_collate,
    )
    eval_loader = None
    if eval_ds is not None:
        if isinstance(eval_ds, ChessClipJsonlDataset):
            eval_ds._rank = rank
            eval_ds._world_size = world_size
        eval_loader = DataLoader(
            eval_ds,
            batch_size=train_cfg.get("per_device_eval_batch_size", 64),
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=_collate,
        )

    # ── Qwen (frozen text tower) ──────────────────────────────────────────────
    if is_main:
        logger.info("Loading Qwen: %s", qwen_model_name)
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    qwen = AutoModel.from_pretrained(
        qwen_model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(device)
    qwen.eval()
    for p in qwen.parameters():
        p.requires_grad_(False)
    # ── Encoder (trainable) ───────────────────────────────────────────────────
    encoder = ChessEncoder(
        in_channels=encoder_cfg.get("in_channels", 19),
        hidden_size=encoder_cfg.get("hidden_size", 256),
        num_blocks=encoder_cfg.get("num_blocks", 10),
        out_dim=encoder_cfg.get("out_dim", 2560),
    ).to(device)

    # Learnable log-temperature (CLIP convention; exp clamped to [0.01, 0.5])
    log_temperature = nn.Parameter(
        torch.tensor(math.log(clip_cfg.get("temperature_init", 0.07)), device=device)
    )

    resume_step = 0
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        if "log_temperature" in ckpt:
            log_temperature.data = ckpt["log_temperature"].to(device)
        resume_step = ckpt.get("step", 0)
        if is_main:
            logger.info("Resumed from %s (step %d)", args.resume, resume_step)

    if world_size > 1:
        encoder = DDP(encoder, device_ids=[local_rank])

    lr = train_cfg.get("learning_rate", 1e-4)
    trainable_params = list(encoder.parameters()) + [log_temperature]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=train_cfg.get("weight_decay", 0.01)
    )

    max_steps = train_cfg.get("max_steps", 16_000)
    warmup_steps = train_cfg.get("warmup_steps", 500)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if ckpt is not None:
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt and not args.reset_scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        if is_iterable:
            train_ds._resume_examples = ckpt.get(
                "samples_seen",
                resume_step * train_batch_size * world_size,
            )

    emb_cache = EmbeddingCache(maxsize=cache_maxsize, dtype=qwen.dtype)
    cache_path = output_dir / "embedding_cache.pt"
    n_loaded = emb_cache.load(cache_path)
    if n_loaded:
        logger.info("[rank%d] Restored %d cached embeddings from %s", rank, n_loaded, cache_path)

    logging_steps = train_cfg.get("logging_steps", 100)
    save_steps = train_cfg.get("save_steps", 2_000)
    eval_steps_cfg = train_cfg.get("eval_steps", 2_000)
    save_total_limit = train_cfg.get("save_total_limit", 5)
    max_grad_norm = train_cfg.get("max_grad_norm", 1.0)

    global_step = resume_step
    saved_checkpoints: list[Path] = []

    def save_checkpoint(step: int, eval_loss: float | None = None) -> None:
        nonlocal saved_checkpoints
        ckpt_dir = output_dir / f"checkpoint-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        raw_enc = encoder.module if isinstance(encoder, DDP) else encoder
        torch.save(
            {
                "encoder": raw_enc.state_dict(),
                "log_temperature": log_temperature.data,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "samples_seen": step * train_batch_size * world_size,
                "eval_loss": eval_loss,
            },
            ckpt_dir / "checkpoint.pt",
        )
        torch.save(raw_enc.state_dict(), ckpt_dir / "encoder_weights.pt")
        if is_main:
            logger.info("Saved checkpoint → %s", ckpt_dir)
        saved_checkpoints.append(ckpt_dir)
        if save_total_limit and len(saved_checkpoints) > save_total_limit:
            import shutil

            shutil.rmtree(saved_checkpoints.pop(0), ignore_errors=True)

    def build_anchors(
        fens: list[str], sf15_t: torch.Tensor, eval_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, 64, D) grid anchors + (B, D) global anchors."""
        B = len(fens)
        grid_descs: list[str] = []
        boards = [chess.Board(fen) for fen in fens]
        for board in boards:
            for sq in range(64):
                grid_descs.append(describe_square(board, sq))
        global_descs = [
            build_global_description(sf15_t[i].tolist(), eval_t[i].item(), boards[i])
            for i in range(B)
        ]
        all_embs = get_anchor_embeddings(
            grid_descs + global_descs, emb_cache, qwen, tokenizer, device
        )
        return all_embs[: B * 64].reshape(B, 64, -1), all_embs[B * 64 :]

    def run_eval() -> dict[str, float]:
        if eval_loader is None:
            return {}
        raw_enc = encoder.module if isinstance(encoder, DDP) else encoder
        raw_enc.eval()
        totals = {
            "loss": 0.0,
            "grid": 0.0,
            "global": 0.0,
            "top1_grid": 0.0,
            "top1_global": 0.0,
            "n": 0.0,
        }
        tau = log_temperature.exp().clamp(0.01, 0.5)
        with torch.no_grad():
            for boards, sf15, eval_score, fens in eval_loader:
                boards = boards.to(device, non_blocking=True)
                sf15_dev = sf15.to(device, non_blocking=True)
                eval_dev = eval_score.to(device, non_blocking=True)
                grid_anchors, global_anchor = build_anchors(fens, sf15_dev, eval_dev)
                tokens = encoder(boards)
                grid_tokens = tokens[:, :64, :]
                global_token = tokens[:, 64:, :]
                grid_tokens = maybe_gather_negatives(
                    grid_tokens,
                    require_grad=False,
                    enabled=cross_rank_negatives,
                )
                global_token = maybe_gather_negatives(
                    global_token,
                    require_grad=False,
                    enabled=cross_rank_negatives,
                )
                grid_anchors = maybe_gather_negatives(
                    grid_anchors,
                    require_grad=False,
                    enabled=cross_rank_negatives,
                )
                global_anchor = maybe_gather_negatives(
                    global_anchor,
                    require_grad=False,
                    enabled=cross_rank_negatives,
                )
                loss, L_grid, L_global = clip_loss(
                    grid_tokens,
                    grid_anchors,
                    global_token.squeeze(1),
                    global_anchor,
                    tau,
                    global_loss_weight,
                    grid_chunk_size=grid_loss_chunk_size,
                )
                bs = boards.size(0)
                totals["loss"] += loss.item() * bs
                totals["grid"] += L_grid.item() * bs
                totals["global"] += L_global.item() * bs
                totals["n"] += bs
                # Top-1 retrieval (tau-agnostic)
                _lbl = torch.arange(grid_tokens.size(0), device=grid_tokens.device)
                _gn = F.normalize(grid_tokens, dim=-1)
                _an = F.normalize(grid_anchors, dim=-1)
                _t1g = 0.0
                for _sq in range(grid_tokens.size(1)):
                    _s = _gn[:, _sq, :] @ _an[:, _sq, :].T
                    _t1g += (_s.argmax(dim=1) == _lbl).float().mean().item()
                totals["top1_grid"] += (_t1g / grid_tokens.size(1)) * bs
                _gn_g = F.normalize(global_token.squeeze(1), dim=-1)
                _an_g = F.normalize(global_anchor, dim=-1)
                totals["top1_global"] += (_gn_g @ _an_g.T).argmax(dim=1).eq(
                    _lbl
                ).float().mean().item() * bs
        raw_enc.train()
        if dist.is_initialized():
            t = torch.tensor(list(totals.values()), dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            for i, k in enumerate(totals):
                totals[k] = t[i].item()
        n = totals["n"]
        return (
            {k: totals[k] / n for k in ("loss", "grid", "global", "top1_grid", "top1_global")}
            if n > 0
            else {}
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    if is_main:
        raw_enc = encoder.module if isinstance(encoder, DDP) else encoder
        enc_params = sum(p.numel() for p in raw_enc.parameters())
        logger.info(
            "Chess-CLIP | encoder=%s params | max_steps=%d | warmup=%d | lr=%.2e | eff_batch=%d",
            f"{enc_params:,}",
            max_steps,
            warmup_steps,
            lr,
            train_cfg.get("per_device_train_batch_size", 256) * world_size,
        )

    encoder.train()
    t0 = time.time()
    running = {
        "loss": 0.0,
        "grid": 0.0,
        "global": 0.0,
        "top1_grid": 0.0,
        "top1_global": 0.0,
        "n": 0,
    }

    for boards, sf15, eval_score, fens in train_loader:
        if global_step >= max_steps:
            break

        boards = boards.to(device, non_blocking=True)
        sf15_dev = sf15.to(device, non_blocking=True)
        eval_dev = eval_score.to(device, non_blocking=True)

        grid_anchors, global_anchor = build_anchors(fens, sf15_dev, eval_dev)

        optimizer.zero_grad()
        tokens = encoder(boards)
        grid_tokens = tokens[:, :64, :]
        global_token = tokens[:, 64:, :]
        grid_tokens = maybe_gather_negatives(
            grid_tokens,
            require_grad=True,
            enabled=cross_rank_negatives,
        )
        global_token = maybe_gather_negatives(
            global_token,
            require_grad=True,
            enabled=cross_rank_negatives,
        )
        grid_anchors = maybe_gather_negatives(
            grid_anchors,
            require_grad=False,
            enabled=cross_rank_negatives,
        )
        global_anchor = maybe_gather_negatives(
            global_anchor,
            require_grad=False,
            enabled=cross_rank_negatives,
        )

        tau = log_temperature.exp().clamp(0.01, 0.5)
        loss, L_grid, L_global = clip_loss(
            grid_tokens,
            grid_anchors,
            global_token.squeeze(1),
            global_anchor,
            tau,
            global_loss_weight,
            grid_chunk_size=grid_loss_chunk_size,
        )

        loss.backward()
        # Sync log_temperature gradient across DDP ranks (not wrapped in DDP)
        if dist.is_initialized() and log_temperature.grad is not None:
            dist.all_reduce(log_temperature.grad, op=dist.ReduceOp.AVG)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()
        scheduler.step()
        global_step += 1

        bs = boards.size(0)
        running["loss"] += loss.item() * bs
        running["grid"] += L_grid.item() * bs
        running["global"] += L_global.item() * bs
        running["n"] += bs

        # Top-1 retrieval accuracy — tau-agnostic, computed from raw cosine sims
        with torch.no_grad():
            _lbl = torch.arange(grid_tokens.size(0), device=grid_tokens.device)
            _gn = F.normalize(grid_tokens.detach(), dim=-1)
            _an = F.normalize(grid_anchors.detach(), dim=-1)
            # Mean top-1 over all 64 squares: for each sq (B,D)@(D,B)→(B,B), argmax==diag
            _top1_grid = 0.0
            for _sq in range(grid_tokens.size(1)):
                _s = _gn[:, _sq, :] @ _an[:, _sq, :].T  # (B, B)
                _top1_grid += (_s.argmax(dim=1) == _lbl).float().mean().item()
            _top1_grid /= grid_tokens.size(1)
            _gn_g = F.normalize(global_token.squeeze(1).detach(), dim=-1)
            _an_g = F.normalize(global_anchor.detach(), dim=-1)
            _top1_global = (_gn_g @ _an_g.T).argmax(dim=1).eq(_lbl).float().mean().item()
        running["top1_grid"] += _top1_grid * bs
        running["top1_global"] += _top1_global * bs

        if is_main and global_step % logging_steps == 0:
            n = running["n"]
            avg = {
                k: running[k] / n for k in ("loss", "grid", "global", "top1_grid", "top1_global")
            }
            lr_now = scheduler.get_last_lr()[0]
            logger.info(
                "step=%d  loss=%.4f grid=%.4f global=%.4f  tau=%.4f  lr=%.2e  "
                "top1=%.3f/%.3f  cache_hit=%.1f%%  %.1fs",
                global_step,
                avg["loss"],
                avg["grid"],
                avg["global"],
                tau.item(),
                lr_now,
                avg["top1_grid"],
                avg["top1_global"],
                emb_cache.hit_rate * 100,
                time.time() - t0,
            )
            if wandb_cfg.get("enabled"):
                import wandb

                wandb.log(
                    {
                        "train/loss": avg["loss"],
                        "train/loss_grid": avg["grid"],
                        "train/loss_global": avg["global"],
                        "train/temperature": tau.item(),
                        "train/lr": lr_now,
                        "train/top1_grid": avg["top1_grid"],
                        "train/top1_global": avg["top1_global"],
                        "train/cache_hit_rate": emb_cache.hit_rate,
                    },
                    step=global_step,
                )
            running = {
                "loss": 0.0,
                "grid": 0.0,
                "global": 0.0,
                "top1_grid": 0.0,
                "top1_global": 0.0,
                "n": 0,
            }
            emb_cache.reset_stats()  # keep counters small
            t0 = time.time()

        if global_step % eval_steps_cfg == 0:
            # All ranks participate in eval (DDP all_reduce inside run_eval).
            metrics = run_eval()
            if is_main and metrics:
                logger.info(
                    "step=%d  eval loss=%.4f grid=%.4f global=%.4f  top1=%.3f/%.3f",
                    global_step,
                    metrics.get("loss", 0),
                    metrics.get("grid", 0),
                    metrics.get("global", 0),
                    metrics.get("top1_grid", 0),
                    metrics.get("top1_global", 0),
                )
                if wandb_cfg.get("enabled"):
                    import wandb

                    wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)

        if is_main and global_step % save_steps == 0:
            save_checkpoint(global_step)
            emb_cache.save(cache_path)
            logger.info("Saving embedding cache (%d entries) → %s", len(emb_cache), cache_path)

    if is_main:
        save_checkpoint(global_step)
        emb_cache.save(cache_path)
        logger.info("Training complete at step %d.", global_step)

    cleanup_ddp()


if __name__ == "__main__":
    main()
