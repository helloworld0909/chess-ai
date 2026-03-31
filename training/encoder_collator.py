"""Data collator for ChessLMWithEncoder — board tensor preparation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chess
import torch
from transformers import PreTrainedTokenizerBase

from src.encoder import BOARD_TOKEN_ID
from src.encoder.board_tensor import board_to_tensor

_logger = logging.getLogger(__name__)


@dataclass
class EncoderDataCollator:
    """Collates token IDs and computes board tensors for every board position.

    Each training example contains one board position represented by
    BOARD_TOKENS_PER_POSITION (64) consecutive <|vision_pad|> sentinel tokens
    in the input. The collator builds one (19, 8, 8) board tensor per board
    position; the CNN forward pass expands each tensor to 64 per-square tokens.

    The collator produces:
      batch["board_tensors_flat"]: (N_boards, 19, 8, 8) — one tensor per board
          position in the batch, concatenated in example order.
      batch["move_counts"]: (B,) — number of board positions per example (always 1
          for board-reading SFT; may be >1 for phase2 coaching with key lines).

    For phase2 coaching compatibility: if line_sans_json is non-empty, additional
    tensors encode the pre-move board for each move in each line.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: Optional[int] = None

    def __post_init__(self) -> None:
        self._board_token_id: int = BOARD_TOKEN_ID

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Pop non-tensor fields before HF padding (unknown keys cause errors)
        fens: List[str] = []
        move_sans: List[str] = []
        line_sans_lists: List[List[List[str]]] = []
        labels_list: List[List[int]] = []

        for feat in features:
            fens.append(feat.pop("fen", chess.STARTING_FEN))
            move_sans.append(feat.pop("move_san", ""))
            raw = feat.pop("line_sans_json", "[]")
            try:
                line_sans_lists.append(json.loads(raw))
            except Exception:
                line_sans_lists.append([])
            # Pop labels; tokenizer.pad() doesn't handle them correctly
            lbl = feat.pop("labels", None)
            if lbl is not None:
                labels_list.append(list(lbl))

        # Pad input_ids + attention_mask only
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Pad labels with -100 to match padded length
        if labels_list:
            max_len = batch["input_ids"].shape[1]
            padded_labels = [lbl + [-100] * (max_len - len(lbl)) for lbl in labels_list]
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        all_tensors: List[torch.Tensor] = []
        board_counts: List[int] = []

        for b_idx, (fen, student_san, line_sans) in enumerate(
            zip(fens, move_sans, line_sans_lists)
        ):
            tensors_for_example: List[torch.Tensor] = []

            board = chess.Board(fen)

            # First tensor: pre-move board (student move position)
            tensors_for_example.append(board_to_tensor(board))

            # Phase2 compatibility: one tensor per line move (pre-move board at each step)
            for line_san_list in line_sans:
                line_board = board.copy()
                for san in line_san_list:
                    tensors_for_example.append(board_to_tensor(line_board))
                    try:
                        mv = line_board.parse_san(san)
                        line_board.push(mv)
                    except Exception:
                        pass

            # Validate: number of board positions must match sentinel groups.
            from src.encoder import BOARD_TOKENS_PER_POSITION

            n_sentinel_tokens = (batch["input_ids"][b_idx] == self._board_token_id).sum().item()
            n_sentinel_groups = n_sentinel_tokens // BOARD_TOKENS_PER_POSITION
            n_tensors = len(tensors_for_example)
            if n_tensors != n_sentinel_groups:
                _logger.debug(
                    "example %d: %d board positions vs %d sentinel groups — adjusting",
                    b_idx,
                    n_tensors,
                    n_sentinel_groups,
                )
                while len(tensors_for_example) < n_sentinel_groups:
                    tensors_for_example.append(board_to_tensor(chess.Board(fen)))
                tensors_for_example = tensors_for_example[:n_sentinel_groups]

            board_counts.append(len(tensors_for_example))
            all_tensors.extend(tensors_for_example)

        batch["board_tensors_flat"] = (
            torch.stack(all_tensors) if all_tensors else torch.zeros(0, 19, 8, 8)
        )
        batch["move_counts"] = torch.tensor(board_counts, dtype=torch.long)

        return batch
