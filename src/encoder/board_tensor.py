"""Convert chess.Board into AlphaZero/Leela-style tensors.

Single-board tensor — (19, 8, 8):
  0-5:   White pieces (P N B R Q K)
  6-11:  Black pieces (P N B R Q K)
  12:    Side to move (all-1 if White, all-0 if Black)
  13:    White kingside castling right
  14:    White queenside castling right
  15:    Black kingside castling right
  16:    Black queenside castling right
  17:    En passant file (1 in the EP file column)
  18:    Move number (normalized to [0,1], assuming max ~200)
"""

from __future__ import annotations

import chess
import torch


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Convert a python-chess board into a (19, 8, 8) float32 tensor."""
    tensor = torch.zeros((19, 8, 8), dtype=torch.float32)

    # 1. Piece positions (Channels 0-11)
    # python-chess squares go 0 (a1) to 63 (h8).
    # We want rank=0 to be the 1st rank (a1-h1), rank=7 to be 8th rank.
    # We can just iterate all 64 squares.
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            # White: 0-5, Black: 6-11
            color_offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = piece.piece_type - 1  # 1=PAWN -> 0, 6=KING -> 5
            channel = color_offset + piece_idx

            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            # board viewed from white's perspective: rank 0 is bottom
            tensor[channel, rank, file] = 1.0

    # 2. Side to move (Channel 12)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # 3. Castling rights (Channels 13-16)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # 4. En passant (Channel 17)
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        tensor[17, :, ep_file] = 1.0

    # 5. Move number (Channel 18)
    # Fullmove number ranges from 1 to roughly 200. We divide by 200 to normalize.
    move_num = min(board.fullmove_number, 200) / 200.0
    tensor[18, :, :] = move_num

    return tensor
