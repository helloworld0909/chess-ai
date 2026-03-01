"""Inference utilities for ChessLMWithEncoder.

Loads the full model from a Trainer.save_model() checkpoint and provides
streaming token generation with board tensor injection.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Generator

import chess
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.encoder import MOVE_TOKEN, MOVE_TOKEN_ID
from src.encoder.board_tensor import boards_to_tensor
from training.lib import load_config

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_encoder_model(checkpoint_dir: str, config_path: str) -> tuple:
    """Load ChessLMWithEncoder from a Trainer.save_model() checkpoint.

    Args:
        checkpoint_dir: Path to directory containing model.safetensors
        config_path: Path to the training config YAML (for architecture params)

    Returns:
        (model, tokenizer) — model in eval mode on CUDA
    """
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from training.encoder_model import ChessLMWithEncoder

    config = load_config(config_path)
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    encoder_cfg = config.get("encoder", {})
    model_name = model_cfg["model_name"]

    _logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for generation

    _logger.info("Loading base LLM (8-bit)...")
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    peft_llm = get_peft_model(base_llm, lora_config)

    _logger.info("Wrapping with CNN encoder...")
    model = ChessLMWithEncoder(
        llm=peft_llm,
        hidden_size=base_llm.config.hidden_size,
        cnn_hidden_size=encoder_cfg.get("hidden_size", 512),
        cnn_num_blocks=encoder_cfg.get("num_blocks", 15),
        move_token_id=MOVE_TOKEN_ID,
    )

    _logger.info("Loading checkpoint from %s", checkpoint_dir)
    state_dict = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        _logger.warning("Missing keys: %s", missing[:5])
    if unexpected:
        _logger.warning("Unexpected keys: %s", unexpected[:5])

    model.eval()
    _logger.info("Model ready.")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Board tensor construction
# ---------------------------------------------------------------------------


def build_board_tensors(
    fen: str,
    move_san: str,
    key_lines: list[list[str]],
) -> torch.Tensor:
    """Build all board tensors for a phase 1 SFT prompt.

    Order matches the left-to-right order of <|vision_pad|> sentinels:
      [student_move_tensor, line0_move0, line0_move1, ..., line1_move0, ...]

    Args:
        fen: Position FEN before the student's move
        move_san: Student's move in SAN notation
        key_lines: list of lines, each a list of SAN strings
                   e.g. [["Nd5", "e4", "Nc3"], ["Nf6", "g5"]]

    Returns:
        Tensor of shape (N, 38, 8, 8)
    """
    tensors: list[torch.Tensor] = []

    board = chess.Board(fen)

    # 1. Student move tensor
    student_move: chess.Move | None = None
    if move_san:
        try:
            student_move = board.parse_san(move_san)
        except Exception:
            pass
    tensors.append(boards_to_tensor(board, student_move))

    # 2. Key line tensors — replay from pre-move board
    for line_sans in key_lines:
        line_board = board.copy()
        for san in line_sans:
            try:
                mv = line_board.parse_san(san)
                tensors.append(boards_to_tensor(line_board, mv))
                line_board.push(mv)
            except Exception:
                tensors.append(boards_to_tensor(line_board, None))

    return torch.stack(tensors)  # (N, 38, 8, 8)


# ---------------------------------------------------------------------------
# Prompt injection (plain SAN → sentinel tokens)
# ---------------------------------------------------------------------------


def inject_sentinels(messages: list[dict], move_san: str) -> list[dict]:
    """Inject <|vision_pad|> sentinels into a joint-format message list.

    - User turn: replace student move SAN and each key-line SAN with sentinel
    - Assistant turn: untouched
    """
    new_msgs = []
    for msg in messages:
        content = msg["content"]
        role = msg["role"]

        if role == "user":
            # Inject student move sentinel
            if move_san:
                content = re.sub(
                    r"(?<=Move:\s)" + re.escape(move_san) + r"(?=\s|$|\n)",
                    MOVE_TOKEN,
                    content,
                )

            # Inject key-line sentinels
            if "## Engine Key Lines" in content:

                def _replace_key_lines(m: re.Match) -> str:
                    inner = m.group(1)
                    new_lines = []
                    for line in inner.split("\n"):
                        lm = re.match(r"^(Line \d+:\s*)(.*)", line)
                        if lm:
                            prefix = lm.group(1)
                            moves_str = lm.group(2)
                            moves = moves_str.split("→")
                            injected = []
                            for move in moves:
                                move = move.strip()
                                if not move.startswith(MOVE_TOKEN):
                                    injected.append(MOVE_TOKEN + move)
                                else:
                                    injected.append(move)
                            new_lines.append(prefix + " → ".join(injected))
                        else:
                            new_lines.append(line)
                    return "## Engine Key Lines\n\n" + "\n".join(new_lines)

                content = re.sub(
                    r"## Engine Key Lines\n\n(.*?)(?=\n\n##|\Z)",
                    _replace_key_lines,
                    content,
                    flags=re.DOTALL,
                )

        new_msgs.append({"role": role, "content": content})
    return new_msgs


# ---------------------------------------------------------------------------
# Streaming generation
# ---------------------------------------------------------------------------


def generate_stream(
    model,
    tokenizer,
    messages: list[dict],
    board_tensors: torch.Tensor,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """Stream tokens from ChessLMWithEncoder.

    Injects board tensors at sentinel positions then generates autoregressively.
    Yields decoded text chunks as they are produced.
    """
    import threading

    from transformers import TextIteratorStreamer

    device = next(model.parameters()).device

    # Tokenize
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    board_tensors = board_tensors.to(device)

    # Validate sentinel count
    n_sentinels = (input_ids == MOVE_TOKEN_ID).sum().item()
    n_tensors = board_tensors.shape[0]
    if n_sentinels != n_tensors:
        _logger.warning("Sentinel count mismatch: %d tokens vs %d tensors", n_sentinels, n_tensors)
        # Trim or pad to match
        if n_tensors > n_sentinels:
            board_tensors = board_tensors[:n_sentinels]
        else:
            pad = torch.zeros(
                n_sentinels - n_tensors,
                *board_tensors.shape[1:],
                dtype=board_tensors.dtype,
                device=device,
            )
            board_tensors = torch.cat([board_tensors, pad], dim=0)

    # Build inputs_embeds with CNN embeddings spliced in
    with torch.no_grad():
        cnn_dtype = next(model.cnn.parameters()).dtype
        embed_dtype = model.embed_tokens.weight.dtype

        cnn_embs = model.cnn(board_tensors.to(dtype=cnn_dtype)).to(dtype=embed_dtype)
        text_embs = model.embed_tokens(input_ids)

        B, L, H = text_embs.shape
        move_positions = (input_ids == MOVE_TOKEN_ID).nonzero(as_tuple=False)
        cnn_canvas = torch.zeros(B, L, H, dtype=embed_dtype, device=device)
        if move_positions.shape[0] > 0:
            b_idx = move_positions[:, 0]
            l_idx = move_positions[:, 1]
            cnn_canvas = cnn_canvas.index_put((b_idx, l_idx), cnn_embs, accumulate=False)

        move_mask = (input_ids == MOVE_TOKEN_ID).unsqueeze(-1).expand(B, L, H)
        inputs_embeds = torch.where(move_mask, cnn_canvas, text_embs)

    # Stream generation
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
    )

    thread = threading.Thread(target=lambda: model.llm.generate(**gen_kwargs), daemon=True)
    thread.start()

    for text in streamer:
        yield text

    thread.join()
