"""Puzzle GRPO training — Lichess tactical puzzles with SGLang rollouts.

Architecture:
  GPU 0 — TRL GRPOTrainer: policy (ChessLMWithEncoder + LoRA), optimizer, loss
  GPU 1 — SGLang server: fast rollout generation with CNN board processing

Each batch:
  1. rollout_func: POST to SGLang /generate → completions + token ids + logprobs
  2. PuzzleGRPOTrainer._compute_loss: stash board tensors, call super()
  3. Reward: gated (format → -1.0; wrong → 0.0; correct → 1.0) + length penalty
  4. Every sync_every_n_steps: merge LoRA → push to SGLang → unmerge

Usage:
    ./recipes-train/qwen3.5-4b-encoder-phase2-puzzle-grpo/start.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import chess
import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.encoder import BOARD_TOKEN, BOARD_TOKEN_ID, BOARD_TOKENS_PER_POSITION
from src.encoder.board_tensor import board_to_tensor
from src.model.encoder_model import ChessLMWithEncoder
from src.verification.puzzle_rewards import compute_rewards

_PIECE_NAMES = {
    chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
    chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king",
}


def _describe_pieces(board: chess.Board) -> str:
    """List all pieces by color and square — same format as board reading SFT."""
    white, black = [], []
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        name = _PIECE_NAMES[piece.piece_type]
        entry = f"{name} on {chess.SQUARE_NAMES[sq]}"
        (white if piece.color == chess.WHITE else black).append(entry)
    lines = []
    if white:
        lines.append("White: " + ", ".join(white) + ".")
    if black:
        lines.append("Black: " + ", ".join(black) + ".")
    return "\n".join(lines)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_SGLANG_URL: str = "http://localhost:8300"
_COMPLETIONS_LOG: str = "/tmp/puzzle-grpo-completions.jsonl"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# SGLang helpers
# ---------------------------------------------------------------------------


def _post_json(url: str, payload: dict, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _sglang_health(url: str) -> bool:
    try:
        urllib.request.urlopen(f"{url}/health", timeout=5)
        return True
    except Exception:
        return False


_SYNC_DIR = "/dev/shm/chess-grpo-sync"


def _sync_lora_to_sglang(model: ChessLMWithEncoder, url: str, step: int) -> None:
    """Merge LoRA into base weights, write LoRA-affected params to /dev/shm,
    then tell SGLang to reload from that path via /update_weights_from_disk.

    /dev/shm is a RAM-backed tmpfs — write is fast (~1-2s for ~340MB of LoRA params).
    This works across GPU 0 (trainer) → GPU 1 (SGLang) without CUDA IPC constraints.
    """
    import shutil

    from safetensors.torch import save_file

    t0 = time.time()
    log.info("[step=%d] Syncing LoRA weights to SGLang via /dev/shm...", step)

    sync_dir = Path(_SYNC_DIR)
    sync_dir.mkdir(parents=True, exist_ok=True)

    # Collect LoRA target base parameter names (before merge)
    lora_target_names: set[str] = set()
    for name, _ in model.llm.named_parameters():
        if "lora_A" in name:
            base = name.replace(".lora_A.default.weight", ".weight")
            base = base.replace("base_model.model.", "")
            lora_target_names.add(base)

    model.llm.merge_adapter()
    try:
        # Extract only LoRA-modified weights (not full 4B model)
        lora_state: dict[str, Any] = {}
        for name, param in model.llm.named_parameters():
            clean = name.replace("base_model.model.", "")
            if clean in lora_target_names:
                lora_state[clean] = param.detach().cpu().contiguous()

        log.info("[step=%d] Saving %d LoRA-merged tensors to %s", step, len(lora_state), sync_dir)
        # Write as safetensors shard (SGLang can load partial state dicts with load_format="auto")
        save_file(lora_state, sync_dir / "model.safetensors")
    finally:
        model.llm.unmerge_adapter()

    # Also copy the config so SGLang can identify the model
    src_model_dir = Path("/tmp/chess-merged")
    for fname in ("config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"):
        src = src_model_dir / fname
        dst = sync_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # Tell SGLang to reload from /dev/shm
    _post_json(
        f"{url}/update_weights_from_disk",
        {"model_path": str(sync_dir), "load_format": "auto"},
    )
    log.info("[step=%d] SGLang weight sync complete (%.1fs)", step, time.time() - t0)
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


PUZZLE_SYSTEM_PROMPT = (
    "You are a chess assistant. The board position is encoded as a sequence of vision tokens "
    "wrapped in <board> </board> tags. Use them to identify pieces and answer questions about the position."
)


def build_messages(fen: str, themes: list[str], color: str) -> list[dict]:
    """Two-turn prompt: pre-filled piece list, then ask for the move.

    The piece list is computed from FEN so the model starts generation
    directly at tactical analysis — no board-reading tokens wasted.
    """
    piece_list = _describe_pieces(chess.Board(fen))
    return [
        {"role": "system", "content": PUZZLE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"<board>{BOARD_TOKEN}</board>\n\n"
                f"It's {color}'s turn. List all pieces on the board with their squares."
            ),
        },
        {"role": "assistant", "content": piece_list},
        {
            "role": "user",
            "content": (
                f"Find the best move for {color}. "
                f"Every puzzle has a forcing solution — check for checks, captures, and threats.\n"
                f"Trust the piece list above. Be concise — think in 2-3 lines, then answer.\n\n"
                f'Output your answer as a JSON object on the last line. Example: {{"move": "Nf3"}}'
            ),
        },
    ]


class PuzzleDataset(Dataset):
    """Load puzzle_grpo_train.jsonl — flat metadata records, prompts built dynamically."""

    def __init__(self, jsonl_path: str, tokenizer, max_prompt_length: int) -> None:
        self.rows: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fen = rec.get("fen", chess.STARTING_FEN)
                solution_uci = rec.get("solution_uci", "")
                themes = rec.get("themes", [])
                color = rec.get("color", "White")
                if not solution_uci:
                    continue
                self.rows.append(
                    {
                        "fen": fen,
                        "solution_uci": solution_uci,
                        "themes": themes,
                        "color": color,
                    }
                )
        log.info("Loaded %d puzzle records from %s", len(self.rows), jsonl_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        messages = build_messages(row["fen"], row["themes"], row["color"])
        return {"prompt": messages, "fen": row["fen"], "solution_uci": row["solution_uci"]}


# ---------------------------------------------------------------------------
# Rollout function
# ---------------------------------------------------------------------------


def make_rollout_func(url: str, num_generations: int, max_new_tokens: int, temperature: float, sync_every: int, log_sample_every: int = 50, two_pass: bool = False):
    """Build the rollout_func closure for GRPOTrainer."""

    from src.verification.puzzle_rewards import compute_rewards, _extract_uci  # noqa: PLC0415

    def puzzle_rollout_func(prompts: list[dict], trainer: GRPOTrainer) -> dict:
        step = getattr(trainer.state, "global_step", 0)
        last_sync = getattr(trainer, "_last_sglang_sync", -1)

        if step > 0 and step != last_sync and step % sync_every == 0:
            _sync_lora_to_sglang(trainer.model, url, step)
            trainer._last_sglang_sync = step

        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        board_tensors: list[torch.Tensor] = []
        solution_ucis: list[str] = []
        all_fens: list[str] = []
        completions_text: list[str] = []

        t_rollout = time.time()
        n_failed = 0

        # TRL passes prompts = [x["prompt"] for x in inputs]; get full inputs from stash
        full_inputs = getattr(trainer, "_current_inputs", None) or []
        tokenizer = trainer.processing_class

        def _generate_one(args):
            i, messages = args
            inp = full_inputs[i] if i < len(full_inputs) else {}
            fen = inp.get("fen", "")
            solution_uci = inp.get("solution_uci", "")

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            payload: dict[str, Any] = {
                "text": prompt_text,
                "image_data": [{"format": "board_tensor", "fen": fen}],
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "n": 1,
                },
                "return_logprob": True,
                "logprob_start_len": -1,
            }
            try:
                resp = _post_json(f"{url}/generate", payload)
            except Exception as e:
                log.warning("SGLang generate failed for FEN %s: %s", fen, e)
                return i, None, fen, solution_uci, prompt_text

            if not two_pass:
                return i, resp if isinstance(resp, dict) else resp[0], fen, solution_uci, prompt_text

            # --- Two-pass: force JSON answer after </think> ---
            _r1 = resp if isinstance(resp, dict) else resp[0]
            think_text = _r1.get("text", "")
            think_ids = _r1.get("output_ids", [])
            think_lp_raw = _r1.get("meta_info", {}).get("output_token_logprobs", [])

            forced_prefix = '</think>\n{"move": "'
            prompt2 = prompt_text + think_text + forced_prefix
            payload2: dict[str, Any] = {
                "text": prompt2,
                "image_data": [{"format": "board_tensor", "fen": fen}],
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": 16,
                    "n": 1,
                },
                "return_logprob": True,
                "logprob_start_len": -1,
            }
            try:
                resp2 = _post_json(f"{url}/generate", payload2)
            except Exception as e:
                log.warning("SGLang pass2 failed for FEN %s: %s", fen, e)
                return i, None, fen, solution_uci, prompt_text

            _r2 = resp2 if isinstance(resp2, dict) else resp2[0]
            answer_text = _r2.get("text", "")
            answer_ids = _r2.get("output_ids", [])
            answer_lp_raw = _r2.get("meta_info", {}).get("output_token_logprobs", [])

            merged = {
                "text": think_text + forced_prefix + answer_text,
                "output_ids": think_ids + answer_ids,
                "meta_info": {"output_token_logprobs": think_lp_raw + answer_lp_raw},
            }
            return i, merged, fen, solution_uci, prompt_text

        # Send all prompts concurrently — SGLang batches them on GPU 1
        from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415
        from tqdm import tqdm  # noqa: PLC0415

        futures_map = {}
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            futures = {pool.submit(_generate_one, (i, m)): i for i, m in enumerate(prompts)}
            results_by_idx: dict[int, Any] = {}
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"[step={step}] rollout", leave=False):
                idx, resp, fen, solution_uci, prompt_text = fut.result()
                results_by_idx[idx] = (resp, fen, solution_uci, prompt_text)

        # Reassemble in original order
        for i in range(len(prompts)):
            if i not in results_by_idx:
                n_failed += 1
                continue
            resp, fen, solution_uci, prompt_text = results_by_idx[i]
            if resp is None:
                n_failed += 1
                continue

            # n=1: SGLang returns a single dict (not a list)
            completion = resp if isinstance(resp, dict) else resp[0]
            raw_ids = tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False
            ).input_ids[0].tolist()
            # SGLang's pad_input_ids expands each sentinel into BOARD_TOKENS_PER_POSITION
            # pad slots before the forward pass. Replicate that expansion here so the
            # trainer's forward pass sees the same 65-token sentinel block.
            prompt_ids = []
            for tok in raw_ids:
                if tok == BOARD_TOKEN_ID:
                    prompt_ids.extend([BOARD_TOKEN_ID] * BOARD_TOKENS_PER_POSITION)
                else:
                    prompt_ids.append(tok)
            board_tensor = board_to_tensor(chess.Board(fen))
            meta = completion.get("meta_info", {})
            text = completion.get("text", "")
            token_ids = completion.get("output_ids", [])
            raw_lp = meta.get("output_token_logprobs", [])
            lp = [entry[0] for entry in raw_lp if entry[0] is not None]
            all_prompt_ids.append(prompt_ids)
            all_completion_ids.append(token_ids)
            all_logprobs.append(lp)
            board_tensors.append(board_tensor)
            solution_ucis.append(solution_uci)
            all_fens.append(fen)
            completions_text.append(text)

        rollout_secs = time.time() - t_rollout

        # --- Observability ---
        if completions_text:
            token_counts = [len(ids) for ids in all_completion_ids]
            rewards = compute_rewards(completions_text, solution_ucis, token_counts, all_fens)

            n_total = len(rewards)
            predicted_list = [_extract_uci(t) for t in completions_text]
            n_no_format = sum(1 for p in predicted_list if p is None)
            n_correct = sum(1 for r in rewards if r >= 1.0)
            n_wrong = sum(1 for p, r in zip(predicted_list, rewards) if p is not None and r < 1.0)
            avg_reward = sum(rewards) / n_total
            avg_tokens = sum(token_counts) / n_total

            log.info(
                "[step=%d] rollout=%.1fs | %d completions (%d failed) | "
                "correct=%d (%.0f%%) wrong=%d no_fmt=%d | "
                "avg_reward=%.3f avg_tok=%.0f",
                step, rollout_secs, n_total, n_failed,
                n_correct, 100 * n_correct / n_total,
                n_wrong, n_no_format,
                avg_reward, avg_tokens,
            )

            # Write all completions to JSONL (full text)
            with open(_COMPLETIONS_LOG, "a") as _cf:
                for _i, _txt in enumerate(completions_text):
                    _cf.write(json.dumps({
                        "step": step,
                        "fen": all_fens[_i] if _i < len(all_fens) else "",
                        "solution": solution_ucis[_i] if _i < len(solution_ucis) else "",
                        "predicted": predicted_list[_i] if _i < len(predicted_list) else "",
                        "reward": rewards[_i],
                        "tokens": token_counts[_i],
                        "completion": _txt,
                    }) + "\n")

            # Log best completion to training log (last 500 chars)
            if completions_text:
                best_idx = max(range(n_total), key=lambda i: rewards[i])
                best_txt = completions_text[best_idx]
                log.info(
                    "[step=%d] BEST | solution=%s predicted=%s reward=%.2f tok=%d\n%s",
                    step, solution_ucis[best_idx], predicted_list[best_idx],
                    rewards[best_idx], token_counts[best_idx], best_txt[-500:],
                )

        # Stash for reward_fn and _compute_loss override
        trainer._pending_board_tensors = board_tensors
        trainer._pending_solution_ucis = solution_ucis
        trainer._pending_completion_ids = all_completion_ids
        trainer._pending_fens = all_fens

        completion_token_counts = [len(ids) for ids in all_completion_ids]
        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            # Extra fields — TRL merges these into inputs, reward_fn receives via **kwargs
            "solution_ucis": solution_ucis,
            "completion_tokens": completion_token_counts,
        }

    return puzzle_rollout_func


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


def make_reward_fn(solution_ucis_ref: list[str]):
    """Build a reward function that uses stashed solution UCIs from rollout."""

    def reward_fn(completions, **kwargs) -> list[float]:
        # TRL may pass completions as [[{"role": "assistant", "content": text}], ...]
        # for conversational datasets. Unwrap to plain strings.
        texts = []
        for c in completions:
            if isinstance(c, list):
                texts.append(next((m["content"] for m in c if m.get("role") == "assistant"), ""))
            else:
                texts.append(str(c))
        solution_ucis = kwargs.get("solution_ucis", solution_ucis_ref)
        completion_tokens = kwargs.get("completion_tokens", [0] * len(texts))
        log.info(
            "[reward_fn] completions=%d solution_ucis=%d completion_tokens=%d",
            len(texts), len(solution_ucis), len(completion_tokens),
        )
        return compute_rewards(texts, solution_ucis, completion_tokens)

    return reward_fn


# ---------------------------------------------------------------------------
# Trainer subclass — board tensor injection
# ---------------------------------------------------------------------------


class PuzzleGRPOTrainer(GRPOTrainer):
    """Minimal GRPOTrainer subclass that threads board tensors into forward passes."""

    def _generate_and_score_completions(self, inputs):
        # Stash full inputs so rollout_func can access fen/solution_uci.
        # TRL's RepeatSampler already sends each prompt num_generations times,
        # so inputs has len = per_device_batch * num_generations. rollout_func
        # returns n=1 completion per input → len(completions) == len(inputs). ✓
        self._current_inputs = inputs
        return super()._generate_and_score_completions(inputs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Inject board tensors stashed by rollout_func directly onto the model.
        # We can't inject via _prepare_inputs because it receives the raw dataloader
        # list-of-dicts before GRPOTrainer processes it into a tensor dict.
        board_tensors = getattr(self, "_pending_board_tensors", None)
        if board_tensors is not None:
            device = next(model.parameters()).device
            model._board_tensors_flat = torch.stack(board_tensors).to(device)
            model._move_counts = torch.ones(len(board_tensors), dtype=torch.long, device=device)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def _save(self, output_dir: str | None = None, state_dict=None):
        # Qwen3 ties embed_tokens and lm_head weights. safetensors refuses to save
        # shared tensors. Untie before saving, restore after.
        unwrapped = self.accelerator.unwrap_model(self.model)
        llm = getattr(unwrapped, "llm", unwrapped)
        embed = getattr(getattr(llm, "base_model", llm), "model", llm)
        tied = (
            hasattr(embed, "model")
            and hasattr(embed.model, "embed_tokens")
            and hasattr(llm, "base_model")
            and hasattr(llm.base_model.model, "lm_head")
            and embed.model.embed_tokens.weight.data_ptr()
            == llm.base_model.model.lm_head.weight.data_ptr()
        )
        if tied:
            llm.base_model.model.lm_head.weight = torch.nn.Parameter(
                llm.base_model.model.lm_head.weight.clone()
            )
        super()._save(output_dir, state_dict)
        if tied:
            # Re-tie after save
            llm.base_model.model.lm_head.weight = llm.base_model.model.model.embed_tokens.weight


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    enc_cfg = cfg["encoder"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    global _SGLANG_URL
    _SGLANG_URL = train_cfg.get("sglang_url", _SGLANG_URL)

    # Truncate completions log at startup so stale data from prior runs doesn't accumulate
    open(_COMPLETIONS_LOG, "w").close()

    # Wait for SGLang to be healthy
    log.info("Checking SGLang server at %s...", _SGLANG_URL)
    if not _sglang_health(_SGLANG_URL):
        log.error("SGLang server not reachable. Start with: ./sglang/serve.sh /tmp/chess-merged")
        sys.exit(1)
    log.info("SGLang server ready.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"], trust_remote_code=True)
    tokenizer.padding_side = "left"

    # Base LLM
    from transformers import AutoModelForCausalLM  # noqa: PLC0415
    from safetensors.torch import load_file as safetensors_load  # noqa: PLC0415
    torch_dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    base_llm = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"],
        torch_dtype=torch_dtype,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )

    # Load merged phase1 weights (LoRA already merged into base)
    merged_weights = model_cfg.get("merged_weights")
    if merged_weights:
        log.info("Loading merged phase1 weights from %s", merged_weights)
        state = safetensors_load(merged_weights, device="cpu")
        missing, unexpected = base_llm.load_state_dict(state, strict=False)
        log.info("Merged weights loaded (missing=%d, unexpected=%d)", len(missing), len(unexpected))

    # LoRA
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    peft_llm = get_peft_model(base_llm, peft_config)
    peft_llm.print_trainable_parameters()

    # Wrap with chess CNN encoder
    model = ChessLMWithEncoder(
        llm=peft_llm,
        hidden_size=base_llm.config.hidden_size,
        cnn_in_channels=enc_cfg.get("in_channels", 19),
        cnn_hidden_size=enc_cfg["hidden_size"],
        cnn_num_blocks=enc_cfg["num_blocks"],
        move_token_id=BOARD_TOKEN_ID,
    )
    model.to(torch_dtype).cuda()

    # Load pretrained CNN encoder weights
    encoder_weights = enc_cfg.get("pretrained_weights")
    if encoder_weights:
        state = torch.load(encoder_weights, map_location="cuda", weights_only=True)
        # encoder_weights.pt has "cnn.*" prefix — strip it for model.cnn
        state = {k[len("cnn."):]: v for k, v in state.items() if k.startswith("cnn.")}
        missing, unexpected = model.cnn.load_state_dict(state, strict=True)
        log.info("CNN weights loaded from %s (missing=%d, unexpected=%d)",
                 encoder_weights, len(missing), len(unexpected))
    else:
        log.warning("No pretrained CNN weights specified — encoder starts from random init")

    # Freeze CNN trunk, keep proj + global_proj trainable for GRPO adaptation
    for name, param in model.cnn.named_parameters():
        if "proj" in name or "global_proj" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    proj_params = sum(p.numel() for n, p in model.cnn.named_parameters() if p.requires_grad)
    log.info("CNN: trunk frozen, %d proj params trainable", proj_params)

    # Dataset
    max_steps = args.max_steps or train_cfg.get("max_steps", 5000)
    dataset = PuzzleDataset(
        train_cfg["train_file"],
        tokenizer,
        train_cfg.get("max_prompt_length", 512),
    )

    # GRPO config
    output_dir = cfg.get("output_dir", "checkpoints/qwen3.5-4b-encoder-phase2-puzzle-grpo")
    grpo_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=train_cfg.get("num_generations", 8),
        max_completion_length=train_cfg.get("max_completion_length", 2048),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=train_cfg.get("warmup_steps", 50),
        max_steps=max_steps,
        optim=train_cfg.get("optim", "adamw_torch"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.1),
        beta=train_cfg.get("beta", 0.01),
        temperature=train_cfg.get("temperature", 1.0),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        logging_steps=train_cfg.get("logging_steps", 5),
        eval_strategy=train_cfg.get("eval_strategy", "no"),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 250),
        save_total_limit=train_cfg.get("save_total_limit", 20),
        seed=train_cfg.get("seed", 42),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        remove_unused_columns=False,
        report_to="none",
    )

    sync_every = train_cfg.get("sync_every_n_steps", 10)
    rollout_func = make_rollout_func(
        url=_SGLANG_URL,
        num_generations=grpo_args.num_generations,
        max_new_tokens=grpo_args.max_completion_length,
        temperature=grpo_args.temperature,
        sync_every=sync_every,
        log_sample_every=train_cfg.get("log_sample_every", 50),
    )

    # Reward function — solution_ucis and completion_tokens from rollout stash
    # (TRL passes extra_fields per prompt, not per completion, so we use the stash)
    _trainer_ref: list = []  # filled after trainer is created

    def reward_fn(completions, **kwargs) -> list[float]:
        # TRL passes completions as [[{"role": "assistant", "content": text}], ...] for conversational datasets
        texts = []
        for c in completions:
            if isinstance(c, list):
                texts.append(next((m["content"] for m in c if m.get("role") == "assistant"), ""))
            else:
                texts.append(c)
        trainer_obj = _trainer_ref[0] if _trainer_ref else None
        solution_ucis = getattr(trainer_obj, "_pending_solution_ucis", None) or [""] * len(texts)
        completion_tokens = [len(ids) for ids in getattr(trainer_obj, "_pending_completion_ids", None) or []] or [0] * len(texts)
        fens = getattr(trainer_obj, "_pending_fens", None) or [""] * len(texts)
        log.info("[reward_fn] completions=%d texts=%d solution_ucis=%d completion_tokens=%d",
                 len(completions), len(texts), len(solution_ucis), len(completion_tokens))
        return compute_rewards(texts, solution_ucis, completion_tokens, fens)

    trainer = PuzzleGRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
        rollout_func=rollout_func,
    )
    _trainer_ref.append(trainer)  # let reward_fn access the stash

    log.info("Starting puzzle GRPO training (max_steps=%d, sync_every=%d)", max_steps, sync_every)
    trainer.train(resume_from_checkpoint=args.resume)

    log.info("Training complete. Saving final model to %s", output_dir)
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()
