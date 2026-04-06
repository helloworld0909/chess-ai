"""ChessQwen3ForCausalLM — SGLang external model plugin.

Wraps Qwen3.5-4B with the chess CNN board encoder. Board tensors arrive as
IMAGE-modality features from ChessBoardProcessor; this model runs the CNN
inside its data_embedding_func and splices the (65, 2560) embeddings at the
<|vision_pad|> sentinel positions via general_mm_embed_routine.

Architecture:
    ChessBoardProcessor  (tokenizer worker, CPU)
        FEN → board_to_tensor → (19, 8, 8) raw tensor
        Stored as MultimodalDataItem.feature

    ChessQwen3ForCausalLM.forward  (model worker, GPU)
        data_embedding_func:
            (19, 8, 8) feature → ChessEncoder → (65, 2560) embeddings
        general_mm_embed_routine splices embeddings at sentinel positions
        → Qwen3.5 forward

Registration:
    EntryClass = ChessQwen3ForCausalLM
    Loaded via SGLANG_EXTERNAL_MODEL_PACKAGE=chess_sglang
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Resolve chess-ai/src path
# ---------------------------------------------------------------------------
_CHESS_AI_SRC = Path(__file__).parent.parent.parent / "src"
if str(_CHESS_AI_SRC) not in sys.path:
    sys.path.insert(0, str(_CHESS_AI_SRC))

from encoder.cnn import ChessEncoder  # noqa: E402

# ---------------------------------------------------------------------------
# SGLang imports (resolved at runtime inside sglang-serve venv)
# ---------------------------------------------------------------------------
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
# Qwen3_5ForCausalLM is the backbone (transformer layers only, returns hidden states).
# We add our own lm_head + LogitsProcessor, following Qwen3VLForConditionalGeneration.
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.utils import add_prefix

BOARD_TOKEN_ID = 248055  # <|vision_pad|>, never appears in chess text


class ChessQwen3ForCausalLM(nn.Module):
    """Qwen3.5-4B + frozen chess CNN board encoder for SGLang serving.

    The CNN trunk + projectors are loaded from encoder_weights.pt (saved by
    merge_adapter.py) and kept frozen. They run on the same GPU as the LLM
    inside the model worker — not in the processor subprocess.

    Weight loading:
      - LLM weights: delegated to Qwen3_5ForCausalLM.load_weights()
        (handles qkv/gate_up stacking and rotary skips automatically)
      - CNN weights: loaded from encoder_weights.pt in the same directory
        as model.safetensors (set by load_weights via model_path)
    """

    def __init__(
        self,
        config,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # ---- LLM backbone ----
        # Qwen3_5ForCausalLM: transformer layers only, returns hidden states.
        # Qwen3_5Config wraps text_config — extract it (Qwen3_5ForCausalLM expects
        # Qwen3_5TextConfig which has layers_block_type property).
        llm_config = getattr(config, "text_config", config)
        self.model = Qwen3_5ForCausalLM(
            config=llm_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        # ---- lm_head + LogitsProcessor ----
        # Qwen3.5-4B has tie_word_embeddings=True: lm_head = embed_tokens (same object).
        # This is set after model init — embed_tokens is populated during load_weights.
        # For now, create a placeholder; we set it properly in load_weights.
        self.lm_head = None  # set in load_weights after backbone is loaded
        self.logits_processor = LogitsProcessor(llm_config)

        # ---- CNN encoder ----
        # hidden_size lives in text_config for Qwen3.5
        _hidden_size = getattr(llm_config, "hidden_size", None) or 2560

        self.cnn = ChessEncoder(
            in_channels=getattr(config, "encoder_in_channels", 19),
            hidden_size=getattr(config, "encoder_hidden_size", 256),
            num_blocks=getattr(config, "encoder_num_blocks", 10),
            out_dim=_hidden_size,  # must match LLM hidden dim (2560 for Qwen3.5-4B)
        )
        # Frozen: no gradients during serving
        for p in self.cnn.parameters():
            p.requires_grad_(False)
        self.cnn.eval()

        self._model_path: Optional[str] = None  # set during load_weights

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    # ------------------------------------------------------------------
    # CNN embedding function — called by general_mm_embed_routine
    # ------------------------------------------------------------------

    def _cnn_embed(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Run CNN on raw board tensors from the processor.

        Args:
            items: List of MultimodalDataItem with .feature = (19, 8, 8) tensor
                   (one item per board position in the batch).

        Returns:
            (N_boards * 65, hidden_size) embedding tensor on the model's device.
        """
        tensors = []
        for item in items:
            feat = item.feature
            if isinstance(feat, torch.Tensor):
                tensors.append(feat)
            else:
                tensors.append(torch.tensor(feat))

        # Stack: (N_boards, 19, 8, 8)
        board_batch = torch.stack(tensors, dim=0)
        device = next(self.cnn.parameters()).device
        dtype = next(self.cnn.parameters()).dtype
        board_batch = board_batch.to(device=device, dtype=dtype)

        with torch.no_grad():
            # (N_boards, 65, hidden_size)
            cnn_out = self.cnn(board_batch)

        # Flatten to (N_boards * 65, hidden_size) for splice routine
        N = cnn_out.shape[0]
        return cnn_out.reshape(N * 65, -1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # general_mm_embed_routine splices CNN embeddings at sentinel positions,
        # then calls self.model(input_ids=None, input_embeds=...) → hidden states.
        # Qwen3_5ForCausalLM.forward accepts input_embeds (no 's') — compatible.
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            data_embedding_funcs={Modality.IMAGE: self._cnn_embed},
            placeholder_tokens={Modality.IMAGE: [BOARD_TOKEN_ID]},
            positions=positions,
            **kwargs,
        )
        # Apply lm_head + logits processor (following Qwen3VLForConditionalGeneration)
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load LLM weights via Qwen3_5ForCausalLM, then load CNN from
        encoder_weights.pt in the same model directory."""

        # Separate CNN keys from LLM keys.
        # Our merged safetensors have keys like model.layers.N.* and model.embed_tokens.*
        # Qwen3_5ForCausalLM.load_weights expects keys WITHOUT the leading "model." prefix
        # (its params_dict has embed_tokens.*, layers.*, norm.*).
        # SGLang's own Qwen3VLForConditionalGeneration strips model.language_model. → model.
        # We strip model. entirely since our backbone is registered at prefix "model".
        llm_weights = []

        for name, param in weights:
            if name.startswith("cnn."):
                continue
            # Strip leading "model." so backbone load_weights finds keys correctly
            if name.startswith("model."):
                name = name[len("model."):]
            llm_weights.append((name, param))

        # Load backbone weights (embed_tokens, transformer layers, norm)
        loaded = self.model.load_weights(iter(llm_weights))

        # Qwen3.5-4B: tie_word_embeddings=True → lm_head IS embed_tokens (same object)
        self.lm_head = self.model.embed_tokens

        # Load CNN weights from encoder_weights.pt
        self._load_cnn_weights()

        return loaded

    def _load_cnn_weights(self):
        """Load CNN weights from encoder_weights.pt next to model.safetensors."""
        # Try to find encoder_weights.pt via SGLANG_MODEL_PATH env var set in serve.sh
        model_dir = os.environ.get("SGLANG_MODEL_PATH", "")
        if not model_dir:
            # Fallback: try to infer from module location (won't work in all cases)
            import importlib.util
            spec = importlib.util.find_spec("chess_sglang")
            if spec:
                model_dir = str(Path(spec.origin).parent.parent)

        encoder_path = Path(model_dir) / "encoder_weights.pt"
        if not encoder_path.exists():
            # Try current working directory
            encoder_path = Path("encoder_weights.pt")

        if encoder_path.exists():
            state = torch.load(str(encoder_path), map_location="cpu", weights_only=True)
            # Keys are cnn.* — strip prefix for self.cnn
            cnn_state = {
                k[len("cnn."):]: v
                for k, v in state.items()
                if k.startswith("cnn.")
            }
            missing, unexpected = self.cnn.load_state_dict(cnn_state, strict=True)
            if missing:
                print(f"[ChessQwen3] CNN missing keys: {missing[:5]}")
            if unexpected:
                print(f"[ChessQwen3] CNN unexpected keys: {unexpected[:5]}")
            print(f"[ChessQwen3] Loaded CNN weights from {encoder_path}")
        else:
            print(f"[ChessQwen3] WARNING: encoder_weights.pt not found at {encoder_path}. "
                  f"Set SGLANG_MODEL_PATH or place encoder_weights.pt in the model directory.")


# SGLang discovers the model class via this attribute
EntryClass = ChessQwen3ForCausalLM
