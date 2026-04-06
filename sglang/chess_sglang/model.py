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
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
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

        # ---- LLM ----
        self.model = Qwen3_5ForCausalLM(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        # ---- CNN encoder ----
        # Qwen3.5 nests hidden_size inside text_config at the top level
        _hidden_size = getattr(config, "hidden_size", None)
        if _hidden_size is None:
            _tc = getattr(config, "text_config", {})
            _hidden_size = (_tc.get("hidden_size") if isinstance(_tc, dict) else getattr(_tc, "hidden_size", None)) or 2560

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
    ) -> torch.Tensor:
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.model,
            data_embedding_funcs={Modality.IMAGE: self._cnn_embed},
            placeholder_tokens={Modality.IMAGE: [BOARD_TOKEN_ID]},
            positions=positions,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load LLM weights via Qwen3_5ForCausalLM, then load CNN from
        encoder_weights.pt in the same model directory."""

        # Separate CNN keys from LLM keys
        llm_weights = []
        model_path_hint: Optional[str] = None

        for name, param in weights:
            if name.startswith("cnn."):
                # CNN weights in safetensors — skip, loaded from .pt below
                continue
            llm_weights.append((name, param))

        # Load LLM weights
        loaded = self.model.load_weights(iter(llm_weights))

        # Load CNN weights from encoder_weights.pt
        self._load_cnn_weights()

        return loaded

    def _load_cnn_weights(self):
        """Load CNN weights from encoder_weights.pt next to model.safetensors."""
        from sglang.srt.utils import get_model_path_from_args

        # Try to find encoder_weights.pt via server_args model path
        # SGLang sets the model path in the environment during loading
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
