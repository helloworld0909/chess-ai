# Chess SGLang external model package.
# Registered via: SGLANG_EXTERNAL_MODEL_PACKAGE=chess_sglang

from chess_sglang.model import ChessQwen3ForCausalLM
from chess_sglang.processor import ChessBoardProcessor

# Wire processor to model class so SGLang's registry finds it
ChessBoardProcessor.models = [ChessQwen3ForCausalLM]
