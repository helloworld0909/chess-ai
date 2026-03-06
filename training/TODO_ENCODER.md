# CNN Board Encoder — Integration Plan

## Motivation

Chess is fundamentally a pattern recognition game. A CNN operating over the raw 8×8
board tensor can capture spatial relationships that text representations (FEN, ASCII)
can only approximate:
- Attack/defense maps (who controls which squares)
- Piece interaction patterns (pins, forks, discovered attacks)
- Pawn structure topology (passed, isolated, doubled, chains)
- King safety geometry

AlphaZero and Leela Chess Zero validate this approach: their ResNet trunks learn rich
positional embeddings purely from the 8×8 board representation.

The integration goal is to prepend a **soft board token** to the Qwen LLM input — the
same architecture LLaVA uses for vision — so the model has direct access to positional
features beyond what tokenised FEN provides.

---

## Board Tensor Format

```
Input shape: (C, 8, 8)

Channels (19 total, AlphaZero-style):
  0–5:   White pieces  — P N B R Q K  (1 where piece exists)
  6–11:  Black pieces  — P N B R Q K
  12:    Side to move  — all-1 if White to move, all-0 if Black
  13:    White kingside castling right
  14:    White queenside castling right
  15:    Black kingside castling right
  16:    Black queenside castling right
  17:    En passant file  (1 in the en-passant file column)
  18:    Move number normalised to [0,1]  (optional, can drop)
```

Builder function: `board_to_tensor(board: chess.Board) -> torch.Tensor` (19, 8, 8).

---

## CNN Architecture (Leela-style ResNet trunk)

```
Input: (19, 8, 8)
→ Conv2d(19, 256, kernel=3, padding=1) + BatchNorm2d + ReLU
→ ResBlock × 6  [each: Conv(256,256,3,pad=1) → BN → ReLU → Conv → BN + skip → ReLU]
→ GlobalAvgPool  →  (256,)
→ Linear(256, hidden_dim)  →  (hidden_dim,)   # projection into LLM space
```

`hidden_dim` = LLM embedding dim (e.g. 2560 for Qwen3.5-4B).

The output is a single vector that becomes one soft prefix token prepended to the
token sequence.

---

## Leela Chess Zero Weights

Leela's network weights are open source (Apache 2.0).

**Download**: https://lczero.org/play/networks/bestnets/
- Best network: `~T80-2024` series — ~256 filters, 6–10 blocks
- Format: `.pb.gz` (protobuf) or `.onnx`

**Extraction options**:

### Option A — lczero Python bindings (cleanest)
```bash
pip install lczero  # or build from source
```
Load weights, extract trunk conv layers, map to PyTorch.

### Option B — ONNX export then load in PyTorch (recommended)
1. Download `.onnx` from Leela's network page
2. Use `onnx2torch` or manually map layer names to a PyTorch ResNet
3. Load state dict

### Option C — Train from scratch with policy/value head
- Supervised on ~1M positions from Lichess (FEN → Stockfish best move + eval)
- Requires a few hours on 2× RTX 5090
- Gives a chess-specific encoder without Leela format issues
- Reasonable fallback if Leela weight extraction proves messy

---

## LLM Integration (LLaVA-style)

The board token is inserted as the **first token** of the user message (before FEN/text):

```
[board_token] [text tokens: "## Position\n\nBoard before..."]
```

### Model wrapper

```python
class ChessLMWithEncoder(nn.Module):
    def __init__(self, llm, cnn_encoder, projection):
        self.llm = llm            # Qwen3.5-4B (frozen or LoRA)
        self.cnn = cnn_encoder    # Leela trunk (frozen or lightly tuned)
        self.proj = projection    # Linear(256, hidden_dim)

    def forward(self, board_tensor, input_ids, attention_mask, labels=None):
        board_emb = self.proj(self.cnn(board_tensor))          # (B, hidden_dim)
        board_token = board_emb.unsqueeze(1)                   # (B, 1, hidden_dim)
        text_embs = self.llm.get_input_embeddings()(input_ids) # (B, T, hidden_dim)
        inputs_embeds = torch.cat([board_token, text_embs], dim=1)
        # Prepend 1 to attention_mask for the board token
        board_mask = torch.ones(B, 1, device=attention_mask.device)
        attention_mask = torch.cat([board_mask, attention_mask], dim=1)
        # Shift labels if provided (mask board token position with -100)
        ...
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
```

---

## Training Strategy

### Phase 1 — Freeze CNN, train projection + LoRA on LLM
- CNN weights frozen (Leela pretrained = already chess-aware)
- `projection` Linear layer trained from scratch
- LLM fine-tuned with QLoRA (same config as current `qwen3_30b.yaml`)
- Data: existing `train.jsonl` coaching data (add `board_tensor` field to each example)

### Phase 2 — Unfreeze CNN last N blocks (optional)
- Only if Phase 1 results suggest encoder is under-fitting
- Fine-tune CNN blocks 4–6 + projection + LoRA jointly
- Lower LR for CNN (1e-5) vs projection/LoRA (3e-4)

---

## Data Pipeline Changes

Each training example needs a `board_tensor` field (or the FEN to derive it at load time).

Simplest approach: **derive tensor from FEN at training time** — no data regeneration needed.

```python
# In training DataLoader / collate_fn:
import chess
board = chess.Board(example["fen"])
tensor = board_to_tensor(board)  # (19, 8, 8) float32
```

This avoids storing large tensors in JSONL and keeps data pipeline unchanged.

---

## Files to Create / Modify

| File | Action | Description |
|------|--------|-------------|
| `src/encoder/board_tensor.py` | Create | `board_to_tensor(board)` utility |
| `src/encoder/cnn.py` | Create | `LeelaTrunk` ResNet, `ChessEncoder` wrapper |
| `src/encoder/load_leela.py` | Create | Weight loading from ONNX / .pb.gz |
| `src/encoder/__init__.py` | Create | Package init |
| `training/train_encoder.py` | Create | Modified train.py with `ChessLMWithEncoder` |
| `training/configs/qwen3_4b_encoder.yaml` | Create | Config for encoder experiment |
| `tests/test_encoder.py` | Create | Unit tests for tensor builder + CNN forward pass |

Existing files unchanged in Phase 1 — encoder is additive.

---

## Open Questions

1. **Leela weight format**: ONNX export of T80 networks maps cleanly to PyTorch?
   Need to verify layer naming. Fallback: train from scratch.

2. **Single board token vs multiple**: LLaVA uses one token per image patch. Could
   project the (256, 8, 8) feature map (before global pool) to 64 tokens (one per square).
   More expressive but 64× more tokens. Start with single token, evaluate.

3. **Qwen3.5-4B vs 30B**: 4B is the target (faster iteration). Encoder adds ~15M params
   (CNN trunk) + 256×2560 projection — negligible overhead.

4. **Does it actually help?** Ablation needed: train identical model with and without
   board encoder on held-out coaching quality eval. Only add complexity if measurable gain.

---

## Implementation Order

- [ ] `src/encoder/board_tensor.py` — FEN → (19,8,8) tensor, unit tested
- [ ] `src/encoder/cnn.py` — LeelaTrunk ResNet architecture in PyTorch
- [ ] `src/encoder/load_leela.py` — weight loading (ONNX or scratch)
- [ ] `tests/test_encoder.py` — forward pass shape tests, FEN round-trip
- [ ] `training/train_encoder.py` — modified SFT with board token injection
- [ ] Ablation: encoder vs no-encoder on 200-example coaching eval set
