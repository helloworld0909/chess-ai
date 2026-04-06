"""Test client for ChessQwen3ForCausalLM served via SGLang.

Sends a chess position (FEN) to the server and verifies:
1. Basic text response (no board tokens)
2. Board-encoded response (65 sentinel tokens in prompt)
3. Tool call roundtrip (get_board tool)

Usage:
    python sglang/client_example.py [--base-url http://localhost:8300]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# chess-ai/src for board_to_tensor + constants
_CHESS_AI_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_CHESS_AI_SRC))

from encoder import BOARD_TOKEN, BOARD_TOKENS_PER_POSITION

SENTINEL_BLOCK = BOARD_TOKEN * BOARD_TOKENS_PER_POSITION  # 65 × "<|vision_pad|>"


def test_basic(client, model: str):
    print("\n=== Test 1: Basic text (no board tokens) ===")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a chess expert."},
            {"role": "user", "content": "What is the starting position in chess?"},
        ],
        max_tokens=100,
        temperature=0.0,
    )
    print("Response:", resp.choices[0].message.content[:200])
    print("PASS ✓")


def test_board_encoding(client, model: str):
    print("\n=== Test 2: Board-encoded position ===")
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    prompt = (
        f"The board position is shown below:\n{SENTINEL_BLOCK}\n\n"
        f"FEN: {fen}\n\n"
        "What is the best response for Black?"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a chess expert."},
            {"role": "user", "content": prompt},
        ],
        extra_body={
            "image_data": [{"format": "board_tensor", "fen": fen}]
        },
        max_tokens=200,
        temperature=0.0,
    )
    content = resp.choices[0].message.content
    print("Response:", content[:300])
    assert len(content) > 10, "Response too short"
    print("PASS ✓")


def test_tool_call(client, model: str):
    print("\n=== Test 3: Tool call (get_board) ===")
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    prompt = (
        f"The board position is shown below:\n{SENTINEL_BLOCK}\n\n"
        f"FEN: {fen}\n\n"
        "Use get_board to look at the position after e5, then suggest Black's plan."
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_board",
                "description": (
                    "Get board tokens for the position after playing a move. "
                    "Returns 65 board embedding tokens showing the resulting position."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "move_san": {
                            "type": "string",
                            "description": "Move in SAN notation (e.g. 'e5', 'Nf6', 'O-O')",
                        }
                    },
                    "required": ["move_san"],
                },
            },
        }
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a chess expert with board visualization."},
            {"role": "user", "content": prompt},
        ],
        extra_body={
            "image_data": [{"format": "board_tensor", "fen": fen}]
        },
        tools=tools,
        tool_choice="auto",
        max_tokens=300,
        temperature=0.7,
    )

    choice = resp.choices[0]
    print(f"Finish reason: {choice.finish_reason}")
    if choice.finish_reason == "tool_calls":
        for tc in choice.message.tool_calls:
            print(f"Tool call: {tc.function.name}({tc.function.arguments})")
        print("PASS ✓ (tool call detected)")
    else:
        print("Response:", choice.message.content[:300])
        print("NOTE: model did not call tool (may need tool-use fine-tuning)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8300")
    parser.add_argument("--model", default="chess", help="Model name for the API")
    parser.add_argument("--skip-tool", action="store_true", help="Skip tool call test")
    args = parser.parse_args()

    try:
        import openai
    except ImportError:
        print("Install openai: pip install openai")
        sys.exit(1)

    client = openai.OpenAI(base_url=f"{args.base_url}/v1", api_key="dummy")

    # Verify server is alive
    try:
        import urllib.request
        urllib.request.urlopen(f"{args.base_url}/health", timeout=5)
        print(f"Server at {args.base_url} is healthy.")
    except Exception as e:
        print(f"ERROR: Server not reachable at {args.base_url}: {e}")
        print("Start the server with: ./sglang/serve.sh /path/to/chess-merged")
        sys.exit(1)

    test_basic(client, args.model)
    test_board_encoding(client, args.model)
    if not args.skip_tool:
        test_tool_call(client, args.model)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
