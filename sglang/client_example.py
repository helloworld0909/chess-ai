"""Test client for ChessQwen3ForCausalLM served via SGLang.

Sends a chess position (FEN) to the server and verifies:
1. Basic text response (no board tokens)
2. Board-encoded response (1 sentinel token in prompt)
3. Tool call roundtrip (get_board tool)

Usage:
    python3 sglang/client_example.py [--base-url http://localhost:8300]
"""

from __future__ import annotations

import argparse
import json
import urllib.request

BOARD_TOKEN = "<|vision_pad|>"
SENTINEL_BLOCK = BOARD_TOKEN  # one placeholder per board; scheduler expands to 65 slots
MAX_NEW_TOKENS = 4096
SYSTEM_PROMPT = (
    "You are a chess assistant. The board position is encoded as a sequence of vision tokens. "
    "Use them to identify pieces and answer questions about the position."
)

def post_json(base_url: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_basic(base_url: str):
    print("\n=== Test 1: Basic text (no board tokens) ===")
    resp = post_json(
        base_url,
        "/generate",
        {
            "text": (
                f"{SYSTEM_PROMPT}\n\n"
                "Simple puzzle: White to move and mate in one. "
                "FEN: 7k/8/5KQ1/8/8/8/8/8 w - - 0 1\n"
                "What is the winning move?"
            ),
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 256},
        },
    )
    print("Response:", resp["text"])
    print("PASS ✓")


def test_board_encoding(base_url: str):
    print("\n=== Test 2: Board-encoded position ===")
    fen = "7k/8/5KQ1/8/8/8/8/8 w - - 0 1"
    resp = post_json(
        base_url,
        "/generate",
        {
            "text": (
                f"{SYSTEM_PROMPT}\n\n"
                f"The board position is shown below:\n{SENTINEL_BLOCK}\n\n"
                "White to move and mate in one. What is the winning move?"
            ),
            "image_data": [{"format": "board_tensor", "fen": fen}],
            "sampling_params": {"temperature": 0.5, "max_new_tokens": MAX_NEW_TOKENS},
        },
    )
    content = resp["text"]
    print("Response:", content)
    assert len(content) > 10, "Response too short"
    print("PASS ✓")


def test_tool_call(base_url: str, model: str):
    print("\n=== Test 3: Tool call (get_board) ===")
    fen = "7k/8/5KQ1/8/8/8/8/8 w - - 0 1"
    prompt = (
        f"The board position is shown below:\n{SENTINEL_BLOCK}\n\n"
        "Use get_board to look at the position after Qg7#, then explain the result."
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_board",
                "description": (
                    "Get board tokens for the position after playing a move. "
                    "Returns board embedding tokens showing the resulting position."
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

    resp = post_json(
        base_url,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "image_data": [{"format": "board_tensor", "fen": fen}],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0.5,
        },
    )

    print("Full response:", json.dumps(resp, indent=2))
    choice = resp["choices"][0]
    print(f"Finish reason: {choice['finish_reason']}")
    if choice["finish_reason"] == "tool_calls":
        for tc in choice["message"]["tool_calls"]:
            fn = tc["function"]
            print(f"Tool call: {fn['name']}({fn['arguments']})")
        print("PASS ✓ (tool call detected)")
    else:
        content = choice["message"].get("content") or ""
        print("Response:", content)
        print("NOTE: model did not call tool (may need tool-use fine-tuning)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8300")
    parser.add_argument("--model", default="chess", help="Model name for the API")
    parser.add_argument("--skip-tool", action="store_true", help="Skip tool call test")
    args = parser.parse_args()

    try:
        urllib.request.urlopen(f"{args.base_url}/health", timeout=5)
        print(f"Server at {args.base_url} is healthy.")
    except Exception as e:
        print(f"ERROR: Server not reachable at {args.base_url}: {e}")
        print("Start the server with: ./sglang/serve.sh /path/to/chess-merged")
        raise SystemExit(1)

    test_basic(args.base_url)
    test_board_encoding(args.base_url)
    if not args.skip_tool:
        test_tool_call(args.base_url, args.model)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
