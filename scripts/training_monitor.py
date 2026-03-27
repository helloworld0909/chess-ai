"""Training monitor — serves a live web UI for completions_phase1_grpo.jsonl.

Shows per-step best/worst completions with rendered chess boards.

Usage:
    python scripts/training_monitor.py
    # Open http://localhost:8500
"""

import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

LOG_PATH = Path(__file__).parent.parent / "completions_phase1_grpo.jsonl"

app = FastAPI()

# Lichess piece SVGs (cburnett set, CC BY-SA 3.0)
_LICHESS_PIECE_BASE = "https://lichess1.org/assets/piece/cburnett"
_PIECE_FILE = {
    "K": "wK", "Q": "wQ", "R": "wR", "B": "wB", "N": "wN", "P": "wP",
    "k": "bK", "q": "bQ", "r": "bR", "b": "bB", "n": "bN", "p": "bP",
}


def fen_to_board_html(fen: str) -> str:
    if not fen:
        return "<div class='board-empty'>No FEN</div>"
    try:
        rows = fen.split()[0].split("/")
        if len(rows) != 8:
            return "<div class='board-empty'>Invalid FEN</div>"
        html = "<table class='board'>"
        for rank_idx, row in enumerate(rows):
            html += "<tr>"
            rank_num = 8 - rank_idx
            html += f"<td class='coord'>{rank_num}</td>"
            squares = []
            for ch in row:
                if ch.isdigit():
                    squares.extend(["."] * int(ch))
                else:
                    squares.append(ch)
            for file_idx, piece in enumerate(squares):
                color = "light" if (rank_idx + file_idx) % 2 == 0 else "dark"
                if piece in _PIECE_FILE:
                    piece_html = f"<img src='{_LICHESS_PIECE_BASE}/{_PIECE_FILE[piece]}.svg' width='52' height='52'>"
                else:
                    piece_html = ""
                html += f"<td class='square {color}'>{piece_html}</td>"
            html += "</tr>"
        html += "<tr><td></td>"
        for f in "abcdefgh":
            html += f"<td class='coord'>{f}</td>"
        html += "</tr></table>"
        return html
    except Exception as e:
        return f"<div class='board-empty'>Error: {e}</div>"


def parse_log() -> list[dict]:
    """Parse completions_phase3.jsonl into a list of step dicts (newest first)."""
    if not LOG_PATH.exists():
        return []
    steps = []
    for line in LOG_PATH.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            steps.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(steps))


def _esc(s: str) -> str:
    if s is None:
        return ""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_judge_output(judge_output: dict | None) -> str:
    if not judge_output:
        return "(no judge output)"
    lines = []
    for k, v in judge_output.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def render_judge_input(judge_input: dict | None) -> str:
    if not judge_input:
        return "(no judge input)"
    lines = []
    for k, v in judge_input.items():
        if isinstance(v, dict):
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def render_sample(s: dict | None, open_by_default: bool = False) -> str:
    if s is None:
        return ""
    board_html = fen_to_board_html(s.get("fen", ""))
    label = s.get("label", "")
    label_class = "best" if label == "BEST" else ("worst" if label == "WORST" else "other")
    reward = s.get("reward")
    move = s.get("move", "")
    fen = s.get("fen", "")
    ts = s.get("ts", "")
    trainee_output = s.get("trainee_output", "")
    judge_input_text = render_judge_input(s.get("judge_input"))
    judge_output_text = render_judge_output(s.get("judge_output"))
    open_attr = "open" if open_by_default else ""

    return f"""
    <details {open_attr} class="sample-details {label_class}">
      <summary class="sample-summary">
        <span class="label-badge {label_class}">{_esc(label)}</span>
        <span class="reward">reward = {reward}</span>
        <span class="move">move: <strong>{_esc(move)}</strong></span>
        <span class="ts-small">{_esc(ts)}</span>
        <span class="fen-raw">{_esc(fen)}</span>
      </summary>
      <div class="sample-body">
        <div class="board-col">
          {board_html}
        </div>
        <div class="text-cols">
          <div class="panel">
            <div class="panel-title">Trainee Output</div>
            <pre>{_esc(trainee_output)}</pre>
          </div>
          <div class="panel">
            <div class="panel-title">Judge Input</div>
            <pre>{_esc(judge_input_text)}</pre>
          </div>
          <div class="panel">
            <div class="panel-title">Judge Output</div>
            <pre>{_esc(judge_output_text)}</pre>
          </div>
        </div>
      </div>
    </details>
    """


@app.get("/api/step-count")
def step_count():
    return {"count": len(parse_log())}


@app.get("/", response_class=HTMLResponse)
def index():
    steps = parse_log()

    steps_html = ""
    for step in steps:
        mean = step.get("mean")
        mean_str = f"{mean:+.3f}" if mean is not None else "N/A"
        best = step.get("best") or {}
        worst = step.get("worst") or {}
        best_score = best.get("reward")
        worst_score = worst.get("reward")

        best_worst_html = render_sample(step.get("best"), open_by_default=True) + render_sample(
            step.get("worst"), open_by_default=True
        )

        # All completions collapsed by default
        all_samples = step.get("all") or []
        best_idx = (step.get("best") or {}).get("idx")
        worst_idx = (step.get("worst") or {}).get("idx")
        other_html = "".join(
            render_sample(s, open_by_default=False)
            for s in all_samples
            if s.get("idx") not in (best_idx, worst_idx)
        )
        all_html = (
            f"""
        <details>
          <summary class="all-summary">All {len(all_samples)} completions</summary>
          {best_worst_html}{other_html}
        </details>
        """
            if all_samples
            else best_worst_html
        )

        steps_html += f"""
        <details open>
          <summary class="step-summary">
            Step {step.get("step")} &nbsp;·&nbsp; {step.get("ts")}
            &nbsp;·&nbsp; mean={mean_str}
            &nbsp;·&nbsp; best={best_score}
            &nbsp;·&nbsp; worst={worst_score}
            &nbsp;·&nbsp; n={step.get("n")} ({step.get("n_judged")} judged)
          </summary>
          {all_html}
        </details>
        """

    if not steps_html:
        steps_html = "<p style='color:#888'>No steps logged yet. Waiting for training...</p>"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Training Monitor</title>
<style>
body {{ font-family: monospace; background: #1a1a1a; color: #ddd; margin: 0; padding: 16px; }}
h1 {{ color: #fff; margin-bottom: 4px; }}
.subtitle {{ color: #888; font-size: 12px; margin-bottom: 20px; }}
details {{ margin-bottom: 8px; border: 1px solid #333; border-radius: 6px; overflow: hidden; }}
summary.step-summary {{ background: #2a2a2a; padding: 10px 14px; cursor: pointer; font-size: 14px; color: #ccc; }}
summary.step-summary:hover {{ background: #333; }}
summary.all-summary {{ background: #222; padding: 6px 14px; cursor: pointer; font-size: 12px; color: #888; }}
summary.all-summary:hover {{ color: #aaa; }}
details.sample-details {{ margin: 8px; border-radius: 6px; border: 2px solid #444; overflow: hidden; }}
details.sample-details.best {{ border-color: #2a5; }}
details.sample-details.worst {{ border-color: #a33; }}
details.sample-details.other {{ border-color: #444; }}
summary.sample-summary {{ padding: 8px 12px; background: #252525; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; cursor: pointer; list-style: none; }}
summary.sample-summary:hover {{ background: #2d2d2d; }}
.label-badge {{ font-weight: bold; padding: 2px 8px; border-radius: 4px; font-size: 12px; }}
.label-badge.best {{ background: #1a4a2a; color: #4f4; }}
.label-badge.worst {{ background: #4a1a1a; color: #f44; }}
.label-badge.other {{ background: #2a2a2a; color: #aaa; }}
.reward {{ color: #ff0; font-size: 14px; }}
.move {{ color: #8bf; }}
.ts-small {{ color: #666; font-size: 11px; }}
.fen-raw {{ color: #555; font-size: 11px; }}
.sample-body {{ display: flex; gap: 12px; padding: 12px; background: #1e1e1e; overflow-x: auto; }}
.board-col {{ flex-shrink: 0; }}
.text-cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; flex: 1; min-width: 0; }}
.panel {{ background: #252525; border-radius: 4px; overflow: hidden; }}
.panel-title {{ background: #333; padding: 4px 8px; font-size: 11px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }}
.panel pre {{ margin: 0; padding: 8px; font-size: 11px; white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto; color: #ccc; }}
table.board {{ border-collapse: collapse; }}
table.board td.square {{ width: 56px; height: 56px; text-align: center; vertical-align: middle; padding: 0; }}
table.board td.light {{ background: #f0d9b5; }}
table.board td.dark {{ background: #b58863; }}
table.board td.coord {{ font-size: 11px; color: #888; text-align: center; width: 16px; padding: 0 2px; }}
</style>
</head>
<body>
<h1>Training Monitor</h1>
<div class="subtitle">Auto-refreshes every 30s · {len(steps)} steps logged · {LOG_PATH}</div>
{steps_html}
<script>
(function() {{
  var known = {len(steps)};
  setInterval(function() {{
    fetch('/api/step-count').then(function(r) {{ return r.json(); }}).then(function(d) {{
      if (d.count !== known) {{ location.reload(); }}
    }}).catch(function() {{}});
  }}, 10000);
}})();
</script>
</body>
</html>"""


if __name__ == "__main__":
    port = int(os.environ.get("MONITOR_PORT", 8500))
    print(f"Training monitor: http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
