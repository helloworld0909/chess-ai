/* Chess Game Review – frontend logic
 *
 * Uses chessground (Lichess board library) loaded as an ES module.
 * No build step, no jQuery, no external image files.
 */

import { Chessground } from 'https://cdn.jsdelivr.net/npm/chessground@9.1.1/+esm';

// ── State ────────────────────────────────────────────────────────────────────

let cg           = null;   // Chessground instance
let currentGame  = null;   // full game object from /api/game/{id}
let currentIndex = -1;     // -1 = start, 0..N-1 = after move[index]
let flipped      = false;
let _username    = '';     // logged-in player username
let encoderCache = {};     // "fen|uci" → encoder result
let baseCache    = {};     // "fen|uci" → base LLM result
let _activeStream = null;  // active AbortController

// Model name for base LLM (must match vLLM --served-model-name or model id)
const MODEL_BASE = 'Qwen/Qwen3-4B-Thinking-2507';

// ── Helpers ───────────────────────────────────────────────────────────────────

function fenPieces(fullFen) { return fullFen.split(' ')[0]; }
function uciToLastMove(uci) {
  if (!uci || uci.length < 4) return undefined;
  return [uci.slice(0, 2), uci.slice(2, 4)];
}
function fenTurn(fullFen) { return fullFen.split(' ')[1] === 'b' ? 'black' : 'white'; }
function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function isFineTuned() {
  return document.getElementById('lora-toggle').checked;
}

// ── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  cg = Chessground(document.getElementById('board'), {
    fen: 'start',
    orientation: 'white',
    movable:   { color: 'none', free: false },
    draggable: { enabled: false },
    selectable: { enabled: false },
    animation: { enabled: true, duration: 200 },
    highlight: { lastMove: true, check: true },
  });

  bindControls();
  bindModelControls();
  updateToggleLabels();
  await loadGameList();
});

// ── Game list ─────────────────────────────────────────────────────────────────

async function loadGameList() {
  const [gamesRes, userRes] = await Promise.all([
    fetch('/api/games'),
    fetch('/api/username'),
  ]);
  const games = await gamesRes.json();
  const { username } = await userRes.json();

  _username = username ?? '';
  document.getElementById('username-label').textContent = username ? `@${username}` : '';

  const sel = document.getElementById('game-select');
  sel.innerHTML = '';

  if (!games.length) {
    sel.innerHTML = '<option value="">No games found</option>';
    return;
  }

  games.forEach((g, i) => {
    const opt = document.createElement('option');
    opt.value = g.id;
    opt.textContent = `${i + 1}. ${g.title}`;
    sel.appendChild(opt);
  });

  sel.addEventListener('change', () => { if (sel.value) loadGame(sel.value); });
  await loadGame(games[0].id);
}

// ── Load game ─────────────────────────────────────────────────────────────────

async function loadGame(gameId) {
  abortStream();
  const res = await fetch(`/api/game/${gameId}`);
  currentGame   = await res.json();
  currentIndex  = -1;
  encoderCache  = {};
  baseCache     = {};
  // Auto-flip: show player's pieces at the bottom
  if (_username) {
    flipped = currentGame.black.toLowerCase() === _username.toLowerCase();
  }
  renderMoveList();
  updateGameInfo();
  goTo(-1);
}

// ── Board update ──────────────────────────────────────────────────────────────

function setBoard(fen, lastMoveUci) {
  cg.set({
    fen:         fenPieces(fen),
    turnColor:   fenTurn(fen),
    lastMove:    uciToLastMove(lastMoveUci),
    orientation: flipped ? 'black' : 'white',
  });
}

// ── Navigation ────────────────────────────────────────────────────────────────

function goTo(index) {
  if (!currentGame) return;
  const max = currentGame.moves.length - 1;
  index = Math.max(-1, Math.min(index, max));
  currentIndex = index;

  if (index === -1) {
    const startFen = currentGame.moves[0]?.fen_before
      ?? 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
    setBoard(startFen, null);
  } else {
    const move = currentGame.moves[index];
    const fen  = index < max
      ? currentGame.moves[index + 1].fen_before
      : currentGame.final_fen;
    setBoard(fen, move.uci);
  }

  highlightActive();
  updatePositionInfo();
  updateButtons();

  if (index >= 0) {
    const m = currentGame.moves[index];
    fetchAnalysis(m.fen_before, m.uci);
  } else {
    clearAnalysis();
  }
}

function bindControls() {
  document.getElementById('btn-start').addEventListener('click', () => goTo(-1));
  document.getElementById('btn-prev') .addEventListener('click', () => goTo(currentIndex - 1));
  document.getElementById('btn-next') .addEventListener('click', () => goTo(currentIndex + 1));
  document.getElementById('btn-end')  .addEventListener('click', () => currentGame && goTo(currentGame.moves.length - 1));

  document.getElementById('btn-flip').addEventListener('click', () => {
    flipped = !flipped;
    goTo(currentIndex);
  });

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowLeft')  { e.preventDefault(); goTo(currentIndex - 1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); goTo(currentIndex + 1); }
    if (e.key === 'ArrowUp')    { e.preventDefault(); goTo(-1); }
    if (e.key === 'ArrowDown')  { e.preventDefault(); currentGame && goTo(currentGame.moves.length - 1); }
  });
}

// ── Model toggle ──────────────────────────────────────────────────────────────

function bindModelControls() {
  document.getElementById('lora-toggle').addEventListener('change', () => {
    updateToggleLabels();
    if (currentIndex >= 0 && currentGame) {
      const m = currentGame.moves[currentIndex];
      fetchAnalysis(m.fen_before, m.uci);
    }
  });
}

function updateToggleLabels() {
  const checked = document.getElementById('lora-toggle').checked;
  document.getElementById('label-base').style.cssText  = checked ? '' : 'color:var(--text);font-weight:600';
  document.getElementById('label-lora').style.cssText  = checked ? 'color:#64b5f6;font-weight:600' : '';
}

// ── Move list ─────────────────────────────────────────────────────────────────

function renderMoveList() {
  const el = document.getElementById('move-list');
  el.innerHTML = '';
  if (!currentGame?.moves.length) return;

  currentGame.moves.forEach((m, i) => {
    if (m.color === 'white') {
      const num = document.createElement('span');
      num.className = 'move-num';
      num.textContent = m.move_number + '.';
      el.appendChild(num);
    }
    const cell = document.createElement('span');
    cell.className = 'move-cell';
    cell.dataset.index = i;
    cell.textContent = m.san;
    cell.addEventListener('click', () => goTo(i));
    el.appendChild(cell);
    if (m.color === 'white' && i === currentGame.moves.length - 1) {
      el.appendChild(document.createElement('span'));
    }
  });
}

function highlightActive() {
  document.querySelectorAll('.move-cell').forEach(el => {
    el.classList.toggle('active', parseInt(el.dataset.index) === currentIndex);
  });
  document.querySelector('.move-cell.active')
    ?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

// ── Info bar ──────────────────────────────────────────────────────────────────

function updatePositionInfo() {
  const el = document.getElementById('move-counter');
  if (!currentGame) { el.textContent = '—'; return; }
  const total = currentGame.moves.length;
  el.textContent = currentIndex === -1
    ? 'Start position'
    : `Move ${currentIndex + 1} / ${total} — ${currentGame.moves[currentIndex].san}`;
}

function updateButtons() {
  const total = currentGame ? currentGame.moves.length : 0;
  document.getElementById('btn-start').disabled = currentIndex <= -1;
  document.getElementById('btn-prev') .disabled = currentIndex <= -1;
  document.getElementById('btn-next') .disabled = currentIndex >= total - 1;
  document.getElementById('btn-end')  .disabled = currentIndex >= total - 1;
}

function updateGameInfo() {
  const el = document.getElementById('game-info');
  if (!currentGame) { el.textContent = ''; return; }
  const resultText = {
    '1-0': 'White wins', '0-1': 'Black wins', '1/2-1/2': 'Draw',
  }[currentGame.result] || currentGame.result;
  el.innerHTML =
    `<strong>${currentGame.white}</strong> vs <strong>${currentGame.black}</strong> &nbsp;·&nbsp; ` +
    `${resultText} by ${currentGame.result_detail} &nbsp;·&nbsp; ` +
    `${currentGame.date} &nbsp;·&nbsp; TC: ${currentGame.time_control}s`;
  document.getElementById('result-badge').textContent = currentGame.result;
}

// ── Analysis dispatch ─────────────────────────────────────────────────────────

function abortStream() {
  if (_activeStream) { try { _activeStream.abort(); } catch (_) {} _activeStream = null; }
}

function clearAnalysis() {
  abortStream();
  document.getElementById('encoder-content').innerHTML =
    '<p class="placeholder">Select a move to see analysis.</p>';
}

function fetchAnalysis(fen, moveUci) {
  if (isFineTuned()) {
    fetchEncoderAnalysis(fen, moveUci);
  } else {
    fetchBaseAnalysis(fen, moveUci);
  }
}

// ── Encoder (fine-tuned) analysis ─────────────────────────────────────────────

async function fetchEncoderAnalysis(fen, moveUci) {
  const key = `${fen}|${moveUci}`;
  const el = document.getElementById('encoder-content');

  if (encoderCache[key]) {
    renderEncoderAnalysis(encoderCache[key], el);
    return;
  }

  abortStream();
  el.innerHTML = '<span class="analysis-loading">Analyzing…</span>';
  const ctrl = new AbortController();
  _activeStream = ctrl;

  try {
    const res = await fetch('/api/analyze_encoder', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen, move_uci: moveUci }),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let sseBuffer = '', metaData = null, tokenText = '', thinkText = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      sseBuffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = sseBuffer.indexOf('\n\n')) !== -1) {
        const message = sseBuffer.slice(0, boundary);
        sseBuffer = sseBuffer.slice(boundary + 2);
        let eventType = 'message', dataStr = '';
        for (const line of message.split('\n')) {
          if (line.startsWith('event: ')) eventType = line.slice(7).trim();
          else if (line.startsWith('data: ')) dataStr = line.slice(6);
        }
        if (!dataStr) continue;

        if (eventType === 'meta') {
          metaData = JSON.parse(dataStr);
          renderEncoderMeta(metaData, el);
        } else if (eventType === 'token') {
          tokenText += JSON.parse(dataStr);
          updateEncoderToken(tokenText, el);
        } else if (eventType === 'think') {
          thinkText += JSON.parse(dataStr);
          updateThinking(thinkText, el);
        } else if (eventType === 'done') {
          const result = { ...metaData, completion: tokenText, thinking: thinkText };
          encoderCache[key] = result;
          const cur = currentIndex >= 0 ? currentGame?.moves[currentIndex] : null;
          if (cur && `${cur.fen_before}|${cur.uci}` === `${fen}|${moveUci}`) {
            renderEncoderAnalysis(result, el);
          }
          return;
        } else if (eventType === 'error') {
          const e = JSON.parse(dataStr);
          el.innerHTML = `<span class="placeholder">Error: ${escapeHtml(e.error ?? String(e))}</span>`;
          return;
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      el.innerHTML = `<span class="placeholder">Tutor unavailable: ${escapeHtml(err.message)}</span>`;
    }
  } finally {
    if (_activeStream === ctrl) _activeStream = null;
  }
}

// ── Base LLM analysis ─────────────────────────────────────────────────────────

async function fetchBaseAnalysis(fen, moveUci) {
  const key = `${fen}|${moveUci}`;
  const el = document.getElementById('encoder-content');

  if (baseCache[key]) {
    renderBaseAnalysis(baseCache[key], el);
    return;
  }

  abortStream();
  el.innerHTML = '<span class="analysis-loading">Analyzing…</span>';
  const ctrl = new AbortController();
  _activeStream = ctrl;

  let metaData = null, commentText = '', thinkText = '';

  try {
    const res = await fetch('/api/analyze/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen, move_uci: moveUci, model: MODEL_BASE }),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let sseBuffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      sseBuffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = sseBuffer.indexOf('\n\n')) !== -1) {
        const message = sseBuffer.slice(0, boundary);
        sseBuffer = sseBuffer.slice(boundary + 2);
        let eventType = 'message', dataStr = '';
        for (const line of message.split('\n')) {
          if (line.startsWith('event: ')) eventType = line.slice(7).trim();
          else if (line.startsWith('data: ')) dataStr = line.slice(6);
        }
        if (!dataStr) continue;

        if (eventType === 'meta') {
          metaData = JSON.parse(dataStr);
          renderBaseStreamHeader(metaData, el);
        } else if (eventType === 'token') {
          commentText += JSON.parse(dataStr);
          updateBaseComment(commentText, el);
        } else if (eventType === 'think') {
          thinkText += JSON.parse(dataStr);
          updateThinking(thinkText, el);
        } else if (eventType === 'done') {
          const result = { ...metaData, comment: commentText, thinking: thinkText };
          baseCache[key] = result;
          const cur = currentIndex >= 0 ? currentGame?.moves[currentIndex] : null;
          if (cur && `${cur.fen_before}|${cur.uci}` === `${fen}|${moveUci}`) {
            renderBaseAnalysis(result, el);
          }
          return;
        } else if (eventType === 'error') {
          const e = JSON.parse(dataStr);
          el.innerHTML = `<span class="placeholder">Error: ${escapeHtml(e.error ?? String(e))}</span>`;
          return;
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      el.innerHTML = `<span class="placeholder">Base model unavailable: ${escapeHtml(err.message)}</span>`;
    }
  } finally {
    if (_activeStream === ctrl) _activeStream = null;
  }
}

// ── Base LLM render helpers ───────────────────────────────────────────────────

function renderBaseStreamHeader(meta, el) {
  const cls = meta.classification;
  const activeCell = document.querySelector('.move-cell.active');
  if (activeCell) activeCell.dataset.class = cls;
  el.innerHTML = `
    <div class="analysis-row">
      <span class="source-badge source-llm">Base</span>
      <span class="class-pill class-${cls}">${cls}</span>
      <span class="eval-pill">${meta.eval_before}</span>
    </div>
    ${!meta.is_best
      ? `<div class="best-move-row">Best: <strong>${meta.best_move}</strong>${meta.cp_loss > 0 ? ` (−${meta.cp_loss} cp)` : ''}</div>`
      : ''}
    <p class="comment-text streaming"></p>
    <details class="thinking-block" hidden>
      <summary>Thinking</summary>
      <pre class="thinking-text"></pre>
    </details>
  `;
}

function updateBaseComment(text, el) {
  const p = el.querySelector('.comment-text.streaming');
  if (p) p.textContent = text;
}

function renderBaseAnalysis(data, el) {
  const cls = data.classification;
  const activeCell = document.querySelector('.move-cell.active');
  if (activeCell) activeCell.dataset.class = cls;

  const thinkingBlock = data.thinking
    ? `<details class="thinking-block"><summary>Thinking</summary><pre class="thinking-text">${escapeHtml(data.thinking)}</pre></details>`
    : '';

  el.innerHTML = `
    <div class="analysis-row">
      <span class="source-badge source-llm">Base</span>
      <span class="class-pill class-${cls}">${cls}</span>
      <span class="eval-pill">${data.eval_before}</span>
    </div>
    ${!data.is_best
      ? `<div class="best-move-row">Best: <strong>${data.best_move}</strong>${data.cp_loss > 0 ? ` (−${data.cp_loss} cp)` : ''}</div>`
      : ''}
    <p class="comment-text">${escapeHtml(data.comment)}</p>
    ${thinkingBlock}
  `;
}

// ── Encoder render helpers ────────────────────────────────────────────────────

function _encoderHeader(data) {
  const cls = data.classification || '';
  const activeCell = document.querySelector('.move-cell.active');
  if (activeCell && cls) activeCell.dataset.class = cls;
  const clsPill = cls ? `<span class="class-pill class-${cls}">${cls}</span>` : '';
  const bestRow = (!data.is_best && data.best_move)
    ? `<div class="best-move-row">Best: <strong>${escapeHtml(data.best_move)}</strong>${data.cp_loss > 0 ? ` (−${data.cp_loss} cp)` : ''}</div>`
    : '';
  return `
    <div class="analysis-row">
      <span class="source-badge source-encoder">Fine-tuned</span>
      ${clsPill}
      <span class="eval-pill">${escapeHtml(data.eval_label || '')}</span>
    </div>
    ${bestRow}`;
}

function renderEncoderMeta(meta, el) {
  const linesHtml = (meta.key_lines || []).map((line, i) =>
    `<div class="encoder-line"><span class="line-num">Line ${i + 1}</span> <span class="line-sans">${escapeHtml(line)}</span></div>`
  ).join('');

  el.innerHTML = `
    ${_encoderHeader(meta)}
    <div class="encoder-lines-section">
      <div class="encoder-lines-title">Engine Key Lines</div>
      ${linesHtml}
    </div>
    <details class="thinking-block" hidden>
      <summary>Reasoning</summary>
      <pre class="thinking-text"></pre>
    </details>
    <div class="encoder-completion streaming"></div>
  `;
}

function updateEncoderToken(text, el) {
  const div = el.querySelector('.encoder-completion.streaming');
  if (!div) return;
  div.innerHTML = formatEncoderCompletion(text);
}

function updateThinking(text, el) {
  const details = el.querySelector('.thinking-block');
  const pre = el.querySelector('.thinking-text');
  if (!details || !pre) return;
  details.hidden = false;
  // Keep collapsed by default — user can expand; just update content
  pre.textContent = text;
}

function renderEncoderAnalysis(data, el) {
  const linesHtml = (data.key_lines || []).map((line, i) =>
    `<div class="encoder-line"><span class="line-num">Line ${i + 1}</span> <span class="line-sans">${escapeHtml(line)}</span></div>`
  ).join('');

  const thinkingBlock = data.thinking
    ? `<details class="thinking-block"><summary>Reasoning</summary><pre class="thinking-text">${escapeHtml(data.thinking)}</pre></details>`
    : '';

  el.innerHTML = `
    ${_encoderHeader(data)}
    <div class="encoder-lines-section">
      <div class="encoder-lines-title">Engine Key Lines</div>
      ${linesHtml}
    </div>
    ${thinkingBlock}
    <div class="encoder-completion">${formatEncoderCompletion(data.completion || '')}</div>
  `;
}

/**
 * Format encoder completion text:
 * - <line>LINE N: ...</line> blocks → styled annotation rows
 * - remaining text → coaching comment paragraph
 */
function formatEncoderCompletion(text) {
  const lineRe = /<line>(.*?)<\/line>/gs;
  const lines = [];
  let lastIndex = 0;
  let match;

  while ((match = lineRe.exec(text)) !== null) {
    lines.push(match[1].trim());
    lastIndex = lineRe.lastIndex;
  }

  const comment = text.slice(lastIndex).trim();

  const linesHtml = lines.map(l => {
    const evalMatch = l.match(/^(.*?)\|\s*eval:\s*(.+)$/s);
    if (evalMatch) {
      const content = evalMatch[1].trim();
      const evalLabel = evalMatch[2].trim();
      const evalClass = evalLabel.includes('white') ? 'eval-white'
                      : evalLabel.includes('black') ? 'eval-black' : 'eval-equal';
      return `<div class="annotated-line">
        <span class="annotated-moves">${escapeHtml(content)}</span>
        <span class="eval-label ${evalClass}">${escapeHtml(evalLabel)}</span>
      </div>`;
    }
    return `<div class="annotated-line"><span class="annotated-moves">${escapeHtml(l)}</span></div>`;
  }).join('');

  const commentHtml = comment
    ? `<p class="encoder-comment">${escapeHtml(comment)}</p>`
    : '';

  return linesHtml + commentHtml;
}
