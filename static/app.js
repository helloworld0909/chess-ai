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
let analysisCache = {};    // "fen|uci|model" → analysis response
let compareMode  = false;  // true = show fine-tuned + base side by side

/** Active AbortControllers for in-flight SSE streams; cancelled on navigation. */
let _activeStreams = [];

// Model names (must match vLLM --lora-modules)
const MODEL_LORA = 'chess-tutor';
const MODEL_BASE = 'Qwen/Qwen3-4B-Thinking-2507';

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Extract piece-placement portion of FEN (chessground only needs this part). */
function fenPieces(fullFen) {
  return fullFen.split(' ')[0];
}

/** Convert UCI move string ("e2e4") to chessground lastMove array (["e2","e4"]). */
function uciToLastMove(uci) {
  if (!uci || uci.length < 4) return undefined;
  return [uci.slice(0, 2), uci.slice(2, 4)];
}

/** Return whose turn it is from a full FEN string. */
function fenTurn(fullFen) {
  return fullFen.split(' ')[1] === 'b' ? 'black' : 'white';
}

/** Escape HTML special characters for safe injection into innerHTML. */
function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  cg = Chessground(document.getElementById('board'), {
    fen: 'start',
    orientation: 'white',
    movable:   { color: 'none', free: false },   // read-only
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
  abortActiveStreams();
  const res = await fetch(`/api/game/${gameId}`);
  currentGame   = await res.json();
  currentIndex  = -1;
  analysisCache = {};
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
    if (_activeTab === 'encoder') fetchEncoderAnalysis(m.fen_before, m.uci);
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

    // Fill empty column when white's move ends the game
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

// ── Model toggle & compare ────────────────────────────────────────────────────

function bindModelControls() {
  document.getElementById('lora-toggle').addEventListener('change', () => {
    updateToggleLabels();
    if (compareMode) return; // toggle disabled in compare mode
    if (currentIndex >= 0 && currentGame) {
      const m = currentGame.moves[currentIndex];
      fetchAnalysis(m.fen_before, m.uci);
    }
  });

  document.getElementById('btn-compare').addEventListener('click', () => {
    compareMode = !compareMode;
    const btn = document.getElementById('btn-compare');
    const label = document.getElementById('compare-mode-label');
    const basePane = document.getElementById('analysis-content-base');
    const toggle = document.getElementById('lora-toggle');
    const wrapper = document.getElementById('analysis-wrapper');

    if (compareMode) {
      btn.classList.add('active');
      label.hidden = false;
      basePane.hidden = false;
      wrapper.classList.add('compare-active');
      toggle.disabled = true;
    } else {
      btn.classList.remove('active');
      label.hidden = true;
      basePane.hidden = true;
      wrapper.classList.remove('compare-active');
      toggle.disabled = false;
    }

    // Re-fetch for the current move
    if (currentIndex >= 0 && currentGame) {
      const m = currentGame.moves[currentIndex];
      fetchAnalysis(m.fen_before, m.uci);
    }
  });
}

/** Return the active model name based on toggle state (single-pane mode). */
function activeModel() {
  return document.getElementById('lora-toggle').checked ? MODEL_LORA : MODEL_BASE;
}

function updateToggleLabels() {
  const checked = document.getElementById('lora-toggle').checked;
  document.getElementById('label-base').style.cssText  = checked ? '' : 'color:var(--text);font-weight:600';
  document.getElementById('label-lora').style.cssText  = checked ? 'color:#64b5f6;font-weight:600' : '';
}

// ── Analysis ──────────────────────────────────────────────────────────────────

/** Cancel any in-flight SSE streams (called when the user navigates to a new move). */
function abortActiveStreams() {
  _activeStreams.forEach(ctrl => { try { ctrl.abort(); } catch (_) {} });
  _activeStreams = [];
}

async function fetchAnalysis(fen, moveUci) {
  abortActiveStreams();

  if (compareMode) {
    fetchCompare(fen, moveUci);
    return;
  }

  const model = activeModel();
  const key = `${fen}|${moveUci}|${model}`;
  const el = document.getElementById('analysis-content');

  // Cache hit — render immediately
  if (analysisCache[key]) {
    renderAnalysis(analysisCache[key], el);
    return;
  }

  showAnalysisLoading(el);
  streamAnalysis(fen, moveUci, model, key, el, null);
}

async function fetchCompare(fen, moveUci) {
  const elLora = document.getElementById('analysis-content');
  const elBase = document.getElementById('analysis-content-base');

  const keyLora = `${fen}|${moveUci}|${MODEL_LORA}`;
  const keyBase = `${fen}|${moveUci}|${MODEL_BASE}`;

  if (analysisCache[keyLora]) {
    renderAnalysis(analysisCache[keyLora], elLora, 'Fine-tuned');
  } else {
    showAnalysisLoading(elLora, 'Fine-tuned');
    streamAnalysis(fen, moveUci, MODEL_LORA, keyLora, elLora, 'Fine-tuned');
  }

  if (analysisCache[keyBase]) {
    renderAnalysis(analysisCache[keyBase], elBase, 'Base model');
  } else {
    showAnalysisLoading(elBase, 'Base model');
    streamAnalysis(fen, moveUci, MODEL_BASE, keyBase, elBase, 'Base model');
  }
}

/**
 * Open an SSE stream to /api/analyze/stream and progressively render tokens.
 * Populates analysisCache[cacheKey] on completion and does a final clean render.
 */
async function streamAnalysis(fen, moveUci, model, cacheKey, el, modelLabel) {
  const ctrl = new AbortController();
  _activeStreams.push(ctrl);

  // Accumulated buffers
  let metaData = null;
  let commentText = '';
  let thinkText = '';

  try {
    const res = await fetch('/api/analyze/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fen, move_uci: moveUci, model }),
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

      // Parse complete SSE messages (terminated by \n\n)
      let boundary;
      while ((boundary = sseBuffer.indexOf('\n\n')) !== -1) {
        const message = sseBuffer.slice(0, boundary);
        sseBuffer = sseBuffer.slice(boundary + 2);

        let eventType = 'message';
        let dataStr = '';
        for (const line of message.split('\n')) {
          if (line.startsWith('event: ')) eventType = line.slice(7).trim();
          else if (line.startsWith('data: ')) dataStr = line.slice(6);
        }
        if (!dataStr) continue;

        if (eventType === 'meta') {
          metaData = JSON.parse(dataStr);
          renderStreamHeader(metaData, el, modelLabel);

        } else if (eventType === 'token') {
          commentText += JSON.parse(dataStr);
          updateStreamComment(commentText, el);

        } else if (eventType === 'think') {
          thinkText += JSON.parse(dataStr);
          updateStreamThinking(thinkText, el);

        } else if (eventType === 'done') {
          // Populate cache and do a clean final render
          if (metaData) {
            const cached = {
              ...metaData,
              comment: commentText,
              comment_source: 'llm',
              thinking: thinkText,
            };
            analysisCache[cacheKey] = cached;
            // Only do final render if this move is still active
            const cur = currentIndex >= 0 ? currentGame?.moves[currentIndex] : null;
            if (cur && `${cur.fen_before}|${cur.uci}` === `${fen}|${moveUci}`) {
              renderAnalysis(cached, el, modelLabel);
            }
          }
          return;

        } else if (eventType === 'error') {
          const errData = JSON.parse(dataStr);
          showAnalysisError(errData.error ?? 'stream error', el);
          return;
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      showAnalysisError(err.message, el);
    }
  } finally {
    _activeStreams = _activeStreams.filter(c => c !== ctrl);
  }
}

/** Render the classification/eval header row immediately when meta event arrives. */
function renderStreamHeader(meta, el, modelLabel) {
  const cls = meta.classification;
  if (!modelLabel || modelLabel === 'Fine-tuned') {
    const activeCell = document.querySelector('.move-cell.active');
    if (activeCell) activeCell.dataset.class = cls;
  }
  const modelBadge = modelLabel ? `<span class="model-badge">${modelLabel}</span>` : '';
  el.innerHTML = `
    <div class="analysis-row">
      ${modelBadge}
      <span class="class-pill class-${cls}">${cls}</span>
      <span class="eval-pill">${meta.eval_before}</span>
      <span class="source-badge source-llm">AI</span>
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

/** Append streamed comment text into the comment paragraph. */
function updateStreamComment(text, el) {
  const p = el.querySelector('.comment-text.streaming');
  if (p) p.textContent = text;
}

/** Append streamed thinking text into the details block. */
function updateStreamThinking(text, el) {
  const details = el.querySelector('.thinking-block');
  const pre = el.querySelector('.thinking-text');
  if (!details || !pre) return;
  details.hidden = false;
  pre.textContent = text;
}

function renderAnalysis(data, el, modelLabel) {
  const cls = data.classification;

  // Tag move cell for CSS annotation badge (★ ?? ? !?) — only from primary pane
  if (!modelLabel || modelLabel === 'Fine-tuned') {
    const activeCell = document.querySelector('.move-cell.active');
    if (activeCell) activeCell.dataset.class = cls;
  }

  const sourceBadge = data.comment_source === 'llm'
    ? '<span class="source-badge source-llm">AI</span>'
    : '<span class="source-badge source-template">engine</span>';

  const modelBadge = modelLabel
    ? `<span class="model-badge">${modelLabel}</span>`
    : '';

  const thinkingBlock = data.thinking
    ? `<details class="thinking-block">
        <summary>Thinking</summary>
        <pre class="thinking-text">${escapeHtml(data.thinking)}</pre>
       </details>`
    : '';

  el.innerHTML = `
    <div class="analysis-row">
      ${modelBadge}
      <span class="class-pill class-${cls}">${cls}</span>
      <span class="eval-pill">${data.eval_before}</span>
      ${sourceBadge}
    </div>
    ${!data.is_best
      ? `<div class="best-move-row">Best: <strong>${data.best_move}</strong>${data.cp_loss > 0 ? ` (−${data.cp_loss} cp)` : ''}</div>`
      : ''}
    <p class="comment-text">${data.comment}</p>
    ${thinkingBlock}
  `;
}

function showAnalysisLoading(el, label) {
  const prefix = label ? `<span class="model-badge">${label}</span> ` : '';
  el.innerHTML = `${prefix}<span class="analysis-loading">Analyzing...</span>`;
}

function showAnalysisError(msg, el) {
  el.innerHTML = `<span class="placeholder">Analysis unavailable: ${msg}</span>`;
}

function clearAnalysis() {
  document.getElementById('analysis-content').innerHTML =
    '<p class="placeholder">Select a move to see analysis.</p>';
  document.getElementById('analysis-content-base').innerHTML =
    '<p class="placeholder">Base model analysis will appear here.</p>';
  document.getElementById('encoder-content').innerHTML =
    '<p class="placeholder">Select a move to see encoder analysis.</p>';
}

// ── Analysis tabs (Tutor / Encoder) ───────────────────────────────────────────

let _activeTab = 'llm';

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      switchTab(tab);
      // Trigger encoder fetch if switching to encoder and a move is selected
      if (tab === 'encoder' && currentIndex >= 0 && currentGame) {
        const m = currentGame.moves[currentIndex];
        fetchEncoderAnalysis(m.fen_before, m.uci);
      }
    });
  });
});

function switchTab(tab) {
  _activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.getElementById('tab-content-llm').hidden = tab !== 'llm';
  document.getElementById('tab-content-encoder').hidden = tab !== 'encoder';
}

// ── Encoder analysis ──────────────────────────────────────────────────────────

let encoderCache = {};  // "fen|uci" → cached encoder result
let _encoderStream = null;

async function fetchEncoderAnalysis(fen, moveUci) {
  const key = `${fen}|${moveUci}`;
  const el = document.getElementById('encoder-content');

  if (encoderCache[key]) {
    renderEncoderAnalysis(encoderCache[key], el);
    return;
  }

  el.innerHTML = '<span class="analysis-loading">Running encoder model…</span>';

  if (_encoderStream) { try { _encoderStream.abort(); } catch (_) {} }
  const ctrl = new AbortController();
  _encoderStream = ctrl;

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
    let sseBuffer = '';
    let metaData = null;
    let tokenText = '';
    let thinkText = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      sseBuffer += decoder.decode(value, { stream: true });

      let boundary;
      while ((boundary = sseBuffer.indexOf('\n\n')) !== -1) {
        const message = sseBuffer.slice(0, boundary);
        sseBuffer = sseBuffer.slice(boundary + 2);

        let eventType = 'message';
        let dataStr = '';
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
          updateStreamThinking(thinkText, el);
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
          el.innerHTML = `<span class="placeholder">Encoder error: ${escapeHtml(e.error ?? String(e))}</span>`;
          return;
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      el.innerHTML = `<span class="placeholder">Encoder unavailable: ${escapeHtml(err.message)}</span>`;
    }
  } finally {
    if (_encoderStream === ctrl) _encoderStream = null;
  }
}

/** Render key lines + placeholder for streaming completion. */
function renderEncoderMeta(meta, el) {
  const linesHtml = (meta.key_lines || []).map((line, i) =>
    `<div class="encoder-line"><span class="line-num">Line ${i + 1}</span> <span class="line-sans">${escapeHtml(line)}</span></div>`
  ).join('');

  el.innerHTML = `
    <div class="analysis-row">
      <span class="source-badge source-encoder">Encoder ✦</span>
      <span class="eval-pill">${escapeHtml(meta.eval_label || '')}</span>
    </div>
    <div class="encoder-lines-section">
      <div class="encoder-lines-title">Engine Key Lines</div>
      ${linesHtml}
    </div>
    <div class="encoder-completion streaming"></div>
    <details class="thinking-block" hidden>
      <summary>Thinking</summary>
      <pre class="thinking-text"></pre>
    </details>
  `;
}

/** Stream completion text into the encoder output div. */
function updateEncoderToken(text, el) {
  const div = el.querySelector('.encoder-completion.streaming');
  if (!div) return;
  // Render <line>...</line> blocks with formatting
  div.innerHTML = formatEncoderCompletion(text);
}

/** Final render of full encoder result. */
function renderEncoderAnalysis(data, el) {
  const linesHtml = (data.key_lines || []).map((line, i) =>
    `<div class="encoder-line"><span class="line-num">Line ${i + 1}</span> <span class="line-sans">${escapeHtml(line)}</span></div>`
  ).join('');

  const thinkingBlock = data.thinking
    ? `<details class="thinking-block"><summary>Thinking</summary><pre class="thinking-text">${escapeHtml(data.thinking)}</pre></details>`
    : '';

  el.innerHTML = `
    <div class="analysis-row">
      <span class="source-badge source-encoder">Encoder ✦</span>
      <span class="eval-pill">${escapeHtml(data.eval_label || '')}</span>
    </div>
    <div class="encoder-lines-section">
      <div class="encoder-lines-title">Engine Key Lines</div>
      ${linesHtml}
    </div>
    <div class="encoder-completion">${formatEncoderCompletion(data.completion || '')}</div>
    ${thinkingBlock}
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
    // Split "LINE N: content | eval: label" for styling
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

