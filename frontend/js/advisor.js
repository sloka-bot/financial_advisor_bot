/*
 * Advisor chat with daily localStorage memory.
 * Only renders once — switching tabs and back preserves the full conversation.
 */

const QUICK_CHIPS = [
  'What should I buy today?',
  'What are my riskiest holdings?',
  'Explain my portfolio allocation',
  'What is my Sharpe ratio?',
  'How does the model work?',
  'Show me top sentiment stocks',
];

let _advisorReady = false;   /* prevent re-render on tab switch */
let _history      = [];

function todayKey() {
  return 'chat_' + new Date().toISOString().split('T')[0];
}
function loadHistory() {
  try { return JSON.parse(localStorage.getItem(todayKey()) || '[]'); } catch { return []; }
}
function saveHistory(msgs) {
  try {
    localStorage.setItem(todayKey(), JSON.stringify(msgs));
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k?.startsWith('chat_') && k !== todayKey()) { localStorage.removeItem(k); break; }
    }
  } catch {}
}

function renderAdvisor() {
  if (_advisorReady) {
    /* tab switch — just refresh status without rebuilding chat */
    checkOllamaStatus();
    return;
  }

  _advisorReady = true;
  _history      = loadHistory();

  document.getElementById('view-advisor').innerHTML = `
    <div class="advisor-layout">
      <div class="chat-panel">
        <div class="chat-messages" id="chat-msgs"></div>
        <div class="quick-chips">
          ${QUICK_CHIPS.map(c => `<button class="quick-chip" onclick="sendMessage('${c}')">${c}</button>`).join('')}
        </div>
        <div class="chat-footer">
          <textarea id="chat-input" class="chat-input" rows="1"
            placeholder="Ask about your portfolio, a stock, or investment concepts…"></textarea>
          <button class="chat-send" id="chat-send-btn">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M2 8L14 2L8 14L7 9L2 8Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="currentColor"/>
            </svg>
          </button>
        </div>
      </div>

      <div class="avatar-panel future-panel">
        <div class="avatar-placeholder">
          <div class="avatar-icon">👤</div>
          <div style="font-weight:600;font-size:14px">AI Avatar</div>
          <div style="font-size:12px;color:var(--txt-3);text-align:center;max-width:200px;line-height:1.6">
            HeyGen streaming avatar
          </div>
          <div id="ollama-card" class="card" style="width:100%;margin-top:20px;padding:14px 16px">
            <div class="chart-title" style="margin-bottom:10px">Local LLM</div>
            <div id="ollama-inner"><div style="font-size:12px;color:var(--txt-2)">Checking…</div></div>
            <div style="font-family:var(--mono);font-size:11px;color:var(--txt-3);margin-top:10px;line-height:1.8">
              brew install ollama<br>ollama pull llama3.2<br>ollama serve
            </div>
          </div>
          <div style="font-size:11px;color:var(--txt-3);margin-top:10px;text-align:center">
            Chat history saves for today · clears at midnight
          </div>
          <button class="btn-secondary" style="margin-top:8px;font-size:11px" onclick="clearChat()">
            Clear today's chat
          </button>
        </div>
      </div>
    </div>`;

  document.getElementById('chat-send-btn')?.addEventListener('click', () => {
    const v = document.getElementById('chat-input')?.value.trim();
    if (v) sendMessage(v);
  });
  document.getElementById('chat-input')?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); const v = e.target.value.trim(); if (v) sendMessage(v); }
  });

  checkOllamaStatus();

  /* paint saved history */
  if (_history.length) {
    _history.forEach(m => appendBubble(m.role, m.content, false));
    document.getElementById('chat-msgs')?.scrollTo(0, 999999);
  } else {
    appendBubble('assistant', 'Hi — I\'m your AI financial advisor. I use FinBERT, XGBoost, and LSTM model outputs to answer questions about your portfolio. Run an analysis first, then ask me anything.');
  }
}

async function sendMessage(text) {
  appendBubble('user', text);
  const input = document.getElementById('chat-input');
  if (input) input.value = '';
  _history.push({ role: 'user', content: text });
  saveHistory(_history);

  const tid     = appendTyping();
  const context = _history.slice(-10).map(m => `${m.role === 'user' ? 'User' : 'Advisor'}: ${m.content}`).join('\n');
  const reply   = await fetchReply(text, context);

  removeTyping(tid);
  appendBubble('assistant', reply);
  _history.push({ role: 'assistant', content: reply });
  saveHistory(_history);
  document.getElementById('chat-msgs')?.scrollTo(0, 999999);
}

async function fetchReply(text, context) {
  try {
    const data = await api.chat(text, state.portfolioData || null, context);
    return data.reply || 'No response received.';
  } catch { return clientFallback(text.toLowerCase()); }
}

function clientFallback(msg) {
  if (!state.portfolioData) return 'Run an analysis from the sidebar first.';
  const rm = state.portfolioData?.portfolio?.risk_metrics || {};
  if (/sharpe|var|risk/.test(msg))
    return `Sharpe: ${rm.annualized_sharpe?.toFixed(2)??'--'} · VaR: ${rm.var_95_1day!=null?(rm.var_95_1day*100).toFixed(2)+'%':'--'}`;
  return 'Server appears offline. Start with: uvicorn backend.main:app --reload --port 8000';
}

function clearChat() {
  _history = [];
  saveHistory(_history);
  const msgs = document.getElementById('chat-msgs');
  if (msgs) {
    msgs.innerHTML = '';
    appendBubble('assistant', 'Chat cleared. Ask me anything about your portfolio.');
  }
}

async function checkOllamaStatus() {
  const el = document.getElementById('ollama-inner');
  if (!el) return;
  try {
    const data = await api.ollamaStatus();
    if (data.ollama_available) {
      const models = data.loaded_models || [];
      el.innerHTML = `
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
          <span style="width:8px;height:8px;border-radius:50%;background:var(--green);display:inline-block"></span>
          <span style="font-size:12px;font-weight:600;color:var(--green)">Running</span>
        </div>
        <div style="font-size:11.5px;color:var(--txt-2)">${models.length ? models.slice(0,3).join(', ') : 'No models loaded'}</div>
        ${!models.some(m => m.includes('llama')) ? '<div style="font-size:11px;color:var(--amber);margin-top:4px">Run: ollama pull llama3.2</div>' : ''}`;
    } else {
      el.innerHTML = `<div style="display:flex;align-items:center;gap:8px"><span style="width:8px;height:8px;border-radius:50%;background:var(--txt-3);display:inline-block"></span><span style="font-size:12px;color:var(--txt-2)">Not running — using templates</span></div>`;
    }
  } catch { el.innerHTML = `<div style="font-size:12px;color:var(--txt-3)">Start the server to check</div>`; }
}

function appendBubble(role, text, scroll = true) {
  const msgs = document.getElementById('chat-msgs');
  if (!msgs) return;
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  const html = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
  div.innerHTML = `<div class="chat-avatar">${role==='user'?'👤':'🤖'}</div><div class="chat-bubble">${html}</div>`;
  msgs.appendChild(div);
  if (scroll) msgs.scrollTo(0, 999999);
}

function appendTyping() {
  const msgs = document.getElementById('chat-msgs');
  if (!msgs) return null;
  const id = 'typing-' + Date.now();
  const div = document.createElement('div');
  div.id = id; div.className = 'chat-msg assistant';
  div.innerHTML = `<div class="chat-avatar">🤖</div><div class="chat-bubble"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div></div>`;
  msgs.appendChild(div);
  msgs.scrollTo(0, 999999);
  return id;
}

function removeTyping(id) { if (id) document.getElementById(id)?.remove(); }
