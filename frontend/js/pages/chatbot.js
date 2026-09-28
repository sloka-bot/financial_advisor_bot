/* Advisor chat with daily history and a beginner/normal mode toggle. */

const QUICK_CHIPS = [
  "What should I buy today?",
  "What are my riskiest holdings?",
  "Explain my portfolio allocation",
  "What is my Sharpe ratio?",
  "How does the model work?",
  "Show me top sentiment stocks",
];

let chatInitialized = false;
let chatHistory = [];
let beginnerMode = false;
let chatQueue = Promise.resolve();
let chatEpoch = 0;

function chatDayKey() {
  return "chat_" + new Date().toISOString().split("T")[0];
}

function readTodayHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(chatDayKey()) || "[]");
    return Array.isArray(saved)
      ? saved.filter(
          (m) =>
            ["user", "assistant"].includes(m.role) &&
            typeof m.content === "string",
        )
      : [];
  } catch {
    return [];
  }
}

function writeTodayHistory(messages) {
  try {
    localStorage.setItem(chatDayKey(), JSON.stringify(messages));
    // Keep only today's chat history.
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k?.startsWith("chat_") && k !== chatDayKey()) {
        localStorage.removeItem(k);
        break;
      }
    }
  } catch {}
}

function renderChatbot() {
  if (chatInitialized) {
    // Refresh Ollama status and keep the conversation on tab switch.
    refreshOllamaStatus();
    return;
  }

  chatInitialized = true;
  chatHistory = readTodayHistory();

  document.getElementById("view-chatbot").innerHTML = `
    <div class="advisor-layout">
      <div class="chat-panel">

        <!-- mode toggle row -->
        <div style="display:flex;align-items:center;gap:10px;padding:10px 16px;border-bottom:1px solid var(--border);background:var(--bg-2)">
          <span style="font-size:12px;color:var(--txt-2);font-weight:500">Mode:</span>
          <label style="display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none">
            <div style="position:relative;width:38px;height:20px">
              <input type="checkbox" id="beginner-toggle" style="opacity:0;width:0;height:0;position:absolute"
                     onchange="toggleBeginnerMode(this.checked)">
              <span id="toggle-track" style="position:absolute;inset:0;background:var(--bg-hover);border-radius:10px;transition:background 0.2s;border:1px solid var(--border)"></span>
              <span id="toggle-thumb" style="position:absolute;top:2px;left:2px;width:14px;height:14px;background:var(--txt-3);border-radius:50%;transition:transform 0.2s,background 0.2s"></span>
            </div>
            <span id="mode-label" style="font-size:12px;color:var(--txt-2)">Normal</span>
          </label>
          <span style="font-size:11px;color:var(--txt-3);margin-left:4px" id="mode-desc">
            Concise, data-driven answers
          </span>
          <button class="btn-secondary" style="margin-left:auto;font-size:12px;padding:4px 12px" onclick="clearChat()">Clear chat</button>
        </div>

        <div class="chat-messages" id="chat-msgs"></div>

        <div class="quick-chips">
          ${QUICK_CHIPS.map((c) => `<button class="quick-chip" onclick="submitChatMessage('${c}')">${c}</button>`).join("")}
        </div>

        <div class="chat-footer">
          <textarea id="chat-input" class="chat-input" rows="1"
            placeholder="Ask about your portfolio, a stock, or investment concepts..."></textarea>
          <button class="chat-send" id="chat-send-btn" aria-label="Send message">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M2 8L14 2L8 14L7 9L2 8Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="currentColor"/>
            </svg>
          </button>
        </div>
      </div>

      <aside class="avatar-panel">
        <div class="section-kicker">Live conversation</div>
        <h2>Talk to your advisor</h2>
        <p>The avatar speaks the replies in this chat. Type a question or enable your microphone. While connected, audio and reply text are sent to LiveAvatar.</p>
        <div id="avatar-placeholder" class="video-placeholder">Your advisor will appear here when a conversation starts.</div>
        <video id="avatar-video" hidden autoplay playsinline controls aria-label="Live advisor avatar" style="width:100%;aspect-ratio:16/9;background:#071426;border-radius:12px"></video>
        <button class="btn-primary" id="avatar-start" onclick="startAvatarSession()">Start conversation</button>
        <button class="btn-secondary" id="avatar-mic" onclick="toggleAvatarMicrophone()" disabled>Enable microphone</button>
        <button class="btn-secondary" id="avatar-interrupt" onclick="interruptAvatar()" disabled>Stop speaking</button>
        <button class="btn-secondary" id="avatar-end" onclick="stopAvatarSession()" disabled>End conversation</button>
        <p id="avatar-status" role="status" class="muted">Checking video connection...</p>
        <div id="ollama-card"><h3>Chat connection</h3><div id="ollama-inner">Checking connection...</div></div>
        <button class="btn-secondary" onclick="clearChat()">Clear today’s chat</button>
      </aside>
    </div>`;

  document.getElementById("chat-send-btn")?.addEventListener("click", () => {
    const v = document.getElementById("chat-input")?.value.trim();
    if (v) submitChatMessage(v);
  });

  document.getElementById("chat-input")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const v = e.target.value.trim();
      if (v) submitChatMessage(v);
    }
  });

  refreshOllamaStatus();

  refreshAvatarConfig();
  // Render saved history.
  if (chatHistory.length) {
    chatHistory.forEach((m) => addChatBubble(m.role, m.content, false));
    document.getElementById("chat-msgs")?.scrollTo(0, 999999);
  } else {
    addChatBubble(
      "assistant",
      "Ask about your saved portfolio, a forecast, or an investment concept. " +
        "Beginner mode explains financial terms in plain language. Model estimates can be wrong.",
    );
  }
}

function toggleBeginnerMode(on) {
  beginnerMode = on;

  const track = document.getElementById("toggle-track");
  const thumb = document.getElementById("toggle-thumb");
  const label = document.getElementById("mode-label");
  const desc = document.getElementById("mode-desc");

  if (track) track.style.background = on ? "var(--green)" : "var(--bg-hover)";
  if (thumb) thumb.style.transform = on ? "translateX(18px)" : "translateX(0)";
  if (thumb) thumb.style.background = on ? "#fff" : "var(--txt-3)";
  if (label) label.textContent = on ? "Beginner" : "Normal";
  if (desc)
    desc.textContent = on
      ? "Simple explanations with analogies - great for learning"
      : "Concise, data-driven answers";
}

function submitChatMessage(text) {
  const epoch = chatEpoch;
  const clean = String(text || "").trim();
  if (!clean) return Promise.resolve();
  chatQueue = chatQueue
    .catch(() => {})
    .then(() => {
      if (epoch === chatEpoch) return sendChatMessage(clean, epoch);
    });
  return chatQueue;
}

async function sendChatMessage(text, epoch) {
  const avatarAtSend = avatarSession;
  interruptAvatar();
  addChatBubble("user", text);
  const input = document.getElementById("chat-input");
  if (input) input.value = "";
  chatHistory.push({ role: "user", content: text });
  writeTodayHistory(chatHistory);

  const typingId = showTypingIndicator();
  const context = chatHistory
    .slice(-10)
    .map((m) => `${m.role === "user" ? "User" : "Advisor"}: ${m.content}`)
    .join("\n");
  const reply = await askAdvisor(text, context);

  clearTypingIndicator(typingId);
  if (epoch !== chatEpoch) return;
  addChatBubble("assistant", reply);
  if (avatarAtSend && avatarAtSend === avatarSession) speakAvatarReply(reply);
  chatHistory.push({ role: "assistant", content: reply });
  writeTodayHistory(chatHistory);
  document.getElementById("chat-msgs")?.scrollTo(0, 999999);
}

async function askAdvisor(text, context) {
  try {
    const data = await api.chat(
      text,
      state.portfolioData || null,
      context,
      beginnerMode ? "beginner" : "normal",
      state.userId || "dev-user",
    );
    return data.reply || "No response received.";
  } catch {
    return offlineReply(text.toLowerCase());
  }
}

function offlineReply(msg) {
  if (!state.portfolioData)
    return "Chat is temporarily unavailable. Your saved holdings are unchanged.";
  const rm = state.portfolioData?.portfolio?.risk_metrics || {};
  if (/sharpe|var|risk/.test(msg))
    return `Sharpe ratio: ${rm.annualized_sharpe?.toFixed(2) ?? "--"} - VaR (95%, 1-day): ${rm.var_95_1day != null ? (rm.var_95_1day * 100).toFixed(2) + "%" : "--"}`;
  return "Chat is temporarily unavailable. Please try again shortly. Your saved holdings are unchanged.";
}

function clearChat() {
  chatEpoch++;
  interruptAvatar();
  chatHistory = [];
  writeTodayHistory(chatHistory);
  const msgs = document.getElementById("chat-msgs");
  if (msgs) {
    msgs.innerHTML = "";
    addChatBubble(
      "assistant",
      "Chat cleared. Ask me anything about your portfolio.",
    );
  }
}

async function refreshOllamaStatus() {
  const el = document.getElementById("ollama-inner");
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
        <div style="font-size:11.5px;color:var(--txt-2)">${models.length ? escapeHtml(models.slice(0, 3).join(", ")) : "No model connected"}</div>
        ${!models.some((m) => m.includes("llama")) ? '<div style="font-size:11px;color:var(--amber);margin-top:4px">Run: ollama pull llama3.2</div>' : ""}
`;
    } else {
      el.innerHTML = `<div style="display:flex;align-items:center;gap:8px">
        <span style="width:8px;height:8px;border-radius:50%;background:var(--txt-3);display:inline-block"></span>
        <span style="font-size:12px;color:var(--txt-2)">Not running - using templates</span>
      </div>`;
    }
  } catch {
    el.innerHTML = `<div style="font-size:12px;color:var(--txt-3)">Connection unavailable</div>`;
  }
}

function addChatBubble(role, text, scroll = true) {
  const msgs = document.getElementById("chat-msgs");
  if (!msgs) return;
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  // Escape message text before applying the supported formatting.
  const html = escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
  const avatar = document.createElement("div");
  avatar.className = "chat-avatar";
  avatar.textContent = role === "user" ? "👤" : "🤖";
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";
  bubble.innerHTML = html; // escaped text with <strong> and <br> only
  div.appendChild(avatar);
  div.appendChild(bubble);
  msgs.appendChild(div);
  if (scroll) msgs.scrollTo(0, 999999);
}

function showTypingIndicator() {
  const msgs = document.getElementById("chat-msgs");
  if (!msgs) return null;
  const id = "typing-" + Date.now();
  const div = document.createElement("div");
  div.id = id;
  div.className = "chat-msg assistant";
  div.innerHTML = `<div class="chat-avatar">🤖</div><div class="chat-bubble"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div></div>`;
  msgs.appendChild(div);
  msgs.scrollTo(0, 999999);
  return id;
}

function clearTypingIndicator(id) {
  if (id) document.getElementById(id)?.remove();
}
