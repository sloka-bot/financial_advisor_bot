/* Avatar speech repeats the advisor reply without a second language model. */
let avatarSession = null;
let avatarStarting = false;
let avatarGeneration = 0;
let avatarMuted = true;
let avatarSpeech = [];
let avatarSpeechId = null;

function avatarControls(connected, message) {
  setText("avatar-status", message);
  for (const name of ["mic", "interrupt", "end"]) {
    const button = document.getElementById(`avatar-${name}`);
    if (button) button.disabled = !connected;
  }
  const start = document.getElementById("avatar-start");
  if (start) start.disabled = connected || avatarStarting;
  setText("avatar-mic", avatarMuted ? "Enable microphone" : "Mute microphone");
}

async function refreshAvatarConfig() {
  try {
    const config = await api.avatarConfig();
    if (!avatarSession && !avatarStarting) {
      avatarControls(
        false,
        config.configured
          ? "Ready. The microphone starts muted. Sessions last up to two minutes."
          : "Video conversation is not connected yet. You can still use text chat.",
      );
      document.getElementById("avatar-start").disabled = !config.configured;
    }
  } catch {
    avatarControls(
      false,
      "Video connection unavailable. Text chat remains available.",
    );
  }
}

async function startAvatarSession() {
  if (avatarStarting || avatarSession) return;
  avatarStarting = true;
  const generation = ++avatarGeneration;
  avatarControls(false, "Connecting video...");
  let session;
  try {
    const result = await api.avatarSession();
    if (generation !== avatarGeneration) return;
    if (result.status !== "ready" || !result.session_token) {
      avatarControls(
        false,
        result.status === "disabled"
          ? "Video conversation is not connected yet."
          : "Video could not connect. Please try again shortly.",
      );
      return;
    }
    const { LiveAvatarSession, SessionEvent, AgentEventsEnum } = LiveAvatarSDK;
    session = new LiveAvatarSession(result.session_token, {
      voiceChat: { defaultMuted: true },
      autoKeepAlive: false,
    });
    avatarSession = session;
    avatarMuted = true;
    const transcripts = new Set();
    session.on(SessionEvent.SESSION_STREAM_READY, () => {
      if (avatarSession !== session) return;
      const video = document.getElementById("avatar-video");
      video.hidden = false;
      document.getElementById("avatar-placeholder").hidden = true;
      session.attach(video);
      video
        .play()
        .catch(() =>
          setText(
            "avatar-status",
            "Press play on the video to hear the advisor.",
          ),
        );
    });
    session.on(AgentEventsEnum.USER_TRANSCRIPTION, (event) => {
      if (avatarSession !== session || avatarMuted || !event.text?.trim())
        return;
      if (event.event_id && transcripts.has(event.event_id)) return;
      if (event.event_id) transcripts.add(event.event_id);
      if (transcripts.size > 200)
        transcripts.delete(transcripts.values().next().value);
      submitChatMessage(event.text);
    });
    session.on(AgentEventsEnum.USER_SPEAK_STARTED, () => {
      if (avatarSession === session && !avatarMuted) interruptAvatar();
    });
    session.on(AgentEventsEnum.AVATAR_SPEAK_ENDED, (event) => {
      if (avatarSession !== session || !avatarSpeechId) return;
      if (event.source_event_id && event.source_event_id !== avatarSpeechId)
        return;
      avatarSpeechId = null;
      nextAvatarSpeech();
    });
    session.on(SessionEvent.SESSION_DISCONNECTED, () => {
      if (avatarSession === session) stopAvatarSession();
    });
    await session.start();
    if (generation !== avatarGeneration || avatarSession !== session) {
      await session.stop();
      return;
    }
    avatarControls(
      true,
      "Connected. Type a question or enable your microphone.",
    );
  } catch {
    if (avatarSession === session) await stopAvatarSession();
    setText(
      "avatar-status",
      "Video could not connect. Text chat remains available.",
    );
  } finally {
    avatarStarting = false;
    const button = document.getElementById("avatar-start");
    if (button) button.disabled = Boolean(avatarSession);
  }
}

async function toggleAvatarMicrophone() {
  const session = avatarSession;
  if (!session) return;
  try {
    if (avatarMuted) {
      if (session.voiceChat.state !== LiveAvatarSDK.VoiceChatState.ACTIVE)
        await session.voiceChat.start({ defaultMuted: true });
      if (session !== avatarSession) return;
      if (session.voiceChat.state !== LiveAvatarSDK.VoiceChatState.ACTIVE)
        throw new Error("Microphone unavailable");
      session.voiceChat.unmute();
    } else session.voiceChat.mute();
    avatarMuted = !avatarMuted;
    avatarControls(
      true,
      avatarMuted
        ? "Microphone muted. You can type a question."
        : "Listening. Speak your question.",
    );
  } catch {
    setText(
      "avatar-status",
      "Microphone access is unavailable. You can type instead.",
    );
  }
}

function interruptAvatar() {
  avatarSpeech = [];
  avatarSpeechId = null;
  try {
    avatarSession?.interrupt();
  } catch {
    /* Disconnected sessions have no speech to stop. */
  }
}

function avatarTextChunks(text) {
  const words = String(text).replace(/\*\*/g, "").trim().split(/\s+/);
  const chunks = [];
  let chunk = "";
  for (const word of words) {
    if (chunk && chunk.length + word.length + 1 > 900) {
      chunks.push(chunk);
      chunk = "";
    }
    chunk += (chunk ? " " : "") + word;
  }
  if (chunk) chunks.push(chunk);
  return chunks;
}

function speakAvatarReply(text) {
  if (!avatarSession) return;
  interruptAvatar();
  avatarSpeech = avatarTextChunks(text);
  nextAvatarSpeech();
}

function nextAvatarSpeech() {
  if (!avatarSession || !avatarSpeech.length) return;
  try {
    avatarSpeechId = avatarSession.repeat(avatarSpeech.shift());
  } catch {
    avatarSpeech = [];
    setText(
      "avatar-status",
      "Audio is unavailable. The complete reply is in chat.",
    );
  }
}

async function stopAvatarSession() {
  ++avatarGeneration;
  interruptAvatar();
  const session = avatarSession;
  avatarSession = null;
  avatarMuted = true;
  const video = document.getElementById("avatar-video");
  if (video) {
    video.srcObject = null;
    video.hidden = true;
  }
  const placeholder = document.getElementById("avatar-placeholder");
  if (placeholder) placeholder.hidden = false;
  avatarControls(false, "Conversation ended. Text chat remains available.");
  try {
    await session?.stop();
  } catch {
    /* The local stream is already detached. */
  }
}

window.addEventListener("pagehide", stopAvatarSession);
