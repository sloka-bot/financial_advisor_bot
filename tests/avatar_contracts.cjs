const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const elements = {};
const sent = [];
let created;
class Session {
  constructor(token, options) {
    assert.equal(token, "temporary");
    assert.equal(options.voiceChat.defaultMuted, true);
    this.events = {};
    this.voiceChat = { state: "active", mute() {}, unmute() {} };
    created = this;
  }
  on(event, cb) {
    this.events[event] = cb;
  }
  async start() {}
  async stop() {
    this.stopped = true;
  }
  repeat(text) {
    sent.push(text);
    return "speech-" + sent.length;
  }
  interrupt() {}
}
const context = {
  window: { addEventListener() {} },
  document: { getElementById: (id) => (elements[id] ??= {}) },
  setText: (id, value) => ((elements[id] ??= {}).textContent = value),
  api: {
    avatarSession: async () => ({
      status: "ready",
      session_token: "temporary",
    }),
    avatarConfig: async () => ({ configured: false }),
  },
  submitChatMessage: (text) => sent.push("input:" + text),
  LiveAvatarSDK: {
    LiveAvatarSession: Session,
    SessionEvent: {
      SESSION_STREAM_READY: "stream",
      SESSION_DISCONNECTED: "disconnected",
    },
    AgentEventsEnum: {
      USER_TRANSCRIPTION: "transcript",
      USER_SPEAK_STARTED: "speaking",
      AVATAR_SPEAK_ENDED: "ended",
    },
    VoiceChatState: { ACTIVE: "active" },
  },
};
vm.createContext(context);
vm.runInContext(
  fs.readFileSync("frontend/js/components/live-avatar.js", "utf8"),
  context,
);
(async () => {
  await vm.runInContext("refreshAvatarConfig()", context);
  assert.equal(elements["avatar-start"].disabled, true);
  await vm.runInContext("startAvatarSession()", context);
  vm.runInContext(
    'speakAvatarReply("Expected return is 0%. It can be wrong.")',
    context,
  );
  assert.equal(sent[0], "Expected return is 0%. It can be wrong.");
  created.events.transcript({ event_id: "one", text: "Question" });
  assert.equal(sent.length, 1, "Muted input must not submit a message");
  await vm.runInContext("toggleAvatarMicrophone()", context);
  created.events.transcript({ event_id: "one", text: "Question" });
  created.events.transcript({ event_id: "one", text: "Question" });
  assert.deepEqual(sent.slice(1), ["input:Question"]);
  const long = "A complete reply with numbers 1 2 3. ".repeat(90).trim();
  const chunks = vm.runInContext(
    `avatarTextChunks(${JSON.stringify(long)})`,
    context,
  );
  assert.equal(Array.from(chunks).join(" "), long);
  assert(chunks.every((chunk) => chunk.length <= 900));
  await vm.runInContext("stopAvatarSession()", context);
  assert(created.stopped);
  created.events.transcript({ event_id: "two", text: "Must not submit" });
  assert.equal(sent.length, 2);
  vm.runInContext('speakAvatarReply("Must not speak")', context);
  assert.equal(sent.length, 2);
  console.log(
    "Avatar contracts passed: exact reply, complete chunks, muted microphone, deduplicated speech, session cleanup.",
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
