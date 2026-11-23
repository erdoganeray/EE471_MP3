// Dream Canvas front end: paint button -> /api/image, mic -> /api/transcribe, send -> /api/chat
const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"), imageBox: $("imageBox"), image: $("image"), empty: $("emptyState"),
  loading: $("imageLoading"), paintBtn: $("paintBtn"), prompt: $("prompt"),
  history: $("history"), micBtn: $("micBtn"), message: $("message"), sendBtn: $("sendBtn"),
};
const chat = [];  // [{role, content}], sent to the server every turn (the server keeps no state)
const SUBTITLE = els.status.textContent;

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// ---------- 1. image generation ----------
async function paint() {
  const prompt = els.prompt.value.trim();
  if (!prompt) { els.prompt.focus(); return; }
  els.paintBtn.disabled = true;
  els.loading.hidden = false;
  try {
    const { image } = await api("/api/image", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt }),
    });
    els.image.src = image;
    els.image.alt = prompt;
    els.image.hidden = false;
    els.empty.hidden = true;
    els.imageBox.classList.add("has-image");
  } catch (e) {
    addBubble("assistant", `Could not paint the image: ${e.message}`, { error: true });
  } finally {
    els.loading.hidden = true;
    els.paintBtn.disabled = false;
  }
}
els.paintBtn.addEventListener("click", paint);
els.prompt.addEventListener("keydown", (e) => { if (e.key === "Enter") paint(); });

// ---------- 2. chat ----------
function addBubble(role, text, { error = false, typing = false } = {}) {
  const row = document.createElement("div");
  row.className = `row ${role}`;
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}${error ? " error" : ""}`;
  const sender = document.createElement("span");
  sender.className = "sender";
  sender.textContent = role === "user" ? "You" : "Assistant";
  bubble.append(sender);
  if (typing) {
    const dots = document.createElement("span");
    dots.className = "typing";
    dots.innerHTML = "<i></i><i></i><i></i>";
    bubble.append(dots);
  } else {
    const p = document.createElement("p");
    p.textContent = text;  // textContent: model output is never parsed as HTML
    bubble.append(p);
    if (role === "assistant" && !error) {
      // shortcut for the example workflow: copy the chatbot's prompt into the prompt box
      const use = document.createElement("button");
      use.className = "use-prompt";
      use.textContent = "Use as prompt";
      use.addEventListener("click", () => { els.prompt.value = text; els.prompt.focus(); });
      bubble.append(use);
    }
  }
  row.append(bubble);
  els.history.append(row);
  els.history.scrollTop = els.history.scrollHeight;
  return row;
}

async function send() {
  const text = els.message.value.trim();
  if (!text || els.sendBtn.disabled) return;
  els.message.value = "";
  addBubble("user", text);
  chat.push({ role: "user", content: text });
  els.sendBtn.disabled = true;
  const typing = addBubble("assistant", "", { typing: true });
  try {
    const { reply } = await api("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages: chat }),
    });
    typing.remove();
    addBubble("assistant", reply);
    chat.push({ role: "assistant", content: reply });
  } catch (e) {
    typing.remove();
    chat.pop();  // drop the failed user turn so the history stays valid
    addBubble("assistant", `Something went wrong: ${e.message}`, { error: true });
  } finally {
    els.sendBtn.disabled = false;
    els.message.focus();
  }
}
els.sendBtn.addEventListener("click", send);
els.message.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });

// ---------- 3. speech to text (mic button toggles recording) ----------
let recorder = null;

async function startRecording() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    addBubble("assistant", `Microphone not available: ${e.message}`, { error: true });
    return;
  }
  const chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    els.micBtn.classList.remove("recording");
    els.micBtn.disabled = true;
    els.message.value = "";
    els.message.placeholder = "Transcribing…";
    try {
      const form = new FormData();
      form.append("audio", new Blob(chunks, { type: recorder.mimeType }), "speech.webm");
      const { text } = await api("/api/transcribe", { method: "POST", body: form });
      els.message.value = text;  // the user can check/edit it, then press send
      els.message.focus();
    } catch (e) {
      addBubble("assistant", `Could not transcribe: ${e.message}`, { error: true });
    } finally {
      els.message.placeholder = "Type or press the mic…";
      els.micBtn.disabled = false;
      recorder = null;
    }
  };
  recorder.start();
  els.micBtn.classList.add("recording");
  els.micBtn.title = "Stop recording";
}

els.micBtn.addEventListener("click", () => {
  if (recorder && recorder.state === "recording") {
    recorder.stop();
    els.micBtn.title = "Speak";
  } else {
    startRecording();
  }
});

// ---------- model status in the subtitle while models load ----------
async function pollStatus() {
  try {
    const s = await api("/api/status");
    const waiting = Object.values(s).filter((m) => !m.ready).length;
    els.status.textContent = waiting ? `Loading AI models… (${3 - waiting}/3 ready)` : SUBTITLE;
    if (waiting) setTimeout(pollStatus, 2000);
  } catch {
    setTimeout(pollStatus, 4000);
  }
}
pollStatus();
