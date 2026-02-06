/**
 * WhisperFlow PWA - Mobile voice dictation client
 */

(function () {
  "use strict";

  // -- State --
  const state = {
    serverUrl: "",
    apiKey: "",
    language: "auto",
    recording: false,
    processing: false,
    mediaRecorder: null,
    audioChunks: [],
    audioStream: null,
  };

  // -- DOM refs --
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  // -- Init --
  function init() {
    loadSettings();
    setupEventListeners();
    registerServiceWorker();
    checkConnection();
  }

  function loadSettings() {
    state.serverUrl = localStorage.getItem("wf_server") || window.location.origin;
    state.apiKey = localStorage.getItem("wf_apikey") || "";
    state.language = localStorage.getItem("wf_language") || "auto";
    $("#setting-server").value = state.serverUrl;
    $("#setting-apikey").value = state.apiKey;
    $("#setting-language").value = state.language;
  }

  function saveSettings() {
    state.serverUrl = $("#setting-server").value || window.location.origin;
    state.apiKey = $("#setting-apikey").value;
    state.language = $("#setting-language").value;
    localStorage.setItem("wf_server", state.serverUrl);
    localStorage.setItem("wf_apikey", state.apiKey);
    localStorage.setItem("wf_language", state.language);
  }

  // -- Connection check --
  async function checkConnection() {
    try {
      const resp = await apiFetch("/api/health");
      const data = await resp.json();
      if (data.auth_required && !state.apiKey) {
        showScreen("auth");
      } else {
        showScreen("main");
      }
    } catch (e) {
      // Server unreachable or wrong URL — show auth/settings
      if (state.serverUrl !== window.location.origin) {
        showScreen("auth");
      } else {
        showScreen("main");
      }
    }
  }

  // -- API helpers --
  function apiFetch(path, options = {}) {
    const url = state.serverUrl + path;
    const headers = options.headers || {};
    if (state.apiKey) {
      headers["Authorization"] = "Bearer " + state.apiKey;
    }
    return fetch(url, { ...options, headers });
  }

  // -- Screens --
  function showScreen(name) {
    $$("#auth-screen, #main-screen").forEach((el) => el.classList.add("hidden"));
    $(`#${name}-screen`).classList.remove("hidden");
  }

  // -- Tabs --
  function switchTab(mode) {
    $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));
    $$("#dictation-panel, #command-panel, #history-panel").forEach((p) =>
      p.classList.add("hidden")
    );
    $(`#${mode}-panel`).classList.remove("hidden");
    $(`#${mode}-panel`).classList.add("active");

    if (mode === "history") loadHistory();
  }

  // -- Audio Recording --
  async function startRecording() {
    try {
      if (!state.audioStream) {
        state.audioStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            sampleRate: 16000,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
          },
        });
      }

      state.audioChunks = [];
      state.mediaRecorder = new MediaRecorder(state.audioStream, {
        mimeType: getSupportedMimeType(),
      });

      state.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) state.audioChunks.push(e.data);
      };

      state.mediaRecorder.start(100); // 100ms chunks
      state.recording = true;
    } catch (err) {
      console.error("Microphone error:", err);
      showError("Microphone access denied. Check permissions.");
    }
  }

  function stopRecording() {
    return new Promise((resolve) => {
      if (!state.mediaRecorder || state.mediaRecorder.state === "inactive") {
        resolve(null);
        return;
      }

      state.mediaRecorder.onstop = () => {
        const blob = new Blob(state.audioChunks, {
          type: state.mediaRecorder.mimeType,
        });
        state.recording = false;
        resolve(blob);
      };

      state.mediaRecorder.stop();
    });
  }

  function getSupportedMimeType() {
    const types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
    for (const type of types) {
      if (MediaRecorder.isTypeSupported(type)) return type;
    }
    return "audio/webm";
  }

  // -- Dictation --
  async function processDictation(audioBlob) {
    if (!audioBlob || audioBlob.size === 0) return;

    setDictationState("processing");

    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");
    formData.append("language", state.language);

    try {
      const resp = await apiFetch("/api/dictate", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${resp.status}`);
      }

      const data = await resp.json();

      if (data.mode === "empty" || !data.text) {
        setDictationState("idle");
        showDictationResult("(No speech detected)", "");
        return;
      }

      showDictationResult(data.text, data.raw_text);
      setDictationState("done");

      const info = [];
      if (data.language) info.push(data.language.toUpperCase());
      if (data.duration) info.push(data.duration.toFixed(1) + "s");
      if (data.processing_time) info.push("in " + data.processing_time + "s");
      $("#status-info").textContent = info.join(" | ");

    } catch (err) {
      console.error("Dictation error:", err);
      setDictationState("idle");
      showError(err.message);
    }
  }

  function showDictationResult(text, rawText) {
    $("#result-placeholder").classList.add("hidden");
    const resultEl = $("#result-text");
    resultEl.textContent = text;
    resultEl.classList.remove("hidden");

    if (rawText && rawText !== text) {
      const rawEl = $("#raw-text");
      rawEl.textContent = "Raw: " + rawText;
      rawEl.classList.remove("hidden");
    } else {
      $("#raw-text").classList.add("hidden");
    }

    $("#copy-btn").classList.remove("hidden");
    if (navigator.share) $("#share-btn").classList.remove("hidden");
  }

  function setDictationState(s) {
    const btn = $("#record-btn");
    const ring = $("#audio-ring");
    const status = $("#status-text");

    btn.classList.remove("active", "processing");
    ring.classList.remove("active", "processing");

    switch (s) {
      case "recording":
        btn.classList.add("active");
        ring.classList.add("active");
        status.textContent = "Listening...";
        break;
      case "processing":
        btn.classList.add("processing");
        ring.classList.add("processing");
        status.textContent = "Processing...";
        break;
      case "done":
        status.textContent = "Done";
        break;
      default:
        status.textContent = "Ready";
    }
  }

  // -- Command Mode --
  async function processCommand(audioBlob) {
    if (!audioBlob || audioBlob.size === 0) return;

    setCommandState("processing");

    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");
    formData.append("selected_text", $("#selected-text").value);
    formData.append("language", state.language);

    try {
      const resp = await apiFetch("/api/command", {
        method: "POST",
        body: formData,
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${resp.status}`);
      }

      const data = await resp.json();

      if (!data.text) {
        setCommandState("idle");
        return;
      }

      $("#command-placeholder").classList.add("hidden");
      const resultEl = $("#command-result");
      resultEl.textContent = data.text;
      resultEl.classList.remove("hidden");
      $("#command-copy-btn").classList.remove("hidden");
      setCommandState("done");
      $("#command-status-text").textContent = `Mode: ${data.mode}`;

    } catch (err) {
      console.error("Command error:", err);
      setCommandState("idle");
      showError(err.message);
    }
  }

  function setCommandState(s) {
    const btn = $("#command-record-btn");

    btn.classList.remove("active", "processing");

    switch (s) {
      case "recording":
        btn.classList.add("active");
        btn.closest(".audio-ring").classList.add("command-active");
        $("#command-status-text").textContent = "Listening...";
        break;
      case "processing":
        btn.classList.add("processing");
        btn.closest(".audio-ring").classList.remove("command-active");
        btn.closest(".audio-ring").classList.add("processing");
        $("#command-status-text").textContent = "Processing...";
        break;
      case "done":
        btn.closest(".audio-ring").classList.remove("command-active", "processing");
        break;
      default:
        btn.closest(".audio-ring").classList.remove("command-active", "processing");
        $("#command-status-text").textContent = "Ready";
    }
  }

  // -- History --
  async function loadHistory() {
    try {
      const resp = await apiFetch("/api/history?limit=50");
      const data = await resp.json();

      const list = $("#history-list");
      if (!data.entries || data.entries.length === 0) {
        list.innerHTML = '<div class="placeholder">No history yet</div>';
        return;
      }

      list.innerHTML = data.entries
        .map(
          (e) => `
        <div class="history-item">
          <div class="text">${escapeHtml(e.processed_text || e.raw_text)}</div>
          <div class="history-meta">
            <span class="badge">${e.mode || "dictation"}</span>
            <span>${e.language || ""}</span>
            <span>${e.app_context || ""}</span>
            <span>${e.duration ? e.duration.toFixed(1) + "s" : ""}</span>
          </div>
        </div>`
        )
        .join("");
    } catch (err) {
      $("#history-list").innerHTML = `<div class="placeholder">Failed to load history</div>`;
    }
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // -- Utility --
  function showError(msg) {
    // Briefly show error in status
    const status = $("#status-text");
    const prev = status.textContent;
    status.textContent = msg;
    status.style.color = "var(--error)";
    setTimeout(() => {
      status.textContent = prev === msg ? "Ready" : prev;
      status.style.color = "";
    }, 4000);
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Fallback for older browsers
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
  }

  // -- Event Listeners --
  function setupEventListeners() {
    // Auth
    $("#auth-submit").addEventListener("click", async () => {
      const key = $("#api-key-input").value;
      state.apiKey = key;
      localStorage.setItem("wf_apikey", key);
      try {
        const resp = await apiFetch("/api/health");
        if (resp.ok) {
          showScreen("main");
        } else {
          $("#auth-error").textContent = "Invalid API key";
          $("#auth-error").classList.remove("hidden");
        }
      } catch {
        $("#auth-error").textContent = "Cannot connect to server";
        $("#auth-error").classList.remove("hidden");
      }
    });

    // Tabs
    $$(".tab").forEach((tab) =>
      tab.addEventListener("click", () => switchTab(tab.dataset.mode))
    );

    // Dictation record button - hold to record
    const recBtn = $("#record-btn");
    let recTimeout;

    function startDictationRec(e) {
      e.preventDefault();
      if (state.processing) return;
      recTimeout = setTimeout(() => {
        startRecording();
        setDictationState("recording");
      }, 100);
    }

    async function stopDictationRec(e) {
      e.preventDefault();
      clearTimeout(recTimeout);
      if (!state.recording) return;
      state.processing = true;
      const blob = await stopRecording();
      await processDictation(blob);
      state.processing = false;
    }

    recBtn.addEventListener("touchstart", startDictationRec, { passive: false });
    recBtn.addEventListener("mousedown", startDictationRec);
    recBtn.addEventListener("touchend", stopDictationRec);
    recBtn.addEventListener("touchcancel", stopDictationRec);
    recBtn.addEventListener("mouseup", stopDictationRec);
    recBtn.addEventListener("mouseleave", stopDictationRec);

    // Command record button
    const cmdBtn = $("#command-record-btn");

    function startCommandRec(e) {
      e.preventDefault();
      if (state.processing) return;
      recTimeout = setTimeout(() => {
        startRecording();
        setCommandState("recording");
      }, 100);
    }

    async function stopCommandRec(e) {
      e.preventDefault();
      clearTimeout(recTimeout);
      if (!state.recording) return;
      state.processing = true;
      const blob = await stopRecording();
      await processCommand(blob);
      state.processing = false;
    }

    cmdBtn.addEventListener("touchstart", startCommandRec, { passive: false });
    cmdBtn.addEventListener("mousedown", startCommandRec);
    cmdBtn.addEventListener("touchend", stopCommandRec);
    cmdBtn.addEventListener("touchcancel", stopCommandRec);
    cmdBtn.addEventListener("mouseup", stopCommandRec);
    cmdBtn.addEventListener("mouseleave", stopCommandRec);

    // Copy / Share
    $("#copy-btn").addEventListener("click", () => {
      const text = $("#result-text").textContent;
      copyText(text);
      $("#copy-btn").textContent = "Copied!";
      setTimeout(() => ($("#copy-btn").textContent = "Copy Text"), 1500);
    });

    $("#share-btn").addEventListener("click", () => {
      const text = $("#result-text").textContent;
      navigator.share({ text }).catch(() => {});
    });

    $("#command-copy-btn").addEventListener("click", () => {
      const text = $("#command-result").textContent;
      copyText(text);
      $("#command-copy-btn").textContent = "Copied!";
      setTimeout(() => ($("#command-copy-btn").textContent = "Copy Result"), 1500);
    });

    // Settings
    $("#settings-btn").addEventListener("click", () =>
      $("#settings-modal").classList.remove("hidden")
    );
    $("#settings-close").addEventListener("click", () =>
      $("#settings-modal").classList.add("hidden")
    );
    $("#settings-modal").addEventListener("click", (e) => {
      if (e.target === $("#settings-modal"))
        $("#settings-modal").classList.add("hidden");
    });

    $("#settings-save").addEventListener("click", () => {
      saveSettings();
      $("#settings-modal").classList.add("hidden");
    });

    $("#settings-logout").addEventListener("click", () => {
      localStorage.clear();
      state.apiKey = "";
      state.serverUrl = window.location.origin;
      showScreen("auth");
      $("#settings-modal").classList.add("hidden");
    });

    // Prevent context menu on long press
    document.addEventListener("contextmenu", (e) => {
      if (e.target.closest(".record-btn")) e.preventDefault();
    });
  }

  // -- Service Worker --
  function registerServiceWorker() {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(() => {});
    }
  }

  // Start
  document.addEventListener("DOMContentLoaded", init);
})();
