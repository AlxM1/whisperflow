/**
 * WhisperFlow PWA - Mobile voice dictation client
 * Features: dictation, raw transcription, command mode, history search,
 * snippets/dictionary management, waveform, swipe nav, tap/hold record,
 * dark/light theme, toast notifications, haptic feedback, keyboard shortcuts,
 * notification sounds, usage stats, backup/restore, dictionary import, history export.
 */

(function () {
  "use strict";

  // -- State --
  const state = {
    serverUrl: "",
    apiKey: "",
    language: "auto",
    recMode: "hold", // "hold" or "tap"
    theme: "dark",
    recording: false,
    processing: false,
    mediaRecorder: null,
    audioChunks: [],
    audioStream: null,
    audioContext: null,
    analyser: null,
    waveformAnimId: null,
    currentTab: "dictation",
    searchDebounce: null,
  };

  const TABS = ["dictation", "transcribe", "command", "history", "snippets", "dictionary", "stats"];

  // -- DOM refs --
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  // ================================================================
  // INIT
  // ================================================================
  function init() {
    loadSettings();
    applyTheme();
    setupEventListeners();
    registerServiceWorker();
    checkConnection();
  }

  function loadSettings() {
    state.serverUrl = localStorage.getItem("wf_server") || window.location.origin;
    state.apiKey = localStorage.getItem("wf_apikey") || "";
    state.language = localStorage.getItem("wf_language") || "auto";
    state.recMode = localStorage.getItem("wf_recmode") || "hold";
    state.theme = localStorage.getItem("wf_theme") || "dark";
    $("#setting-server").value = state.serverUrl;
    $("#setting-apikey").value = state.apiKey;
    $("#setting-language").value = state.language;
    $("#setting-rec-mode").value = state.recMode;
  }

  function saveSettings() {
    state.serverUrl = $("#setting-server").value || window.location.origin;
    state.apiKey = $("#setting-apikey").value;
    state.language = $("#setting-language").value;
    state.recMode = $("#setting-rec-mode").value;
    localStorage.setItem("wf_server", state.serverUrl);
    localStorage.setItem("wf_apikey", state.apiKey);
    localStorage.setItem("wf_language", state.language);
    localStorage.setItem("wf_recmode", state.recMode);
  }

  // ================================================================
  // THEME
  // ================================================================
  function applyTheme() {
    document.documentElement.setAttribute("data-theme", state.theme);
    const meta = document.getElementById("theme-color-meta");
    if (meta) meta.content = state.theme === "light" ? "#f5f5f7" : "#1a1a2e";
    $("#theme-icon-dark").classList.toggle("hidden", state.theme === "light");
    $("#theme-icon-light").classList.toggle("hidden", state.theme === "dark");
  }

  function toggleTheme() {
    state.theme = state.theme === "dark" ? "light" : "dark";
    localStorage.setItem("wf_theme", state.theme);
    applyTheme();
  }

  // ================================================================
  // CONNECTION
  // ================================================================
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
      if (state.serverUrl !== window.location.origin) {
        showScreen("auth");
      } else {
        showScreen("main");
      }
    }
  }

  // ================================================================
  // API HELPERS
  // ================================================================
  function apiFetch(path, options = {}) {
    const url = state.serverUrl + path;
    const headers = options.headers || {};
    if (state.apiKey) {
      headers["Authorization"] = "Bearer " + state.apiKey;
    }
    return fetch(url, { ...options, headers });
  }

  // ================================================================
  // SCREENS & TABS
  // ================================================================
  function showScreen(name) {
    $$("#auth-screen, #main-screen").forEach((el) => el.classList.add("hidden"));
    $(`#${name}-screen`).classList.remove("hidden");
  }

  function switchTab(mode) {
    state.currentTab = mode;
    $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));
    $$(".panel").forEach((p) => p.classList.add("hidden"));
    $(`#${mode}-panel`).classList.remove("hidden");
    $(`#${mode}-panel`).classList.add("active");

    // Scroll active tab into view
    const activeTab = $(`.tab[data-mode="${mode}"]`);
    if (activeTab) activeTab.scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });

    // Load data for management tabs
    if (mode === "history") loadHistory();
    if (mode === "snippets") loadSnippets();
    if (mode === "dictionary") loadDictionary();
    if (mode === "stats") loadStats();
  }

  // ================================================================
  // SWIPE GESTURES
  // ================================================================
  function setupSwipe() {
    const container = $("#panels-container");
    let startX = 0;
    let startY = 0;
    let swiping = false;

    container.addEventListener("touchstart", (e) => {
      // Don't intercept swipes on inputs/textareas/buttons
      if (e.target.closest("input, textarea, button, .record-btn, .manage-item")) return;
      startX = e.touches[0].clientX;
      startY = e.touches[0].clientY;
      swiping = true;
    }, { passive: true });

    container.addEventListener("touchend", (e) => {
      if (!swiping) return;
      swiping = false;
      const dx = e.changedTouches[0].clientX - startX;
      const dy = e.changedTouches[0].clientY - startY;

      // Only trigger for horizontal swipes > 60px and not too vertical
      if (Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy) * 1.5) {
        const idx = TABS.indexOf(state.currentTab);
        if (dx < 0 && idx < TABS.length - 1) {
          switchTab(TABS[idx + 1]);
        } else if (dx > 0 && idx > 0) {
          switchTab(TABS[idx - 1]);
        }
      }
    }, { passive: true });
  }

  // ================================================================
  // AUDIO RECORDING
  // ================================================================
  async function startRecording() {
    try {
      if (!state.audioStream) {
        state.audioStream = await navigator.mediaDevices.getUserMedia({
          audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
      }

      state.audioChunks = [];
      state.mediaRecorder = new MediaRecorder(state.audioStream, {
        mimeType: getSupportedMimeType(),
      });

      state.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) state.audioChunks.push(e.data);
      };

      state.mediaRecorder.start(100);
      state.recording = true;
      haptic();
      startWaveform();
    } catch (err) {
      console.error("Microphone error:", err);
      showToast("Microphone access denied. Check permissions.", "error");
    }
  }

  function stopRecording() {
    return new Promise((resolve) => {
      if (!state.mediaRecorder || state.mediaRecorder.state === "inactive") {
        resolve(null);
        return;
      }

      state.mediaRecorder.onstop = () => {
        const blob = new Blob(state.audioChunks, { type: state.mediaRecorder.mimeType });
        state.recording = false;
        haptic();
        stopWaveform();
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

  // ================================================================
  // WAVEFORM VISUALIZATION
  // ================================================================
  function startWaveform() {
    if (!state.audioStream) return;
    try {
      if (!state.audioContext) {
        state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      // Disconnect previous source to avoid leaking AudioNodes
      if (state._waveformSource) {
        try { state._waveformSource.disconnect(); } catch (_) {}
      }
      const source = state.audioContext.createMediaStreamSource(state.audioStream);
      state._waveformSource = source;
      state.analyser = state.audioContext.createAnalyser();
      state.analyser.fftSize = 256;
      source.connect(state.analyser);

      const canvasId = `waveform-${state.currentTab}`;
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      canvas.classList.remove("hidden");

      const ctx = canvas.getContext("2d");
      const bufferLength = state.analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);

      function draw() {
        if (!state.recording) return;
        state.waveformAnimId = requestAnimationFrame(draw);
        state.analyser.getByteFrequencyData(dataArray);

        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        const barW = Math.max(2, w / bufferLength);
        const gap = 1;
        const accentColor = state.currentTab === "command" ? "#7c5cbf"
          : state.currentTab === "transcribe" ? "#10b981" : "#e94560";

        for (let i = 0; i < bufferLength; i++) {
          const barH = (dataArray[i] / 255) * h * 0.9;
          const x = i * (barW + gap);
          if (x > w) break;
          ctx.fillStyle = accentColor;
          ctx.fillRect(x, h - barH, barW, barH);
        }
      }

      draw();
    } catch (err) {
      // AudioContext not available, skip waveform
    }
  }

  function stopWaveform() {
    if (state.waveformAnimId) {
      cancelAnimationFrame(state.waveformAnimId);
      state.waveformAnimId = null;
    }
    // Disconnect audio source to release resources
    if (state._waveformSource) {
      try { state._waveformSource.disconnect(); } catch (_) {}
      state._waveformSource = null;
    }
    // Hide all canvases
    $$(".waveform-canvas").forEach((c) => {
      const ctx = c.getContext("2d");
      ctx.clearRect(0, 0, c.width, c.height);
      c.classList.add("hidden");
    });
  }

  // ================================================================
  // HAPTIC FEEDBACK
  // ================================================================
  function haptic() {
    if (navigator.vibrate) {
      navigator.vibrate(30);
    }
  }

  // ================================================================
  // TOAST NOTIFICATIONS
  // ================================================================
  function showToast(msg, type = "info", duration = 3500) {
    const container = $("#toast-container");
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = msg;
    container.appendChild(toast);

    setTimeout(() => {
      toast.classList.add("toast-exit");
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  // ================================================================
  // DICTATION
  // ================================================================
  async function processDictation(audioBlob) {
    if (!audioBlob || audioBlob.size === 0) return;
    setRecordState("dictation", "processing");

    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");
    formData.append("language", state.language);

    try {
      const resp = await apiFetch("/api/dictate", { method: "POST", body: formData });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${resp.status}`);
      }
      const data = await resp.json();

      if (data.mode === "empty" || !data.text) {
        setRecordState("dictation", "idle");
        showToast("No speech detected", "warning");
        return;
      }

      showDictationResult(data.text, data.raw_text);
      setRecordState("dictation", "done");
      playNotificationSound();
      // Language memory: remember last detected language
      if (data.language && data.language !== state.language) {
        localStorage.setItem("wf_last_lang", data.language);
      }

      const info = [];
      if (data.language) info.push(data.language.toUpperCase());
      if (data.duration) info.push(data.duration.toFixed(1) + "s");
      if (data.processing_time) info.push("in " + data.processing_time + "s");
      $("#status-info").textContent = info.join(" | ");
    } catch (err) {
      console.error("Dictation error:", err);
      setRecordState("dictation", "idle");
      showToast(err.message, "error");
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

  // ================================================================
  // TRANSCRIBE (raw STT)
  // ================================================================
  async function processTranscribe(audioBlob) {
    if (!audioBlob || audioBlob.size === 0) return;
    setRecordState("transcribe", "processing");

    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");
    formData.append("language", state.language);

    try {
      const resp = await apiFetch("/api/transcribe", { method: "POST", body: formData });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${resp.status}`);
      }
      const data = await resp.json();

      if (!data.text) {
        setRecordState("transcribe", "idle");
        showToast("No speech detected", "warning");
        return;
      }

      $("#transcribe-placeholder").classList.add("hidden");
      const resultEl = $("#transcribe-text");
      resultEl.textContent = data.text;
      resultEl.classList.remove("hidden");
      $("#transcribe-copy-btn").classList.remove("hidden");
      if (navigator.share) $("#transcribe-share-btn").classList.remove("hidden");

      setRecordState("transcribe", "done");
      playNotificationSound();
      if (data.language) localStorage.setItem("wf_last_lang", data.language);

      const info = [];
      if (data.language) info.push(data.language.toUpperCase());
      if (data.confidence) info.push(Math.round(data.confidence * 100) + "%");
      if (data.duration) info.push(data.duration.toFixed(1) + "s");
      if (data.processing_time) info.push("in " + data.processing_time + "s");
      $("#transcribe-status-info").textContent = info.join(" | ");
    } catch (err) {
      console.error("Transcribe error:", err);
      setRecordState("transcribe", "idle");
      showToast(err.message, "error");
    }
  }

  // ================================================================
  // COMMAND MODE
  // ================================================================
  async function processCommand(audioBlob) {
    if (!audioBlob || audioBlob.size === 0) return;
    setRecordState("command", "processing");

    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");
    formData.append("selected_text", $("#selected-text").value);
    formData.append("language", state.language);

    try {
      const resp = await apiFetch("/api/command", { method: "POST", body: formData });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${resp.status}`);
      }
      const data = await resp.json();

      if (!data.text) {
        setRecordState("command", "idle");
        return;
      }

      $("#command-placeholder").classList.add("hidden");
      const resultEl = $("#command-result");
      resultEl.textContent = data.text;
      resultEl.classList.remove("hidden");
      $("#command-copy-btn").classList.remove("hidden");
      setRecordState("command", "done");
      playNotificationSound();
      $("#command-status-text").textContent = `Mode: ${data.mode}`;
    } catch (err) {
      console.error("Command error:", err);
      setRecordState("command", "idle");
      showToast(err.message, "error");
    }
  }

  // ================================================================
  // RECORD STATE (generic for all 3 modes)
  // ================================================================
  function setRecordState(mode, s) {
    const config = {
      dictation: { btn: "#record-btn", ring: "#audio-ring", status: "#status-text", ringClass: "active" },
      transcribe: { btn: "#transcribe-record-btn", ring: "#transcribe-audio-ring", status: "#transcribe-status-text", ringClass: "transcribe-active" },
      command: { btn: "#command-record-btn", ring: null, status: "#command-status-text", ringClass: "command-active" },
    }[mode];
    if (!config) return;

    const btn = $(config.btn);
    const ring = config.ring ? $(config.ring) : btn.closest(".audio-ring");
    const status = $(config.status);

    btn.classList.remove("active", "processing");
    ring.classList.remove("active", "transcribe-active", "command-active", "processing");

    switch (s) {
      case "recording":
        btn.classList.add("active");
        ring.classList.add(config.ringClass);
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

  // ================================================================
  // HISTORY
  // ================================================================
  async function loadHistory(searchQuery) {
    try {
      let resp;
      if (searchQuery) {
        resp = await apiFetch(`/api/history/search?q=${encodeURIComponent(searchQuery)}&limit=50`);
      } else {
        resp = await apiFetch("/api/history?limit=50");
      }
      if (!resp.ok) throw new Error("Failed to load history");
      const data = await resp.json();
      const list = $("#history-list");

      if (!data.entries || data.entries.length === 0) {
        list.innerHTML = `<div class="placeholder">${searchQuery ? "No results found" : "No history yet"}</div>`;
        return;
      }

      list.innerHTML = data.entries
        .map((e) => `
          <div class="history-item">
            <div class="text">${escapeHtml(e.processed_text || e.raw_text)}</div>
            <div class="history-meta">
              <span class="badge">${e.mode || "dictation"}</span>
              <span>${e.language || ""}</span>
              <span>${e.app_context || ""}</span>
              <span>${e.duration ? e.duration.toFixed(1) + "s" : ""}</span>
            </div>
            ${e.audio_path ? `<div class="history-audio"><audio controls preload="none" src="${state.serverUrl}/api/history/${e.id}/audio"></audio></div>` : ""}
          </div>`)
        .join("");
    } catch (err) {
      $("#history-list").innerHTML = '<div class="placeholder">Failed to load history</div>';
    }
  }

  // ================================================================
  // SNIPPETS MANAGEMENT
  // ================================================================
  async function loadSnippets() {
    try {
      const resp = await apiFetch("/api/snippets");
      if (!resp.ok) throw new Error("Failed to load snippets");
      const data = await resp.json();
      const list = $("#snippets-list");
      const snippets = data.snippets || {};
      const keys = Object.keys(snippets);

      if (keys.length === 0) {
        list.innerHTML = '<div class="placeholder">No snippets yet. Add voice shortcuts above.</div>';
        return;
      }

      list.innerHTML = keys.map((trigger) => {
        const snip = snippets[trigger];
        const content = typeof snip === "string" ? snip : snip.content || "";
        const desc = typeof snip === "object" ? snip.description || "" : "";
        return `
          <div class="manage-item" data-trigger="${escapeAttr(trigger)}">
            <div class="manage-item-content">
              <div class="trigger">"${escapeHtml(trigger)}"</div>
              <div class="content-preview">${escapeHtml(content)}</div>
              ${desc ? `<div class="content-preview">${escapeHtml(desc)}</div>` : ""}
            </div>
            <button class="delete-btn" data-action="delete-snippet" data-trigger="${escapeAttr(trigger)}" title="Delete">&#x2715;</button>
          </div>`;
      }).join("");
    } catch (err) {
      $("#snippets-list").innerHTML = '<div class="placeholder">Failed to load snippets</div>';
    }
  }

  async function addSnippet() {
    const trigger = $("#snippet-trigger").value.trim();
    const content = $("#snippet-content").value.trim();
    const desc = $("#snippet-desc").value.trim();
    if (!trigger || !content) {
      showToast("Trigger and content are required", "warning");
      return;
    }

    const formData = new FormData();
    formData.append("trigger", trigger);
    formData.append("content", content);
    formData.append("description", desc);

    try {
      const resp = await apiFetch("/api/snippets", { method: "POST", body: formData });
      if (!resp.ok) throw new Error("Failed to save");
      showToast("Snippet saved", "success");
      $("#snippet-form").classList.add("hidden");
      $("#snippet-trigger").value = "";
      $("#snippet-content").value = "";
      $("#snippet-desc").value = "";
      loadSnippets();
    } catch (err) {
      showToast("Failed to save snippet", "error");
    }
  }

  async function deleteSnippet(trigger) {
    try {
      const resp = await apiFetch(`/api/snippets/${encodeURIComponent(trigger)}`, { method: "DELETE" });
      if (!resp.ok) throw new Error("Failed to delete");
      showToast("Snippet deleted", "success");
      loadSnippets();
    } catch (err) {
      showToast("Failed to delete snippet", "error");
    }
  }

  // ================================================================
  // DICTIONARY MANAGEMENT
  // ================================================================
  async function loadDictionary() {
    try {
      const resp = await apiFetch("/api/dictionary");
      if (!resp.ok) throw new Error("Failed to load dictionary");
      const data = await resp.json();
      const list = $("#dictionary-list");
      const words = data.words || {};
      const keys = Object.keys(words);

      if (keys.length === 0) {
        list.innerHTML = '<div class="placeholder">No custom words yet. Add words to improve transcription accuracy.</div>';
        return;
      }

      list.innerHTML = keys.sort().map((word) => `
        <div class="manage-item" data-word="${escapeAttr(word)}">
          <div class="manage-item-content">
            <div class="word">${escapeHtml(word)}</div>
            <div class="category">${escapeHtml(words[word] || "")}</div>
          </div>
          <button class="delete-btn" data-action="delete-word" data-word="${escapeAttr(word)}" title="Delete">&#x2715;</button>
        </div>`
      ).join("");
    } catch (err) {
      $("#dictionary-list").innerHTML = '<div class="placeholder">Failed to load dictionary</div>';
    }
  }

  async function addWord() {
    const word = $("#word-input").value.trim();
    if (!word) {
      showToast("Enter a word to add", "warning");
      return;
    }

    const formData = new FormData();
    formData.append("word", word);

    try {
      const resp = await apiFetch("/api/dictionary", { method: "POST", body: formData });
      if (!resp.ok) throw new Error("Failed to save");
      showToast(`"${word}" added to dictionary`, "success");
      $("#word-form").classList.add("hidden");
      $("#word-input").value = "";
      loadDictionary();
    } catch (err) {
      showToast("Failed to add word", "error");
    }
  }

  async function deleteWord(word) {
    try {
      const resp = await apiFetch(`/api/dictionary/${encodeURIComponent(word)}`, { method: "DELETE" });
      if (!resp.ok) throw new Error("Failed to delete");
      showToast(`"${word}" removed`, "success");
      loadDictionary();
    } catch (err) {
      showToast("Failed to delete word", "error");
    }
  }

  // ================================================================
  // UTILITY
  // ================================================================
  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str || "";
    return div.innerHTML;
  }

  function escapeAttr(str) {
    return (str || "").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      showToast("Copied to clipboard", "success", 1500);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      showToast("Copied to clipboard", "success", 1500);
    }
  }

  // ================================================================
  // NOTIFICATION SOUND
  // ================================================================
  function playNotificationSound() {
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = 880;
      osc.type = "sine";
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.3);
      setTimeout(() => ctx.close(), 500);
    } catch (_) {}
  }

  // ================================================================
  // STATS DASHBOARD
  // ================================================================
  async function loadStats() {
    const container = $("#stats-content");
    try {
      const resp = await apiFetch("/api/stats");
      if (!resp.ok) throw new Error("Failed to load stats");
      const s = await resp.json();

      const mins = Math.round(s.total_duration_seconds / 60);
      const recentMins = Math.round((s.last_7_days?.duration_seconds || 0) / 60);

      let modesHtml = Object.entries(s.modes || {}).map(([m, c]) =>
        `<div class="stat-breakdown-item"><span class="label">${escapeHtml(m)}</span><span class="value">${c}</span></div>`
      ).join("");

      let langsHtml = Object.entries(s.languages || {}).map(([l, c]) =>
        `<div class="stat-breakdown-item"><span class="label">${l.toUpperCase()}</span><span class="value">${c}</span></div>`
      ).join("");

      container.innerHTML = `
        <div class="stat-card">
          <div class="stat-value">${s.total_transcriptions || 0}</div>
          <div class="stat-label">Total Transcriptions</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${mins}</div>
          <div class="stat-label">Minutes Transcribed</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${(s.total_words || 0).toLocaleString()}</div>
          <div class="stat-label">Total Words</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${s.last_7_days?.transcriptions || 0}</div>
          <div class="stat-label">Last 7 Days</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${s.dictionary_words || 0}</div>
          <div class="stat-label">Dictionary Words</div>
        </div>
        <div class="stat-card">
          <div class="stat-value">${s.snippets_count || 0}</div>
          <div class="stat-label">Snippets</div>
        </div>
        ${modesHtml ? `<div class="stat-card wide"><div class="stat-label" style="margin-bottom:8px;">BY MODE</div><div class="stat-breakdown">${modesHtml}</div></div>` : ""}
        ${langsHtml ? `<div class="stat-card wide"><div class="stat-label" style="margin-bottom:8px;">TOP LANGUAGES</div><div class="stat-breakdown">${langsHtml}</div></div>` : ""}
      `;
    } catch (err) {
      container.innerHTML = '<div class="placeholder">Failed to load stats</div>';
    }
  }

  // ================================================================
  // HISTORY EXPORT
  // ================================================================
  async function exportHistory() {
    try {
      const resp = await apiFetch("/api/history/export?format=json");
      if (!resp.ok) throw new Error("Export failed");
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "whisperflow_history.json";
      a.click();
      URL.revokeObjectURL(url);
      showToast("History exported", "success");
    } catch (err) {
      showToast("Failed to export history", "error");
    }
  }

  // ================================================================
  // DICTIONARY IMPORT
  // ================================================================
  async function importDictionary(file) {
    const formData = new FormData();
    formData.append("file", file);
    try {
      const resp = await apiFetch("/api/dictionary/import", { method: "POST", body: formData });
      if (!resp.ok) throw new Error("Import failed");
      const data = await resp.json();
      showToast(`Imported ${data.imported} words`, "success");
      loadDictionary();
    } catch (err) {
      showToast("Failed to import dictionary", "error");
    }
  }

  // ================================================================
  // BACKUP & RESTORE
  // ================================================================
  async function downloadBackup() {
    try {
      const resp = await apiFetch("/api/backup");
      if (!resp.ok) throw new Error("Backup failed");
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "whisperflow_backup.zip";
      a.click();
      URL.revokeObjectURL(url);
      showToast("Backup downloaded", "success");
    } catch (err) {
      showToast("Failed to create backup", "error");
    }
  }

  async function restoreBackup(file) {
    const formData = new FormData();
    formData.append("file", file);
    try {
      const resp = await apiFetch("/api/restore", { method: "POST", body: formData });
      if (!resp.ok) throw new Error("Restore failed");
      const data = await resp.json();
      showToast(`Restored: ${data.restored.join(", ")}`, "success");
      loadStats();
    } catch (err) {
      showToast("Failed to restore backup", "error");
    }
  }

  // ================================================================
  // KEYBOARD SHORTCUTS
  // ================================================================
  function setupKeyboardShortcuts() {
    let spaceRecording = false;

    document.addEventListener("keydown", (e) => {
      // Ignore when typing in inputs
      if (e.target.closest("input, textarea, select")) return;

      if (e.code === "Space" && !e.repeat) {
        e.preventDefault();
        if (state.processing) return;
        const mode = state.currentTab;
        if (!["dictation", "transcribe", "command"].includes(mode)) return;

        if (state.recMode === "tap") {
          // Tap mode: toggle recording
          const btnMap = { dictation: "#record-btn", transcribe: "#transcribe-record-btn", command: "#command-record-btn" };
          const btn = $(btnMap[mode]);
          if (btn) btn.click();
        } else {
          // Hold mode: start on keydown
          if (!spaceRecording) {
            spaceRecording = true;
            startRecording();
            setRecordState(mode, "recording");
          }
        }
      }
    });

    document.addEventListener("keyup", (e) => {
      if (e.target.closest("input, textarea, select")) return;

      if (e.code === "Space" && spaceRecording) {
        e.preventDefault();
        spaceRecording = false;
        const mode = state.currentTab;
        if (!["dictation", "transcribe", "command"].includes(mode)) return;
        if (!state.recording) return;

        const callbacks = { dictation: processDictation, transcribe: processTranscribe, command: processCommand };
        (async () => {
          state.processing = true;
          const blob = await stopRecording();
          await callbacks[mode](blob);
          state.processing = false;
        })();
      }
    });
  }

  // ================================================================
  // RECORD BUTTON HELPERS (hold + tap modes)
  // ================================================================
  function setupRecordButton(btnSel, mode, processCallback) {
    const btn = $(btnSel);
    let recTimeout;
    let tapRecording = false;

    function onStart(e) {
      e.preventDefault();
      if (state.processing) return;

      if (state.recMode === "tap") {
        if (tapRecording) {
          // Second tap: stop
          tapRecording = false;
          onFinish();
        } else {
          // First tap: start
          tapRecording = true;
          startRecording();
          setRecordState(mode, "recording");
        }
      } else {
        // Hold mode: start after small delay
        recTimeout = setTimeout(() => {
          startRecording();
          setRecordState(mode, "recording");
        }, 100);
      }
    }

    async function onFinish() {
      if (state.recMode === "tap" && tapRecording) return; // Still recording in tap mode
      clearTimeout(recTimeout);
      if (!state.recording) return;
      state.processing = true;
      const blob = await stopRecording();
      await processCallback(blob);
      state.processing = false;
    }

    function onEnd(e) {
      e.preventDefault();
      if (state.recMode === "hold") {
        onFinish();
      }
      // For tap mode, do nothing on release (handled by onStart toggle)
    }

    btn.addEventListener("touchstart", onStart, { passive: false });
    btn.addEventListener("mousedown", onStart);
    btn.addEventListener("touchend", onEnd);
    btn.addEventListener("touchcancel", onEnd);
    btn.addEventListener("mouseup", onEnd);
    btn.addEventListener("mouseleave", onEnd);
  }

  // ================================================================
  // EVENT LISTENERS
  // ================================================================
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

    // Theme toggle
    $("#theme-toggle").addEventListener("click", toggleTheme);

    // Tabs
    $$(".tab").forEach((tab) =>
      tab.addEventListener("click", () => switchTab(tab.dataset.mode))
    );

    // Record buttons
    setupRecordButton("#record-btn", "dictation", processDictation);
    setupRecordButton("#transcribe-record-btn", "transcribe", processTranscribe);
    setupRecordButton("#command-record-btn", "command", processCommand);

    // Copy / Share — Dictation
    $("#copy-btn").addEventListener("click", () => copyText($("#result-text").textContent));
    $("#share-btn").addEventListener("click", () => {
      navigator.share({ text: $("#result-text").textContent }).catch(() => {});
    });

    // Copy / Share — Transcribe
    $("#transcribe-copy-btn").addEventListener("click", () => copyText($("#transcribe-text").textContent));
    $("#transcribe-share-btn").addEventListener("click", () => {
      navigator.share({ text: $("#transcribe-text").textContent }).catch(() => {});
    });

    // Copy — Command
    $("#command-copy-btn").addEventListener("click", () => copyText($("#command-result").textContent));

    // History search (debounced)
    $("#history-search").addEventListener("input", (e) => {
      clearTimeout(state.searchDebounce);
      state.searchDebounce = setTimeout(() => {
        loadHistory(e.target.value.trim());
      }, 400);
    });

    // Snippets management
    $("#add-snippet-btn").addEventListener("click", () => {
      $("#snippet-form").classList.toggle("hidden");
    });
    $("#snippet-save").addEventListener("click", addSnippet);
    $("#snippet-cancel").addEventListener("click", () => {
      $("#snippet-form").classList.add("hidden");
    });

    // Dictionary management
    $("#add-word-btn").addEventListener("click", () => {
      $("#word-form").classList.toggle("hidden");
    });
    $("#word-save").addEventListener("click", addWord);
    $("#word-cancel").addEventListener("click", () => {
      $("#word-form").classList.add("hidden");
    });

    // Delegated delete handlers for snippets & dictionary
    document.addEventListener("click", (e) => {
      const del = e.target.closest("[data-action]");
      if (!del) return;

      if (del.dataset.action === "delete-snippet") {
        deleteSnippet(del.dataset.trigger);
      } else if (del.dataset.action === "delete-word") {
        deleteWord(del.dataset.word);
      }
    });

    // Settings
    $("#settings-btn").addEventListener("click", () => $("#settings-modal").classList.remove("hidden"));
    $("#settings-close").addEventListener("click", () => $("#settings-modal").classList.add("hidden"));
    $("#settings-modal").addEventListener("click", (e) => {
      if (e.target === $("#settings-modal")) $("#settings-modal").classList.add("hidden");
    });

    $("#settings-save").addEventListener("click", () => {
      saveSettings();
      $("#settings-modal").classList.add("hidden");
      showToast("Settings saved", "success", 1500);
    });

    $("#settings-logout").addEventListener("click", () => {
      localStorage.clear();
      state.apiKey = "";
      state.serverUrl = window.location.origin;
      showScreen("auth");
      $("#settings-modal").classList.add("hidden");
    });

    // History export
    $("#export-history-btn").addEventListener("click", exportHistory);

    // Dictionary import
    $("#import-dict-btn").addEventListener("click", () => {
      $("#dict-file-input").click();
    });
    $("#dict-file-input").addEventListener("change", (e) => {
      if (e.target.files[0]) {
        importDictionary(e.target.files[0]);
        e.target.value = "";
      }
    });

    // Backup & Restore
    $("#backup-btn").addEventListener("click", downloadBackup);
    $("#restore-file-input").addEventListener("change", (e) => {
      if (e.target.files[0]) {
        restoreBackup(e.target.files[0]);
        e.target.value = "";
      }
    });

    // Prevent context menu on long press
    document.addEventListener("contextmenu", (e) => {
      if (e.target.closest(".record-btn")) e.preventDefault();
    });

    // Setup swipe gestures & keyboard shortcuts
    setupSwipe();
    setupKeyboardShortcuts();
  }

  // ================================================================
  // SERVICE WORKER
  // ================================================================
  function registerServiceWorker() {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(() => {});
    }
  }

  // Start
  document.addEventListener("DOMContentLoaded", init);
})();
