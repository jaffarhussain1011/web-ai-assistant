/**
 * chat-widget.js
 * ──────────────
 * Self-contained, embeddable AI chat widget.
 *
 * Drop into any page:
 *   <script src="chat-widget.js"></script>
 *   <script>
 *     window.ChatWidget.init({ apiUrl: "http://localhost:8000/ask" });
 *   </script>
 *
 * The widget injects its own CSS so no external stylesheet is needed.
 */

(function (global) {
  "use strict";

  /* ── Default configuration ─────────────────────────────────────────────── */
  const DEFAULTS = {
    apiUrl:         "http://localhost:8000/ask",
    title:          "AI Assistant",
    placeholder:    "Ask a question…",
    primaryColor:   "#4F46E5",      // Indigo
    position:       "bottom-right", // bottom-right | bottom-left
    top_k:          5,              // context chunks to retrieve
    welcomeMessage: "👋 Hello! How can I help you today?",
  };

  /* ── CSS injected into the page ─────────────────────────────────────────── */
  const CSS = `
    #cw-launcher {
      position: fixed;
      z-index: 99998;
      width: 56px;
      height: 56px;
      border-radius: 50%;
      border: none;
      cursor: pointer;
      box-shadow: 0 4px 16px rgba(0,0,0,0.25);
      display: flex;
      align-items: center;
      justify-content: center;
      transition: transform 0.2s, box-shadow 0.2s;
      background: var(--cw-primary);
    }
    #cw-launcher:hover {
      transform: scale(1.07);
      box-shadow: 0 6px 20px rgba(0,0,0,0.32);
    }
    #cw-launcher svg { pointer-events: none; }

    #cw-popup {
      position: fixed;
      z-index: 99999;
      width: 370px;
      max-width: calc(100vw - 24px);
      height: 540px;
      max-height: calc(100vh - 100px);
      border-radius: 16px;
      box-shadow: 0 12px 40px rgba(0,0,0,0.22);
      display: flex;
      flex-direction: column;
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-size: 14px;
      background: #fff;
      transition: opacity 0.2s, transform 0.2s;
    }
    #cw-popup.cw-hidden {
      opacity: 0;
      pointer-events: none;
      transform: translateY(12px) scale(0.97);
    }

    #cw-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 14px 16px;
      color: #fff;
      font-weight: 600;
      font-size: 15px;
      background: var(--cw-primary);
      flex-shrink: 0;
    }
    #cw-header-left { display: flex; align-items: center; gap: 10px; }
    #cw-avatar {
      width: 32px; height: 32px; border-radius: 50%;
      background: rgba(255,255,255,0.25);
      display: flex; align-items: center; justify-content: center;
    }
    #cw-close-btn {
      background: none; border: none; cursor: pointer;
      color: #fff; font-size: 20px; line-height: 1;
      opacity: 0.8; padding: 0 4px;
    }
    #cw-close-btn:hover { opacity: 1; }

    #cw-messages {
      flex: 1;
      overflow-y: auto;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      background: #f9fafb;
    }
    #cw-messages::-webkit-scrollbar { width: 4px; }
    #cw-messages::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }

    .cw-msg {
      display: flex;
      flex-direction: column;
      max-width: 88%;
      word-break: break-word;
      line-height: 1.5;
    }
    .cw-msg.cw-user  { align-self: flex-end; align-items: flex-end; }
    .cw-msg.cw-bot   { align-self: flex-start; align-items: flex-start; }
    .cw-msg.cw-error { align-self: flex-start; }

    .cw-bubble {
      padding: 10px 14px;
      border-radius: 14px;
      white-space: pre-wrap;
    }
    .cw-user  .cw-bubble { background: var(--cw-primary); color: #fff; border-bottom-right-radius: 4px; }
    .cw-bot   .cw-bubble { background: #fff; color: #111; border: 1px solid #e5e7eb; border-bottom-left-radius: 4px; }
    .cw-error .cw-bubble { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; font-size: 13px; }

    .cw-meta { font-size: 11px; color: #9ca3af; margin-top: 3px; padding: 0 4px; }

    .cw-typing .cw-bubble {
      background: #fff;
      border: 1px solid #e5e7eb;
      display: flex;
      gap: 5px;
      align-items: center;
      padding: 12px 16px;
    }
    .cw-dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: #9ca3af;
      animation: cw-bounce 1.2s infinite;
    }
    .cw-dot:nth-child(2) { animation-delay: 0.2s; }
    .cw-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes cw-bounce {
      0%, 60%, 100% { transform: translateY(0); }
      30%           { transform: translateY(-6px); }
    }

    #cw-input-area {
      display: flex;
      padding: 10px 12px;
      gap: 8px;
      border-top: 1px solid #e5e7eb;
      background: #fff;
      flex-shrink: 0;
    }
    #cw-input {
      flex: 1;
      border: 1px solid #d1d5db;
      border-radius: 10px;
      padding: 9px 12px;
      font-size: 14px;
      outline: none;
      resize: none;
      min-height: 40px;
      max-height: 120px;
      font-family: inherit;
      line-height: 1.4;
      transition: border-color 0.15s;
    }
    #cw-input:focus { border-color: var(--cw-primary); }
    #cw-send-btn {
      width: 40px; height: 40px;
      border-radius: 10px;
      border: none;
      cursor: pointer;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      background: var(--cw-primary);
      color: #fff;
      transition: opacity 0.15s;
    }
    #cw-send-btn:disabled { opacity: 0.45; cursor: default; }
    #cw-send-btn:not(:disabled):hover { opacity: 0.88; }

    #cw-powered {
      text-align: center;
      font-size: 11px;
      color: #9ca3af;
      padding: 5px;
      background: #fff;
      border-top: 1px solid #f3f4f6;
      flex-shrink: 0;
    }
  `;

  /* ── Icons (inline SVG) ─────────────────────────────────────────────────── */
  const ICON_CHAT = `<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>`;
  const ICON_SEND = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>`;
  const ICON_BOT  = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.9)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>`;

  /* ── ChatWidget ─────────────────────────────────────────────────────────── */
  const ChatWidget = {
    _cfg: null,
    _open: false,
    _busy: false,
    _history: [],   // [{role, text, ts}]

    /* ── init ─────────────────────────────────────────────────────────── */
    init(userCfg = {}) {
      // Merge order: DEFAULTS < server config < userCfg
      // Server config is fetched async; widget renders immediately with
      // DEFAULTS + userCfg, then updates colour/title if server responds.
      this._cfg = Object.assign({}, DEFAULTS, userCfg);
      this._injectCSS();
      this._buildDOM();
      this._bindEvents();
      this._addBotMessage(this._cfg.welcomeMessage);
      this._fetchServerConfig(userCfg);
    },

    /* ── Fetch server widget config ───────────────────────────────────── */
    _fetchServerConfig(userCfg) {
      try {
        const base = new URL(this._cfg.apiUrl).origin;
        fetch(`${base}/widget/config`)
          .then(r => r.ok ? r.json() : null)
          .then(serverCfg => {
            if (!serverCfg) return;
            // Re-merge: DEFAULTS < server < userCfg (userCfg always wins)
            const merged = Object.assign({}, DEFAULTS, serverCfg, userCfg);
            this._cfg = merged;
            // Apply dynamic updates to already-rendered DOM
            document.documentElement.style.setProperty("--cw-primary", merged.primaryColor);
            const header = this._popup.querySelector("#cw-header span");
            if (header) header.textContent = merged.title;
            const inp = this._popup.querySelector("#cw-input");
            if (inp) inp.placeholder = merged.placeholder;
          })
          .catch(() => { /* server unreachable — keep local defaults */ });
      } catch (_) { /* invalid apiUrl — skip */ }
    },

    /* ── DOM construction ─────────────────────────────────────────────── */
    _injectCSS() {
      const style = document.createElement("style");
      style.textContent = CSS;
      document.head.appendChild(style);
      document.documentElement.style.setProperty("--cw-primary", this._cfg.primaryColor);
    },

    _buildDOM() {
      const isRight = this._cfg.position !== "bottom-left";
      const side    = isRight ? "right: 20px" : "left: 20px";

      // Launcher button
      this._launcher = document.createElement("button");
      this._launcher.id = "cw-launcher";
      this._launcher.innerHTML = ICON_CHAT;
      this._launcher.style.cssText = `bottom: 20px; ${side};`;
      this._launcher.title = "Open chat";
      document.body.appendChild(this._launcher);

      // Popup
      this._popup = document.createElement("div");
      this._popup.id = "cw-popup";
      this._popup.classList.add("cw-hidden");
      this._popup.style.cssText = `bottom: 88px; ${side};`;
      this._popup.innerHTML = `
        <div id="cw-header">
          <div id="cw-header-left">
            <div id="cw-avatar">${ICON_BOT}</div>
            <span>${this._cfg.title}</span>
          </div>
          <button id="cw-close-btn" title="Close">&#x2715;</button>
        </div>
        <div id="cw-messages"></div>
        <div id="cw-input-area">
          <textarea id="cw-input" rows="1"
            placeholder="${this._cfg.placeholder}"></textarea>
          <button id="cw-send-btn" title="Send">${ICON_SEND}</button>
        </div>
        <div id="cw-powered">Powered by local AI · no data leaves your server</div>
      `;
      document.body.appendChild(this._popup);

      this._msgs    = this._popup.querySelector("#cw-messages");
      this._input   = this._popup.querySelector("#cw-input");
      this._sendBtn = this._popup.querySelector("#cw-send-btn");
    },

    _bindEvents() {
      this._launcher.addEventListener("click", () => this._toggle());
      this._popup.querySelector("#cw-close-btn").addEventListener("click", () => this._close());

      this._input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          this._send();
        }
      });
      // Auto-grow textarea
      this._input.addEventListener("input", () => {
        this._input.style.height = "auto";
        this._input.style.height = Math.min(this._input.scrollHeight, 120) + "px";
      });
      this._sendBtn.addEventListener("click", () => this._send());
    },

    /* ── Open / close ─────────────────────────────────────────────────── */
    _toggle() { this._open ? this._close() : this._openPopup(); },
    _openPopup() {
      this._open = true;
      this._popup.classList.remove("cw-hidden");
      this._launcher.innerHTML = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
      this._input.focus();
      this._scrollBottom();
    },
    _close() {
      this._open = false;
      this._popup.classList.add("cw-hidden");
      this._launcher.innerHTML = ICON_CHAT;
    },

    /* ── Message rendering ───────────────────────────────────────────── */
    _addUserMessage(text) {
      this._appendMessage({ role: "user", text });
    },
    _addBotMessage(text, latencyMs = null) {
      this._appendMessage({ role: "bot", text, latencyMs });
    },
    _addErrorMessage(text) {
      this._appendMessage({ role: "error", text });
    },
    _appendMessage({ role, text, latencyMs = null }) {
      const wrap = document.createElement("div");
      wrap.className = `cw-msg cw-${role}`;

      const bubble = document.createElement("div");
      bubble.className = "cw-bubble";
      bubble.textContent = text;
      wrap.appendChild(bubble);

      if (latencyMs !== null) {
        const meta = document.createElement("div");
        meta.className = "cw-meta";
        meta.textContent = `${(latencyMs / 1000).toFixed(1)}s`;
        wrap.appendChild(meta);
      }

      this._msgs.appendChild(wrap);
      this._scrollBottom();
    },

    _showTyping() {
      const wrap = document.createElement("div");
      wrap.className = "cw-msg cw-bot cw-typing";
      wrap.id = "cw-typing";
      wrap.innerHTML = `<div class="cw-bubble"><span class="cw-dot"></span><span class="cw-dot"></span><span class="cw-dot"></span></div>`;
      this._msgs.appendChild(wrap);
      this._scrollBottom();
    },
    _hideTyping() {
      const el = this._popup.querySelector("#cw-typing");
      if (el) el.remove();
    },

    _scrollBottom() {
      this._msgs.scrollTop = this._msgs.scrollHeight;
    },

    /* ── API call ────────────────────────────────────────────────────── */
    async _send() {
      const text = this._input.value.trim();
      if (!text || this._busy) return;

      this._busy = true;
      this._input.value = "";
      this._input.style.height = "auto";
      this._sendBtn.disabled = true;

      this._addUserMessage(text);
      this._showTyping();

      try {
        const res = await fetch(this._cfg.apiUrl, {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify({ question: text, top_k: this._cfg.top_k }),
        });

        this._hideTyping();

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          this._addErrorMessage(`Error ${res.status}: ${err.detail || "Unknown error"}`);
          return;
        }

        const data = await res.json();
        this._addBotMessage(data.answer, data.latency_ms);

      } catch (err) {
        this._hideTyping();
        this._addErrorMessage(
          "Could not reach the server. Is the backend running?\n" + err.message
        );
      } finally {
        this._busy = false;
        this._sendBtn.disabled = false;
        this._input.focus();
      }
    },
  };

  /* ── Expose globally ──────────────────────────────────────────────────────── */
  global.ChatWidget = ChatWidget;

})(window);
