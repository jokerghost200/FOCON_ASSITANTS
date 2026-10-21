(function () {
  const $ = (id) => document.getElementById(id);
  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    return r.json().catch(() => ({}));
  };
  const esc = (s) => String(s ?? "").replace(/[&<>]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  async function refreshPositions() {
    const p = await api("/api/positions");
    const tb = $("posBody");
    tb.innerHTML = "";
    if (!p.ok || !p.positions || !p.positions.length) {
      tb.innerHTML = '<tr><td colspan="9">Aucune position.</td></tr>';
      return;
    }
    for (const x of p.positions) {
      const tr = document.createElement("tr");
      tr.innerHTML = ["ticket", "symbol", "type", "volume", "price_open", "sl", "tp", "price_current", "profit"]
        .map((k) => `<td>${esc(x[k])}</td>`).join("");
      tb.appendChild(tr);
    }
  }

  function renderSignals(results) {
    const box = $("signals");
    box.innerHTML = "";
    (results || []).forEach((res) => {
      const div = document.createElement("div");
      div.className = "signal";
      if (res.error) {
        div.textContent = "Erreur: " + res.error;
      } else if (res.status === "WAIT") {
        div.textContent = `WAIT (${res.symbol}) — pas d'opportunite claire.`;
      } else if (res.status === "CLOSED") {
        div.textContent = "Marches fermes ignores : " + (res.symbols || []).join(", ");
      } else if (res.status === "PENDING_OB") {
        const d = res.decision || {};
        const lbl = res.pending_label ? ` « ${res.pending_label} »` : "";
        div.textContent = `📌 PENDING${lbl} ${d.action || ""} sur ${res.symbol}\nConfiance : ${d.confidence}% | Entree limite ${d.entry} | SL ${d.sl} | TP ${d.tp} | ${d.lot} lots\nNom modifiable (31 car. max) : `;
        const nameInput = document.createElement("input");
        nameInput.value = res.pending_label || `OB ${res.symbol} ${d.action || ""}`;
        nameInput.maxLength = 31;
        nameInput.placeholder = "Nom du Pending Block...";
        nameInput.style.width = "220px";
        nameInput.onchange = () => { res.pending_label = nameInput.value.slice(0, 31); };
        div.append(nameInput);
        const row = document.createElement("div");
        row.className = "actions";
        const take = document.createElement("button");
        take.className = "success";
        take.textContent = "📌 PLACER ORDRE LIMITE";
        take.onclick = async () => {
          take.disabled = true;
          res.pending_label = nameInput.value.slice(0, 31);
          const r = await api("/api/trade/take", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ signal: res }),
          });
          div.append(document.createTextNode("\n-> " + (r.ok ? `Ordre pending place` : `Echec : ${r.reason}`)));
          refreshPositions();
        };
        const ignP = document.createElement("button");
        ignP.className = "danger";
        ignP.textContent = "IGNORER";
        ignP.onclick = () => div.remove();
        row.append(take, ignP);
        div.append(row);
      } else {
        const d = res.decision || {};
        div.textContent = `${d.action} sur ${res.symbol}\nConfiance : ${d.confidence}% | Entree ${d.entry} | SL ${d.sl} | TP ${d.tp} | ${d.lot} lots\n${d.reason || ""}`;
        const row = document.createElement("div");
        row.className = "actions";
        const take = document.createElement("button");
        take.className = "success";
        take.textContent = "PRENDRE POSITION";
        take.onclick = async () => {
          take.disabled = true;
          const r = await api("/api/trade/take", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ signal: res }),
          });
          div.append(document.createTextNode("\n-> " + (r.ok ? `Ordre execute (deal ${r.deal_id})` : `Echec : ${r.reason}`)));
          refreshPositions();
        };
        const ign = document.createElement("button");
        ign.className = "danger";
        ign.textContent = "IGNORER";
        ign.onclick = () => div.remove();
        row.append(take, ign);
        div.append(row);
      }
      box.append(div);
    });
  }

  let scanTimer = null;
  async function pollScan() {
    const s = await api("/api/scan/status");
    if (!s.ok) return;
    const sc = s.scan || {};
    $("scanStatus").textContent = sc.message || (sc.running ? "Scan en cours…" : "En attente…");
    $("btnScan").disabled = !!sc.running;
    if (!sc.running && sc.results && sc.results.length) {
      renderSignals(sc.results);
    }
    if (!sc.running && scanTimer) { clearInterval(scanTimer); scanTimer = null; }
  }

  $("btnScan").onclick = async () => {
    const tf = $("cfgTf").value || "M5";
    $("scanStatus").textContent = "Demarrage du scan…";
    await api("/api/scan", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ timeframe: tf }),
    });
    if (scanTimer) clearInterval(scanTimer);
    scanTimer = setInterval(pollScan, 2000);
    pollScan();
  };

  $("btnStop").onclick = async () => {
    await api("/api/scan/stop", { method: "POST" });
    pollScan();
  };

  async function refreshAuto() {
    const a = await api("/api/autopilot/status");
    const on = !!(a.ok && a.autopilot && a.autopilot.enabled);
    $("autoTxt").textContent = "Autopilot : " + (on ? "ON" : "OFF");
    $("btnAuto").textContent = "AUTOPILOT : " + (on ? "ON" : "OFF");
    if (a.ok && a.autopilot && a.autopilot.message) {
      $("scanStatus").textContent = a.autopilot.message;
    }
  }

  $("btnAuto").onclick = async () => {
    const isOn = $("autoTxt").textContent.includes("ON");
    await api(isOn ? "/api/autopilot/stop" : "/api/autopilot/start", { method: "POST" });
    refreshAuto();
  };

  $("btnPos").onclick = refreshPositions;

  async function refreshHealth() {
    const h = await api("/api/health");
    const ok = !!(h.ok && h.mt5_connected);
    $("healthDot").style.color = ok ? "#4CAF50" : "#F44336";
    $("healthTxt").textContent = "MT5 : " + (ok ? "connecté" : "déconnecté");
  }

  async function refreshAccount() {
    const a = await api("/api/account");
    if (!a.ok) { $("accountBox").textContent = "Compte indisponible (MT5 déconnecté)."; return; }
    const c = a.account || {};
    $("accountBox").textContent =
      `Login ${c.login ?? "—"} | Balance ${c.balance ?? "—"} | Equity ${c.equity ?? "—"} | Marge libre ${c.margin_free ?? "—"}`;
  }

  async function loadConfig() {
    const r = await api("/api/config");
    if (!r.ok) return;
    const c = r.config || {}, rm = c.risk_management || {}, tr = c.trading || {};
    $("cfgRisk").value = rm.default_risk_percent ?? 1;
    $("cfgLoss").value = rm.max_daily_loss_percent ?? 5;
    $("cfgLot").value = rm.max_lot_size ?? 1;
    $("cfgPos").value = rm.max_open_positions ?? 3;
    $("cfgMode").value = (tr.automation_mode || "assist").toLowerCase();
    const allowed = tr.allowed_market_types || [];
    document.querySelectorAll(".mtype").forEach((cb) => { cb.checked = allowed.includes(cb.value); });
    if (tr.timeframes && tr.timeframes[0]) $("cfgTf").value = tr.timeframes[0];
  }

  $("btnCfg").onclick = async () => {
    const allowed = [...document.querySelectorAll(".mtype")].filter((c) => c.checked).map((c) => c.value);
    const body = {
      risk_management: {
        default_risk_percent: parseFloat($("cfgRisk").value),
        max_daily_loss_percent: parseFloat($("cfgLoss").value),
        max_lot_size: parseFloat($("cfgLot").value),
        max_open_positions: parseInt($("cfgPos").value, 10),
      },
      trading: {
        automation_mode: $("cfgMode").value,
        allowed_market_types: allowed,
        timeframes: [$("cfgTf").value],
      },
    };
    const r = await api("/api/config", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("scanStatus").textContent = r.ok ? "Configuration sauvegardée." : ("Echec config : " + (r.reason || "?"));
  };

  function addChat(role, text) {
    const d = document.createElement("div");
    d.innerHTML = `<b>${esc(role)} :</b> ${esc(text)}`;
    $("chatBox").append(d);
    $("chatBox").scrollTop = $("chatBox").scrollHeight;
  }

  async function sendChat() {
    const msg = $("chatInput").value.trim();
    if (!msg) return;
    $("chatInput").value = "";
    addChat("Toi", msg);
    const r = await api("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg }),
    });
    addChat("IA", r.ok ? r.reply : ("Erreur : " + (r.reason || "?")));
  }

  function speak(text) {
    if (!$("voiceReplies").checked || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(String(text).slice(0, 1800));
    utterance.lang = "fr-FR";
    utterance.rate = 1;
    window.speechSynthesis.speak(utterance);
  }

  function formatValue(value) {
    return Array.isArray(value) ? value.join(", ") : String(value ?? "-");
  }

  function renderProposal(result) {
    const box = $("proposalBox");
    box.innerHTML = "";
    const changes = result.changes || [];
    if (!result.proposal_id || !changes.length) {
      box.hidden = true;
      return;
    }
    const title = document.createElement("strong");
    title.textContent = "Proposition a valider";
    box.append(title);
    const list = document.createElement("ul");
    changes.forEach((change) => {
      const item = document.createElement("li");
      item.textContent = `${change.label} : ${formatValue(change.current)} -> ${formatValue(change.value)}. ${change.reason || ""}`;
      list.append(item);
    });
    const apply = document.createElement("button");
    apply.className = "success";
    apply.textContent = "APPLIQUER";
    apply.onclick = async () => {
      if (!window.confirm("Appliquer ces reglages au bot ? Cette action sera enregistree dans les journaux.")) return;
      apply.disabled = true;
      const response = await api("/api/advisor/apply", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ proposal_id: result.proposal_id }),
      });
      if (response.ok) {
        $("voiceStatus").textContent = "Reglages appliques apres votre confirmation.";
        addChat("Conseiller", "Les reglages proposes ont ete appliques.");
        speak("Les reglages proposes ont ete appliques.");
        box.hidden = true;
        loadConfig();
      } else {
        $("voiceStatus").textContent = "Application refusee : " + (response.reason || "erreur inconnue");
        apply.disabled = false;
      }
    };
    const dismiss = document.createElement("button");
    dismiss.className = "danger";
    dismiss.textContent = "REFUSER";
    dismiss.onclick = () => {
      box.hidden = true;
      $("voiceStatus").textContent = "Proposition refusee. Aucun reglage n'a ete modifie.";
    };
    box.append(list, apply, dismiss);
    box.hidden = false;
  }

  async function askAdvisor(message) {
    const msg = String(message || $("chatInput").value || "").trim();
    if (!msg) {
      $("voiceStatus").textContent = "Formulez votre demande avant de demander un conseil.";
      return;
    }
    $("chatInput").value = "";
    addChat("Toi", msg);
    $("voiceStatus").textContent = "Le conseiller prepare une proposition...";
    const result = await api("/api/advisor", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg }),
    });
    const reply = result.ok ? result.reply : ("Erreur : " + (result.reason || "?"));
    addChat("Conseiller", reply);
    $("voiceStatus").textContent = result.ok
      ? (result.requires_confirmation ? "Proposition disponible : validez-la ou refusez-la." : "Aucun reglage n'a ete modifie.")
      : "Le conseiller n'a pas pu traiter la demande.";
    speak(reply);
    if (result.ok) renderProposal(result);
  }

  function setupVoice() {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const voiceButton = $("btnVoice");
    if (!Recognition) {
      voiceButton.disabled = true;
      $("voiceStatus").textContent = "La reconnaissance vocale n'est pas prise en charge par ce navigateur.";
      return;
    }
    const recognition = new Recognition();
    recognition.lang = "fr-FR";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onstart = () => {
      voiceButton.disabled = true;
      $("voiceStatus").textContent = "Ecoute en cours...";
    };
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript.trim();
      $("voiceStatus").textContent = "Demande recue : " + transcript;
      askAdvisor(transcript);
    };
    recognition.onerror = (event) => {
      $("voiceStatus").textContent = "Erreur micro : " + event.error;
    };
    recognition.onend = () => { voiceButton.disabled = false; };
    voiceButton.onclick = () => {
      try { recognition.start(); } catch (_) { /* already listening */ }
    };
  }

  $("btnChat").onclick = sendChat;
  $("btnAdvisor").onclick = () => askAdvisor();
  $("chatInput").addEventListener("keydown", (e) => { if (e.key === "Enter") sendChat(); });

  setupVoice();
  loadConfig(); refreshHealth(); refreshAccount(); refreshPositions(); refreshAuto();
  setInterval(refreshPositions, 10000);
  setInterval(refreshHealth, 15000);
  setInterval(refreshAuto, 5000);
})();
