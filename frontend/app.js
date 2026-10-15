"use strict";

/* NewsSort front end. No build step, no external scripts.
   All user text is inserted with textContent (never innerHTML). */

const TOPICS = ["World", "Sports", "Business", "Sci/Tech"];
const ORDER = ["bert", "logreg", "rf"];
const NAMES = { bert: "BERT", logreg: "Logistic Regression", rf: "Random Forest" };
const TOPIC_CLASS = { "World": "t-world", "Sports": "t-sports", "Business": "t-business", "Sci/Tech": "t-scitech" };
const HINT_TEXT = {
  close_call: "Close call: the top two topics are within 0.15. Please review this one.",
  business_scitech_overlap: "Business and Sci/Tech overlap. Please check this one.",
};
const PAGES = ["classify", "batch", "performance", "history", "about"];
const HISTORY_KEY = "newssort-history", THEME_KEY = "newssort-theme", MAX_HISTORY = 50;
const LOW_CONFIDENCE = 0.7;

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const state = { health: null, metrics: null, examples: [], exampleIdx: 0, batchRows: [], sort: { key: "row", dir: 1 }, file: null };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}
const clear = (node) => { while (node.firstChild) node.removeChild(node.firstChild); };
const pct = (x) => (x * 100).toFixed(1) + "%";
const truncate = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);
const topicPill = (topic) => (topic ? el("span", { class: "pill " + TOPIC_CLASS[topic], text: topic }) : el("span", { class: "muted", text: "-" }));

function formatError(detail) {
  if (Array.isArray(detail)) return detail.map((d) => (d.msg || "").replace(/^Value error, /, "")).join(" ");
  return typeof detail === "string" ? detail : "Something went wrong.";
}
async function api(path, options) {
  let res;
  try { res = await fetch(path, options); }
  catch { throw new Error("Cannot reach the server. Is NewsSort running?"); }
  let body = null;
  try { body = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new Error(formatError(body && body.detail));
  return body;
}
function showError(box, message) { box.textContent = message; box.hidden = !message; }

/* ------------------------------------------------------------ safe storage */
function loadHistory() { try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; } }
function saveHistory(items) { try { localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, MAX_HISTORY))); } catch { /* storage unavailable */ } }

/* ------------------------------------------------------------------ theme */
function applyTheme(theme) { document.documentElement.setAttribute("data-theme", theme); }
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem(THEME_KEY); } catch { /* ignore */ }
  applyTheme(saved || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem(THEME_KEY, next); } catch { /* ignore */ }
  });
}

/* ----------------------------------------------------------------- routing */
function showPage(name) {
  if (!PAGES.includes(name)) name = "classify";
  PAGES.forEach((p) => { $("#page-" + p).hidden = p !== name; });
  $$("#nav a").forEach((a) => { if (a.dataset.page === name) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
  if (name === "performance") loadPerformance();
  if (name === "history") renderHistory();
  window.scrollTo(0, 0);
}

/* ------------------------------------------------------------------ health */
async function loadHealth() {
  const chip = $("#status-chip");
  try {
    state.health = await api("/api/health");
  } catch (e) {
    chip.textContent = "Server offline"; chip.className = "chip bad"; return;
  }
  const models = state.health.models;
  const ready = ORDER.filter((k) => models[k].available);
  chip.textContent = ready.length ? `${ready.length} of 3 models ready` : "No models loaded";
  chip.className = "chip " + (ready.length ? "ok" : "bad");
  chip.title = ORDER.map((k) => `${NAMES[k]}: ${models[k].available ? "ready" : models[k].reason}`).join("\n");
  for (const sel of [$("#model"), $("#batch-model")]) {
    for (const opt of $$("option", sel)) {
      if (opt.value === "all") continue;
      opt.disabled = !models[opt.value].available;
      opt.textContent = NAMES[opt.value] + (opt.disabled ? " (not loaded)" : "");
    }
    if (ready.length && sel.selectedOptions[0].disabled) sel.value = ready[0];
  }
  $("#batch-limit").textContent = state.health.limits.max_batch_rows;
  updateCounter();
}

/* ---------------------------------------------------------------- classify */
function maxChars() { return (state.health && state.health.limits.max_chars) || 2000; }
function updateCounter() {
  const n = $("#title").value.length + $("#description").value.length;
  const c = $("#counter");
  c.textContent = `${n} / ${maxChars()} characters`;
  c.style.color = n > maxChars() ? "var(--err-text)" : "";
}
function hintBlock(hints) { return hints.map((h) => el("div", { class: "hint", text: HINT_TEXT[h] || h })); }
function resultCard(res, heading) {
  return el("article", { class: "result" },
    el("h3", { text: heading || res.model_name }),
    el("span", { class: "badge " + TOPIC_CLASS[res.topic], text: res.topic }),
    el("div", { class: "conf" }, el("span", { text: "Confidence " + pct(res.confidence) }),
      el("div", { class: "bar", role: "img", "aria-label": "Confidence " + pct(res.confidence) },
        el("span", { style: `width:${(res.confidence * 100).toFixed(1)}%` }))),
    el("div", { class: "probs" }, TOPICS.map((t) =>
      el("div", { class: "prob" }, el("span", { text: t }),
        el("div", { class: "bar" }, el("span", { style: `width:${(res.probabilities[t] * 100).toFixed(1)}%` })),
        el("b", { text: pct(res.probabilities[t]) })))),
    hintBlock(res.hints));
}
function renderResults(data, mode) {
  const box = $("#results"); clear(box);
  const present = ORDER.filter((k) => data.results[k]);
  if (mode === "all") {
    if (data.agreement === false) box.append(el("div", { class: "alert warn", text: "Models disagree: review this article manually." }));
    else if (data.agreement === true) box.append(el("div", { class: "alert ok", text: "All available models agree." }));
    const missing = Object.entries(data.unavailable);
    if (missing.length) box.append(el("div", { class: "alert warn", text: "Not available: " + missing.map(([k, v]) => `${NAMES[k]} (${v})`).join("; ") }));
    box.append(el("div", { class: "cards" }, present.map((k) => resultCard(data.results[k]))));
  } else {
    box.append(resultCard(data.primary, data.primary.model_name + " prediction"));
  }
}
function addHistory(data, mode, title, description) {
  const present = ORDER.filter((k) => data.results[k]);
  const items = loadHistory();
  items.unshift({
    id: Date.now() + Math.random().toString(16).slice(2, 6), ts: new Date().toISOString(), title, description,
    models: present.map((k) => ({ model: k, topic: data.results[k].topic, confidence: data.results[k].confidence })),
    mode, correction: "",
  });
  saveHistory(items);
}
async function classify() {
  const title = $("#title").value.trim(), description = $("#description").value.trim();
  const errBox = $("#error"); showError(errBox, "");
  if (!title && !description) { showError(errBox, "Please enter a title or a description."); return; }
  const model = $("#model").value, btn = $("#btn-classify");
  btn.disabled = true; btn.replaceChildren(el("span", { class: "spin" }), "Classifying...");
  try {
    const data = await api("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title, description, model }) });
    renderResults(data, model); addHistory(data, model, title, description);
  } catch (e) { clear($("#results")); showError(errBox, e.message); }
  finally { btn.disabled = false; btn.textContent = "Classify"; }
}
function nextExample() {
  if (!state.examples.length) return;
  const ex = state.examples[state.exampleIdx++ % state.examples.length];
  $("#title").value = ex.title; $("#description").value = ex.description; $("#example-note").textContent = ex.note || "";
  updateCounter();
}
function initClassify() {
  $("#btn-classify").addEventListener("click", classify);
  $("#btn-example").addEventListener("click", nextExample);
  $("#btn-clear").addEventListener("click", () => {
    $("#title").value = ""; $("#description").value = ""; $("#example-note").textContent = "";
    clear($("#results")); showError($("#error"), ""); updateCounter();
  });
  for (const id of ["#title", "#description"]) $(id).addEventListener("input", updateCounter);
  $("#description").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) classify(); });
  api("/api/examples").then((ex) => { state.examples = ex; }).catch(() => {});
}

/* ------------------------------------------------------------------- batch */
function csvCell(v) {
  v = v === null || v === undefined ? "" : String(v);
  if (/^[=+\-@\t\r]/.test(v)) v = "'" + v;                 // block spreadsheet formula injection
  return '"' + v.replace(/"/g, '""') + '"';
}
function setFile(file) {
  state.file = file || null;
  $("#file-name").textContent = file ? `${file.name} (${(file.size / 1024).toFixed(1)} KB)` : "No file selected";
  $("#btn-batch").disabled = !file;
}
function batchView() {
  const topic = $("#batch-filter").value, lowOnly = $("#batch-low").checked, { key, dir } = state.sort;
  const rows = state.batchRows.filter((r) => (!topic || r.topic === topic) && (!lowOnly || r.review || (r.confidence !== null && r.confidence < LOW_CONFIDENCE)));
  const val = (r) => (key === "title" ? (r.title || r.description).toLowerCase() : r[key]);
  rows.sort((a, b) => {
    const x = val(a), y = val(b);
    if (x === y) return a.row - b.row;
    if (x === null || x === "") return 1;
    if (y === null || y === "") return -1;
    return (x > y ? 1 : -1) * dir;
  });
  return rows;
}
function renderBatch() {
  const tbody = $("#batch-table tbody"); clear(tbody);
  const rows = batchView();
  for (const r of rows) {
    tbody.append(el("tr", {},
      el("td", { text: r.row }),
      el("td", { text: truncate(r.title || r.description || "(empty)", 110), title: r.error || "" }),
      el("td", {}, r.error ? el("span", { class: "muted", text: r.error }) : topicPill(r.topic)),
      el("td", { text: r.confidence === null ? "-" : pct(r.confidence) }),
      el("td", { text: r.review ? "Check" : "" })));
  }
  const total = state.batchRows.length, flagged = state.batchRows.filter((r) => r.review).length;
  $("#batch-summary").textContent = `${rows.length} of ${total} rows shown. ${flagged} flagged for review.`;
}
async function runBatch() {
  if (!state.file) return;
  const err = $("#batch-error"); showError(err, ""); $("#batch-out").hidden = true;
  if (state.file.size > 5 * 1024 * 1024) { showError(err, "File is larger than 5 MB."); return; }
  const btn = $("#btn-batch"), progress = $("#progress");
  btn.disabled = true; progress.hidden = false;
  const form = new FormData(); form.append("file", state.file);
  try {
    const data = await api(`/api/predict-batch?model=${encodeURIComponent($("#batch-model").value)}&format=json`, { method: "POST", body: form });
    state.batchRows = data.rows; state.batchModel = data.model;
    $("#batch-out").hidden = false; renderBatch();
  } catch (e) { showError(err, e.message); }
  finally { btn.disabled = false; progress.hidden = true; }
}
function downloadBatch() {
  const header = ["row", "title", "description", "predicted_topic", "confidence", "review_flag", "model", "error"];
  const lines = [header.map(csvCell).join(",")];
  for (const r of state.batchRows) {
    lines.push([r.row, r.title, r.description, r.topic, r.confidence === null ? "" : r.confidence, r.review ? "yes" : "", state.batchModel, r.error].map(csvCell).join(","));
  }
  download("newssort_predictions.csv", lines.join("\r\n"));
}
function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const a = el("a", { href: url, download: name }); document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function initBatch() {
  const file = $("#file"), drop = $("#drop");
  file.addEventListener("change", () => setFile(file.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) setFile(f); });
  $("#btn-batch").addEventListener("click", runBatch);
  $("#btn-download").addEventListener("click", downloadBatch);
  $("#batch-filter").addEventListener("change", renderBatch);
  $("#batch-low").addEventListener("change", renderBatch);
  const sortBy = (th) => {
    const key = th.dataset.key;
    state.sort = { key, dir: state.sort.key === key ? -state.sort.dir : 1 }; renderBatch();
  };
  $$("#batch-table th[data-key]").forEach((th) => {
    th.addEventListener("click", () => sortBy(th));
    th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sortBy(th); } });
  });
}

/* ------------------------------------------------------------- performance */
function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function fmtTime(s) { return s >= 60 ? `${Math.round(s)} s (${(s / 60).toFixed(1)} min)` : s < 1 ? `${s.toFixed(2)} s` : `${s.toFixed(1)} s`; }
function drawChart(m) {
  const keys = ORDER.filter((k) => m.models[k]);
  const W = 640, H = 320, L = 46, R = 10, T = 18, B = 44, min = m.chart_axis_min, max = 96;
  const y = (v) => T + (H - T - B) * (1 - (v - min) / (max - min));
  const gw = (W - L - R) / keys.length, bw = gw * 0.26;
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Accuracy and macro-F1 for each model, axis from ${min} percent">`;
  for (let v = min; v <= max; v += 2) svg += `<line class="axis" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  keys.forEach((k, i) => {
    const x0 = L + i * gw + gw / 2;
    [["acc", m.models[k].accuracy, -bw - 2], ["f1", m.models[k].f1_macro, 2]].forEach(([cls, v, dx]) => {
      svg += `<rect class="${cls}" x="${x0 + dx}" y="${y(v)}" width="${bw}" height="${y(min) - y(v)}" rx="3"><title>${esc(m.models[k].name)}: ${v}%</title></rect>`;
      svg += `<text x="${x0 + dx + bw / 2}" y="${y(v) - 5}" text-anchor="middle">${v.toFixed(2)}</text>`;
    });
    svg += `<text x="${x0}" y="${H - 18}" text-anchor="middle" font-weight="600">${esc(m.models[k].name)}</text>`;
  });
  svg += `<rect class="acc" x="${W - 190}" y="4" width="10" height="10"/><text x="${W - 175}" y="13">Accuracy</text>`;
  svg += `<rect class="f1" x="${W - 105}" y="4" width="10" height="10"/><text x="${W - 90}" y="13">Macro-F1</text></svg>`;
  $("#chart").innerHTML = svg;                       // values and names come from our own static JSON
}
function drawHeatmap(m, key) {
  const cm = m.models[key].confusion, labels = m.labels, box = $("#heatmap"); clear(box);
  const table = el("table", { class: "heat", "aria-label": `Confusion matrix for ${m.models[key].name}` });
  table.append(el("thead", {}, el("tr", {}, el("th", { text: "Actual \\ Predicted" }), labels.map((l) => el("th", { text: l })))));
  const tbody = el("tbody");
  cm.forEach((row, i) => {
    tbody.append(el("tr", {}, el("th", { text: labels[i] }), row.map((v, j) => {
      const a = Math.max(0.04, v / 1900), hot = (i === 2 && j === 3) || (i === 3 && j === 2);
      return el("td", { class: hot ? "hot" : "", text: v.toLocaleString(), style: `background:rgba(var(--heat),${a.toFixed(2)});${a > 0.5 ? "color:#fff" : ""}` });
    })));
  });
  table.append(tbody); box.append(table);
}
async function loadPerformance() {
  if (state.metrics) return;
  try { state.metrics = await api("/api/metrics"); } catch (e) { showError($("#perf-error"), e.message); return; }
  const m = state.metrics;
  $("#perf-dataset").textContent = m.dataset;
  const tbody = $("#perf-table tbody"); clear(tbody);
  for (const k of ORDER) {
    const r = m.models[k];
    tbody.append(el("tr", {}, el("td", { text: r.name }), el("td", { text: r.accuracy.toFixed(2) + "%" }), el("td", { text: (r.f1_macro / 100).toFixed(4) }),
      el("td", { text: r.errors }), el("td", { text: fmtTime(r.train_seconds) }), el("td", { text: fmtTime(r.test_seconds) })));
  }
  drawChart(m); drawHeatmap(m, "bert");
  $("#cm-model").addEventListener("change", (e) => drawHeatmap(m, e.target.value));
  $("#perf-note").textContent = m.note + ` Articles wrong for all three models: ${m.shared_errors.all_three_wrong} (${m.shared_errors.share_of_test_set_pct}% of the test set).`;
}

/* ----------------------------------------------------------------- history */
function renderHistory() {
  const items = loadHistory(), table = $("#history-table"), tbody = $("tbody", table); clear(tbody);
  $("#history-empty").hidden = items.length > 0; table.hidden = items.length === 0;
  for (const it of items) {
    const when = new Date(it.ts).toLocaleString([], { dateStyle: "short", timeStyle: "short" });
    const first = it.models[0];
    const correctionCell = el("td");
    const show = () => {
      clear(correctionCell);
      if (it.correction) {
        correctionCell.append(topicPill(it.correction), " ",
          el("button", { class: "btn ghost", type: "button", text: "Change", onclick: edit }));
      } else correctionCell.append(el("button", { class: "btn", type: "button", text: "Wrong topic?", onclick: edit }));
    };
    const edit = () => {
      clear(correctionCell);
      const sel = el("select", { "aria-label": "Correct topic" }, TOPICS.map((t) => el("option", { value: t, text: t })));
      if (it.correction) sel.value = it.correction;
      correctionCell.append(sel, " ", el("button", { class: "btn primary", type: "button", text: "Save", onclick: () => {
        const all = loadHistory(), target = all.find((x) => x.id === it.id);
        if (target) { target.correction = sel.value; it.correction = sel.value; saveHistory(all); }
        show();
      } }), " ", el("button", { class: "btn ghost", type: "button", text: "Cancel", onclick: show }));
    };
    show();
    tbody.append(el("tr", {},
      el("td", { text: when }),
      el("td", { text: truncate(it.title || it.description, 90) }),
      el("td", { text: it.mode === "all" ? "All models" : NAMES[first.model] }),
      el("td", {}, it.models.map((x) => el("div", {}, topicPill(x.topic), it.mode === "all" ? ` ${NAMES[x.model]}` : ""))),
      el("td", { text: it.models.map((x) => pct(x.confidence)).join(" / ") }),
      correctionCell));
  }
}
function initHistory() {
  $("#btn-clear-history").addEventListener("click", () => {
    if (confirm("Delete all saved predictions from this browser?")) { saveHistory([]); renderHistory(); }
  });
  $("#btn-export").addEventListener("click", () => {
    const rows = loadHistory().filter((x) => x.correction);
    if (!rows.length) { alert("No corrections recorded yet."); return; }
    const lines = [["title", "description", "model", "predicted_topic", "correct_topic"].map(csvCell).join(",")];
    for (const r of rows) lines.push([r.title, r.description, r.models.map((m) => NAMES[m.model]).join(" + "), r.models[0].topic, r.correction].map(csvCell).join(","));
    download("newssort_corrections.csv", lines.join("\r\n"));
  });
}

/* -------------------------------------------------------------------- init */
document.addEventListener("DOMContentLoaded", () => {
  initTheme(); initClassify(); initBatch(); initHistory();
  window.addEventListener("hashchange", () => showPage(location.hash.slice(1)));
  showPage(location.hash.slice(1) || "classify");
  loadHealth();
});
