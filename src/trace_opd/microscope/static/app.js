/* TRACE-OPD 蒸馏显微镜工作台(零依赖 vanilla JS)。
 * 路由:#/samples?step= | #/traj?step=&hash=&track=&pos= | #/explore | #/evolution?task=
 * 一切数值由后端从观察包实时派生;本文件只做渲染与交互。 */
"use strict";
const $ = (sel) => document.querySelector(sel);
const api = async (path) => {
  const r = await fetch(path);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.status);
  return j;
};
let META = null;
let TRAJ = null;           // current /api/sample payload
let TRAJ_KEY = "";         // `${step}:${hash}`
let SEL = -1;              // selected position

/* ---------- viridis-ish perceptual colormap ---------- */
const VIRIDIS = [[68,1,84],[71,44,122],[59,81,139],[44,113,142],[33,144,141],
  [39,173,129],[92,200,99],[170,220,50],[253,231,37]];
function cmap(t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (VIRIDIS.length - 1), i = Math.floor(x), f = x - i;
  const a = VIRIDIS[i], b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function quantile(vals, q) {
  const s = vals.filter((v) => v != null && isFinite(v)).slice().sort((a, b) => a - b);
  if (!s.length) return 1;
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}

/* ---------- routing ---------- */
function route() {
  const h = location.hash || "#/samples";
  const [path, qs] = h.slice(2).split("?");
  const q = Object.fromEntries(new URLSearchParams(qs || ""));
  document.querySelectorAll("nav a").forEach((a) =>
    a.classList.toggle("active", a.dataset.view === path));
  document.querySelectorAll(".view").forEach((v) =>
    v.classList.toggle("active", v.id === `view-${path}`));
  if (path === "samples") viewSamples(q);
  else if (path === "traj") viewTraj(q);
  else if (path === "explore") viewExplore(q);
  else if (path === "evolution") viewEvolution(q);
}
function go(path, q) {
  location.hash = `#/${path}?${new URLSearchParams(q)}`;
}
window.addEventListener("hashchange", route);

/* ---------- samples ---------- */
const S_COLS = [
  ["task_id", "task"], ["sample_role", "role"], ["won", "won"],
  ["reward", "reward"], ["resp_len", "len"], ["turns", "turns"],
  ["kl_mean", "KL μ"], ["kl_max", "KL max"], ["gap_mean", "gap μ"],
  ["overlap_mean", "overlap μ"], ["teacher_entropy_mean", "T-ent μ"],
];
let S_SORT = ["kl_mean", -1];
async function viewSamples(q) {
  const sel = $("#s-step");
  if (!sel.options.length) {
    META.steps.forEach((s) => sel.add(new Option(`step ${s}`, s)));
    sel.onchange = () => go("samples", { step: sel.value });
    $("#s-role").onchange = $("#s-won").onchange = renderSamples;
  }
  const step = q.step != null ? +q.step : +sel.value || META.steps[META.steps.length - 1];
  sel.value = step;
  window._rows = await api(`/api/samples?step=${step}`);
  renderSamples();
}
function renderSamples() {
  const role = $("#s-role").value, won = $("#s-won").value;
  let rows = (window._rows || []).filter((r) =>
    (!role || r.sample_role === role) && (won === "" || String(r.won) === won));
  rows.sort((a, b) => (a[S_SORT[0]] > b[S_SORT[0]] ? 1 : -1) * S_SORT[1]);
  $("#s-count").textContent = `${rows.length} 条`;
  $("#s-table thead").innerHTML =
    "<tr>" + S_COLS.map(([k, label]) =>
      `<th data-k="${k}">${label}${S_SORT[0] === k ? (S_SORT[1] < 0 ? " ↓" : " ↑") : ""}</th>`
    ).join("") + "</tr>";
  document.querySelectorAll("#s-table th").forEach((th) => th.onclick = () => {
    S_SORT = [th.dataset.k, S_SORT[0] === th.dataset.k ? -S_SORT[1] : -1];
    renderSamples();
  });
  $("#s-table tbody").innerHTML = rows.map((r) =>
    `<tr data-h="${r.resp_hash}" data-s="${r.step}">` + S_COLS.map(([k]) => {
      let v = r[k];
      if (typeof v === "number" && !Number.isInteger(v)) v = v.toFixed(3);
      if (k === "won") return `<td class="won-${r.won}">${r.won ? "✓" : "✗"}</td>`;
      if (k === "task_id") v = String(v).slice(0, 46);
      return `<td>${v}</td>`;
    }).join("") + "</tr>").join("");
  document.querySelectorAll("#s-table tbody tr").forEach((tr) => tr.onclick = () =>
    go("traj", { step: tr.dataset.s, hash: tr.dataset.h }));
}

/* ---------- trajectory microscope ---------- */
const TRACKS = ["kl_forward", "kl_reverse", "gap", "teacher_entropy",
  "student_entropy", "overlap_count", "token_loss"];
const MINIMAP_TRACKS = ["kl_forward", "gap", "teacher_entropy", "student_entropy", "overlap_count"];
async function viewTraj(q) {
  const trackSel = $("#t-track");
  if (!trackSel.options.length) {
    TRACKS.forEach((t) => trackSel.add(new Option(t, t)));
    trackSel.onchange = () => { paintTokens(); paintMinimap(); };
  }
  if (q.track) trackSel.value = q.track;
  if (!q.step || !q.hash) { $("#t-title").textContent = "从样本页选择一条轨迹"; return; }
  const key = `${q.step}:${q.hash}`;
  if (TRAJ_KEY !== key) {
    $("#t-title").textContent = "载入中…";
    TRAJ = await api(`/api/sample?step=${q.step}&hash=${q.hash}`);
    TRAJ_KEY = key;
    SEL = -1;
    buildTokens();
  }
  const s = TRAJ.summary;
  $("#t-title").innerHTML =
    `<b>${s.task_id.slice(0, 40)}</b> · step ${s.step} · ` +
    `<span class="won-${s.won}">${s.won ? "WON" : "LOST"}</span> · ` +
    `${s.resp_len} tok · ${s.turns} turns · ${TRAJ.loss_mode}`;
  const a = TRAJ.audit;
  $("#t-audit").textContent =
    `audit ${a.ok ? "✓" : "✗"} (${a.checked} tok, max_err ${(+a.max_err).toFixed(3)})`;
  paintTokens(); paintMinimap();
  if (q.pos != null && +q.pos !== SEL) selectPos(+q.pos, false);
}
function buildTokens() {
  const host = $("#t-tokens");
  host.innerHTML = "";
  const spans = TRAJ.turn_spans, starts = new Map(spans.map(([a], i) => [a, i]));
  TRAJ.tokens.forEach((t, i) => {
    if (starts.has(i)) {
      const sep = document.createElement("span");
      sep.className = "turn-sep"; sep.textContent = `── turn ${starts.get(i) + 1} ──`;
      host.appendChild(sep);
    }
    const el = document.createElement("span");
    el.className = "tk"; el.dataset.i = i;
    el.textContent = t.text.replace(/\n/g, "⏎\n");
    if (!TRAJ.tracks.mask[i]) el.classList.add("obs");
    if (!TRAJ.tracks.covered[i]) el.classList.add("uncov");
    el.onclick = () => selectPos(i, true);
    host.appendChild(el);
  });
  host.onkeydown = (ev) => {
    if (SEL < 0) return;
    const kl = TRAJ.tracks.kl_forward;
    if (ev.key === "ArrowRight") selectPos(Math.min(SEL + 1, kl.length - 1), true);
    else if (ev.key === "ArrowLeft") selectPos(Math.max(SEL - 1, 0), true);
    else if (ev.key === "n" || ev.key === "p") {
      const thr = quantile(kl, 0.95), dir = ev.key === "n" ? 1 : -1;
      for (let j = SEL + dir; j >= 0 && j < kl.length; j += dir)
        if (kl[j] != null && kl[j] >= thr && TRAJ.tracks.mask[j]) { selectPos(j, true); break; }
    } else return;
    ev.preventDefault();
  };
}
function paintTokens() {
  const track = $("#t-track").value, vals = TRAJ.tracks[track] || [];
  const hi = quantile(vals, 0.98) || 1, lo = Math.min(0, quantile(vals, 0.02));
  document.querySelectorAll("#t-tokens .tk").forEach((el) => {
    const v = vals[+el.dataset.i];
    el.style.background = v == null ? "transparent"
      : cmap((v - lo) / (hi - lo || 1)) + "55";
  });
}
function paintMinimap() {
  const cv = $("#t-minimap"), ctx = cv.getContext("2d");
  const n = TRAJ.tokens.length;
  cv.width = cv.clientWidth * devicePixelRatio;
  const W = cv.width, rowH = cv.height / MINIMAP_TRACKS.length;
  ctx.clearRect(0, 0, W, cv.height);
  MINIMAP_TRACKS.forEach((track, r) => {
    const vals = TRAJ.tracks[track];
    const hi = quantile(vals, 0.98) || 1, lo = Math.min(0, quantile(vals, 0.02));
    for (let x = 0; x < W; x++) {
      const i = Math.floor((x / W) * n), v = vals[i];
      ctx.fillStyle = v == null ? "#222" : cmap((v - lo) / (hi - lo || 1));
      ctx.fillRect(x, r * rowH, 1, rowH - 2);
    }
  });
  ctx.fillStyle = "#ffffffcc";
  ctx.font = `${10 * devicePixelRatio}px monospace`;
  MINIMAP_TRACKS.forEach((t, r) => ctx.fillText(t, 4, r * rowH + 10 * devicePixelRatio));
  if (SEL >= 0) {
    ctx.fillStyle = "#fff";
    ctx.fillRect((SEL / n) * W, 0, 2, cv.height);
  }
  cv.onclick = (ev) => {
    const i = Math.floor((ev.offsetX / cv.clientWidth) * n);
    selectPos(i, true);
    const el = document.querySelector(`#t-tokens .tk[data-i="${i}"]`);
    if (el) el.scrollIntoView({ block: "center" });
  };
}
async function selectPos(i, push) {
  SEL = i;
  document.querySelectorAll("#t-tokens .tk.sel").forEach((e) => e.classList.remove("sel"));
  const el = document.querySelector(`#t-tokens .tk[data-i="${i}"]`);
  if (el) el.classList.add("sel");
  $("#t-tokens").focus({ preventScroll: true });
  paintMinimap();
  const [step, hash] = TRAJ_KEY.split(":");
  if (push) history.replaceState(null, "",
    `#/traj?${new URLSearchParams({ step, hash, track: $("#t-track").value, pos: i })}`);
  const p = await api(`/api/position?step=${step}&hash=${hash}&pos=${i}`);
  renderInspector(p);
}
function renderInspector(p) {
  const host = $("#t-inspector");
  const fmt = (v, d = 3) => (v == null ? "—" : (+v).toFixed(d));
  const union = new Map();
  p.teacher.forEach((e, r) => union.set(e.id, { t: e, tr: r }));
  p.student.forEach((e, r) => {
    const u = union.get(e.id) || {};
    u.s = e; u.sr = r; union.set(e.id, u);
  });
  const rows = [...union.values()].sort((a, b) =>
    (b.t ? b.t.p : 0) - (a.t ? a.t.p : 0) || (b.s ? b.s.p : 0) - (a.s ? a.s.p : 0));
  const maxP = Math.max(...rows.map((r) => Math.max(r.t ? r.t.p : 0, r.s ? r.s.p : 0)), 1e-6);
  const sMassOwn = p.scalars.student_mass_own, outflow = sMassOwn == null ? null : 1 - sMassOwn;
  host.innerHTML = `
    <h4>pos ${p.pos} · <span class="label-chip">${p.label}</span>
      ${p.covered ? "" : " · 未覆盖"} ${p.masked ? "" : " · mask=0"}</h4>
    <div class="kv">
      <b>realized</b><span>"${esc(p.realized.text)}" ·
        S lp ${fmt(p.realized.student_lp)} (rank ${rk(p.realized.student_rank)}) ·
        T lp ${fmt(p.realized.teacher_lp)} (rank ${rk(p.realized.teacher_rank)})</span>
      <b>KL fwd/rev</b><span>${fmt(p.scalars.kl_forward)} / ${fmt(p.scalars.kl_reverse)}</span>
      <b>训练 loss</b><span>${fmt(p.scalars.token_loss)}</span>
      <b>gap (T−S)</b><span>${fmt(p.scalars.gap)}</span>
      <b>熵 T/S</b><span>${fmt(p.scalars.teacher_entropy)} / ${fmt(p.scalars.student_entropy)}</span>
      <b>mass T/S支撑/S自</b><span>${fmt(p.scalars.teacher_mass)} /
        ${fmt(p.scalars.student_mass_on_support)} / ${fmt(sMassOwn)}</span>
      <b>overlap</b><span>${p.scalars.overlap_count} / 64</span>
    </div>
    ${outflow != null && outflow > 0.05 ? `<div class="outflow">student 质量外流
      (top-64 之外):<b>${(outflow * 100).toFixed(1)}%</b></div>` : ""}
    <div class="dist">
      <div class="drow"><span class="dim">token</span>
        <span class="legend"><span class="sw" style="background:var(--teacher)"></span>teacher
        <span class="sw" style="background:var(--student)"></span>student</span></div>
      ${rows.slice(0, 40).map((r) => {
        const id = (r.t || r.s).id, hit = id === p.realized.id;
        const text = esc((r.t || r.s).text);
        return `<div class="drow">
          <span class="dtok${hit ? " hit" : ""}" title="id ${id}">"${text}"</span>
          <span class="bars">
            ${r.t ? `<span class="b bt" style="width:${(r.t.p / maxP) * 100}%"
              title="T p=${r.t.p} rank ${r.tr}"></span>` : ""}
            ${r.s ? `<span class="b bs" style="width:${(r.s.p / maxP) * 100}%"
              title="S p=${r.s.p} rank ${r.sr}"></span>` : ""}
          </span></div>`;
      }).join("")}
    </div>
    <div class="ctx">${p.context.tokens.map((t, j) =>
      p.context.start + j === p.pos ? `<span class="cur">${esc(t)}</span>` : esc(t)
    ).join("")}</div>`;
}
const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
const rk = (r) => (r < 0 ? ">64" : `#${r + 1}`);

/* ---------- explore ---------- */
async function viewExplore(q) {
  const m = $("#e-metric");
  if (!m.options.length) {
    META.metrics.forEach((x) => m.add(new Option(x, x)));
    $("#e-lo").value = META.steps[0] ?? 0;
    $("#e-hi").value = META.steps[META.steps.length - 1] ?? 0;
    $("#e-run").onclick = runExplore;
  }
}
async function runExplore() {
  $("#e-status").textContent = "扫描中…(首次较慢,结果进程内缓存)";
  try {
    const rows = await api(`/api/leaderboard?` + new URLSearchParams({
      mode: $("#e-mode").value, metric: $("#e-metric").value,
      step_lo: $("#e-lo").value, step_hi: $("#e-hi").value, limit: 200,
    }));
    $("#e-status").textContent = `${rows.length} 个位置`;
    $("#e-table tbody").innerHTML = rows.map((r) =>
      `<tr data-s="${r.step}" data-h="${r.hash}" data-p="${r.pos}">
        <td>${r.step}</td><td>${esc(r.task.slice(0, 40))}</td>
        <td class="won-${r.won}">${r.won ? "✓" : "✗"}</td>
        <td>${r.pos}</td><td>${r.value == null ? "—" : (+r.value).toFixed(3)}</td>
        <td class="dim">${esc(r.snippet)}</td></tr>`).join("");
    document.querySelectorAll("#e-table tbody tr").forEach((tr) => tr.onclick = () =>
      go("traj", { step: tr.dataset.s, hash: tr.dataset.h, pos: tr.dataset.p }));
  } catch (err) { $("#e-status").textContent = String(err); }
}

/* ---------- evolution ---------- */
async function viewEvolution(q) {
  const taskSel = $("#v-task"), mSel = $("#v-metric");
  if (!mSel.options.length) {
    META.metrics.forEach((x) => mSel.add(new Option(x, x)));
    META.tasks.forEach((t) => taskSel.add(new Option(t.slice(0, 60), t)));
    taskSel.onchange = mSel.onchange = () =>
      go("evolution", { task: taskSel.value, metric: mSel.value });
  }
  const task = q.task || taskSel.value;
  if (!task) return;
  taskSel.value = task;
  const data = await api(`/api/evolution?` + new URLSearchParams({
    task, metric: q.metric || mSel.value }));
  drawEvolution(data);
}
function drawEvolution(data) {
  const cv = $("#v-grid"), ctx = cv.getContext("2d");
  cv.width = cv.clientWidth * devicePixelRatio;
  const steps = [...new Set(data.cells.map((c) => c.step))].sort((a, b) => a - b);
  const turns = Math.max(0, ...data.cells.map((c) => c.turn)) + 1;
  const vals = data.cells.map((c) => c.value);
  const hi = quantile(vals, 0.98) || 1;
  const cw = cv.width / Math.max(steps.length, 1), ch = cv.height / Math.max(turns, 1);
  ctx.clearRect(0, 0, cv.width, cv.height);
  const index = new Map(data.cells.map((c) => [`${c.step}:${c.turn}`, c]));
  steps.forEach((s, x) => { for (let t = 0; t < turns; t++) {
    const c = index.get(`${s}:${t}`);
    ctx.fillStyle = c ? cmap((c.value || 0) / hi) : "#1a1d24";
    ctx.fillRect(x * cw, t * ch, cw - 1, ch - 1);
  } });
  ctx.fillStyle = "#ffffffbb"; ctx.font = `${10 * devicePixelRatio}px monospace`;
  steps.forEach((s, x) => { if (!(x % Math.ceil(steps.length / 20)))
    ctx.fillText(s, x * cw + 2, cv.height - 4); });
  cv.onclick = (ev) => {
    const x = Math.floor(ev.offsetX / cv.clientWidth * steps.length);
    const t = Math.floor(ev.offsetY / cv.clientHeight * turns);
    const c = index.get(`${steps[x]}:${t}`);
    if (c) go("traj", { step: c.step, hash: c.hash });
  };
}

/* ---------- boot ---------- */
(async () => {
  META = await api("/api/meta");
  $("#meta").textContent =
    `${META.steps.length} steps · ${META.obs_dir}` +
    (META.tokenizer ? "" : " · ⚠ 无 tokenizer(仅显示 id)");
  route();
})();
