/* TRACE-OPD 蒸馏显微镜工作台(零依赖 vanilla JS)。
 * 路由:#/samples?step= | #/traj?step=&hash=&track=&pos= | #/explore | #/evolution?task=
 * 一切数值由后端从观察包实时派生;本文件只做渲染与交互。
 *
 * 可用性契约(2026-07-29 用户裁定):任何网络请求都必须让用户看得见——
 * 全局状态条显示"在加载什么、已等几秒、后端正在解压哪个 step";失败必须
 * 显示错误与重试入口,绝不静默停在空白/"载入中"。轨迹详情三段渐进:
 * 骨架(瞬时)→ token 文本 → per-token 指标(重),每段就绪即渲染。 */
"use strict";
const $ = (sel) => document.querySelector(sel);

/* ---------- tracked fetch: every request is visible ---------- */
const ACTIVE = new Map();   // id -> {label, t0}
let REQ_ID = 0;
let LAST_ERR = null;
// One AbortController per view: re-entering a view CANCELS its stale
// requests (the screenshot showed two identical "step 9 样本表" fetches in
// flight — duplicated load that also doubled backend work).
const VIEW_ABORT = {};
function viewSignal(view) {
  if (VIEW_ABORT[view]) VIEW_ABORT[view].abort();
  VIEW_ABORT[view] = new AbortController();
  return VIEW_ABORT[view].signal;
}
async function api(path, label, signal) {
  const id = ++REQ_ID;
  ACTIVE.set(id, { label: label || path, t0: Date.now() });
  renderStatus();
  try {
    const r = await fetch(path, signal ? { signal } : undefined);
    let j;
    try { j = await r.json(); }
    catch { throw new Error(`响应不是 JSON(HTTP ${r.status})`); }
    if (!r.ok) throw new Error(j.error || `HTTP ${r.status}`);
    LAST_ERR = null;
    return j;
  } catch (err) {
    if (err.name !== "AbortError")  // an aborted stale request is not an error
      LAST_ERR = { label: label || path, msg: String(err.message || err) };
    throw err;
  } finally {
    ACTIVE.delete(id);
    renderStatus();
  }
}
// Transient failures (backend busy decompressing) deserve retries, visibly.
async function apiRetry(path, label, signal, tries = 3) {
  for (let a = 1; ; a++) {
    try { return await api(path, a > 1 ? `${label}(重试 ${a}/${tries})` : label, signal); }
    catch (err) {
      if (err.name === "AbortError" || a >= tries) throw err;
      await new Promise((res) => setTimeout(res, 1500 * a * a));
    }
  }
}
let SERVER_STATUS = null;   // /api/status payload while requests are slow
setInterval(async () => {
  renderStatus();
  const oldest = [...ACTIVE.values()].reduce((m, a) => Math.min(m, a.t0), Infinity);
  if (ACTIVE.size && Date.now() - oldest > 2500) {
    try {  // untracked on purpose: the probe must not feed the spinner
      const r = await fetch("/api/status");
      SERVER_STATUS = await r.json();
    } catch { SERVER_STATUS = null; }
  } else if (!ACTIVE.size) SERVER_STATUS = null;
}, 1000);
function renderStatus() {
  const el = $("#g-status");
  if (!el) return;
  let html = "";
  if (ACTIVE.size) {
    const parts = [...ACTIVE.values()].map((a) => {
      const s = ((Date.now() - a.t0) / 1000).toFixed(0);
      return `${a.label} · ${s}s`;
    });
    let extra = "";
    const loading = SERVER_STATUS && SERVER_STATUS.loading;
    if (loading && Object.keys(loading).length) {
      extra = " — 后端正在解压观察包:" + Object.entries(loading)
        .map(([s, t]) => `step ${s}(已 ${t}s${sizeHint(s)})`).join("、");
    }
    html += `<span class="spin"></span> 加载中:${esc(parts.join(" | "))}${esc(extra)}`;
  }
  if (LAST_ERR) {  // shown ALONGSIDE busy — a failure must never be masked
    html += `${html ? "<br>" : ""}<span class="errline">⚠ ${esc(LAST_ERR.label)} 失败:` +
      `${esc(LAST_ERR.msg)}</span> <a href="javascript:location.reload()">刷新</a>` +
      ` <a href="javascript:void(dismissErr())">知道了</a>`;
  }
  el.className = "status" + (LAST_ERR ? " err" : ACTIVE.size ? " busy" : "");
  el.innerHTML = html;
}
function dismissErr() { LAST_ERR = null; renderStatus(); }
function sizeHint(step) {
  const mb = META && META.sizes_mb && META.sizes_mb[String(step)];
  return mb ? `,${mb}MB,首次打开需数十秒` : "";
}

let META = null;
let TRAJ = null;           // { skeleton, tokens, tracks, repeat_mask } — 渐进填充
let TRAJ_KEY = "";         // `${step}:${hash}`
let SEL = -1;              // selected position
let TK_ELS = [];           // token span elements by position (no DOM queries)

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
  const views = { samples: viewSamples, traj: viewTraj,
                  explore: viewExplore, evolution: viewEvolution };
  (views[path] || viewSamples)(q).catch(() => {});  // errors already on the status bar
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
  $("#s-count").textContent = "…";
  const signal = viewSignal("samples");  // cancels the previous step's fetch
  window._rows = await apiRetry(`/api/samples?step=${step}`, `step ${step} 样本表`, signal);
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
      if (k === "task_id") v = esc(String(v).slice(0, 46));
      return `<td>${v}</td>`;
    }).join("") + "</tr>").join("");
  document.querySelectorAll("#s-table tbody tr").forEach((tr) => tr.onclick = () =>
    go("traj", { step: tr.dataset.s, hash: tr.dataset.h }));
}

/* ---------- trajectory microscope (three-stage progressive load) ---------- */
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
  if (TRAJ_KEY === key) {
    if (q.pos != null && +q.pos !== SEL) selectPos(+q.pos, false);
    return;
  }
  TRAJ_KEY = key;
  TRAJ = { skeleton: null, tokens: null, tracks: null, repeat_mask: null,
           tokensDone: 0, tracksDone: 0, n: 0 };
  SEL = -1;
  $("#t-title").innerHTML = `<span class="spin"></span> 载入骨架…${esc(sizeHint(q.step))}`;
  $("#t-audit").textContent = "";
  $("#t-tokens").innerHTML = `<div class="dim pad">等待骨架…</div>`;
  $("#t-inspector").innerHTML = `<div class="dim pad">点击一个 token(指标就绪后)</div>`;

  const stale = () => TRAJ_KEY !== key;
  const signal = viewSignal("traj");  // switching rows cancels the old streams
  // 1) skeleton first (cheap): it carries resp_len, which drives the
  //    chunked streams below — then tokens & tracks stream IN PARALLEL,
  //    each 2000-token window rendering the moment it lands.
  let sk;
  try {
    sk = await apiRetry(`/api/sample?step=${q.step}&hash=${q.hash}`, "轨迹骨架", signal);
  } catch (err) {
    if (!stale() && err.name !== "AbortError") {
      $("#t-title").innerHTML = `<span class="errline">骨架加载失败:` +
        `${esc(String(err.message || err))}</span> ` +
        `<a href="javascript:void(retryTraj())">重试</a>`;
      window._retryQ = q;
      TRAJ_KEY = "";  // unlock: re-clicking the same row restarts the view
    }
    return;
  }
  if (stale()) return;
  TRAJ.skeleton = sk;
  TRAJ.n = sk.summary.resp_len;
  renderSkeleton();
  TRAJ.tokens = new Array(TRAJ.n).fill(null);
  TRAJ.tracks = null;  // allocated on the first tracks chunk (keys unknown yet)
  $("#t-tokens").innerHTML = "";
  buildTokensShell();

  const CHUNK = 2000;
  const streamTokens = (async () => {
    for (let lo = 0; lo < TRAJ.n; lo += CHUNK) {
      if (stale()) return;
      const part = await apiRetry(
        `/api/sample_tokens?step=${q.step}&hash=${q.hash}&lo=${lo}&hi=${lo + CHUNK}`,
        `token ${lo}–${Math.min(lo + CHUNK, TRAJ.n)}/${TRAJ.n}`, signal);
      if (stale()) return;
      part.tokens.forEach((t, j) => { TRAJ.tokens[part.lo + j] = t; });
      TRAJ.tokensDone = part.hi;
      fillTokenChunk(part.lo, part.hi);
    }
  })().catch(() => { if (!stale()) { window._retryQ = q; $("#t-tokens").insertAdjacentHTML(
      "beforeend", `<div class="dim pad">token 流中断 —
      <a href="javascript:void(retryTraj())">重新载入本轨迹</a></div>`); } });

  const streamTracks = (async () => {
    for (let lo = 0; lo < TRAJ.n; lo += CHUNK) {
      if (stale()) return;
      const part = await apiRetry(
        `/api/sample_tracks?step=${q.step}&hash=${q.hash}&lo=${lo}&hi=${lo + CHUNK}`,
        lo === 0 ? "指标派生(首次较慢,之后走缓存)" :
          `指标 ${lo}–${Math.min(lo + CHUNK, TRAJ.n)}/${TRAJ.n}`, signal);
      if (stale()) return;
      if (!TRAJ.tracks) {
        TRAJ.tracks = {}; TRAJ.repeat_mask = new Array(TRAJ.n).fill(0);
        Object.keys(part.tracks).forEach((k) => TRAJ.tracks[k] = new Array(TRAJ.n).fill(null));
      }
      Object.entries(part.tracks).forEach(([k, arr]) =>
        arr.forEach((v, j) => { TRAJ.tracks[k][part.lo + j] = v; }));
      part.repeat_mask.forEach((v, j) => { TRAJ.repeat_mask[part.lo + j] = v; });
      TRAJ.tracksDone = part.hi;
      paintTokens(part.lo, part.hi); paintMinimap(); applyMaskClasses(part.lo, part.hi);
    }
    if (!stale()) { paintTokens(); applyMaskClasses(); }  // final full pass (scale settles)
    if (!stale() && q.pos != null) selectPos(+q.pos, false);
  })().catch(() => { if (!stale()) { window._retryQ = q; $("#t-inspector").innerHTML =
      `<div class="dim pad">指标派生中断 —
      <a href="javascript:void(retryTraj())">重新载入本轨迹</a></div>`; } });

  await Promise.allSettled([streamTokens, streamTracks]);
}
function renderSkeleton() {
  const s = TRAJ.skeleton.summary;
  $("#t-title").innerHTML =
    `<b>${esc(s.task_id.slice(0, 40))}</b> · step ${s.step} · ` +
    `<span class="won-${s.won}">${s.won ? "WON" : "LOST"}</span> · ` +
    `${s.resp_len} tok · ${s.turns} turns · ${TRAJ.skeleton.loss_mode}`;
  const a = TRAJ.skeleton.audit;
  $("#t-audit").textContent =
    `audit ${a.ok ? "✓" : "✗"} (${a.checked} tok, exceed ${a.n_exceed ?? 0}, ` +
    `max_err ${(+a.max_err).toFixed(3)})`;
}
function buildTokensShell() {
  // one placeholder span per position, filled chunk by chunk as text streams
  const host = $("#t-tokens");
  host.innerHTML = "";
  const spans = (TRAJ.skeleton && TRAJ.skeleton.turn_spans) || [];
  const starts = new Map(spans.map(([a], i) => [a, i]));
  const frag = document.createDocumentFragment();
  TK_ELS = new Array(TRAJ.n);
  for (let i = 0; i < TRAJ.n; i++) {
    if (starts.has(i)) {
      const sep = document.createElement("span");
      sep.className = "turn-sep"; sep.textContent = `── turn ${starts.get(i) + 1} ──`;
      frag.appendChild(sep);
    }
    const el = document.createElement("span");
    el.className = "tk pending"; el.dataset.i = i;
    el.textContent = "·";
    el.onclick = () => selectPos(i, true);
    TK_ELS[i] = el;
    frag.appendChild(el);
  }
  host.appendChild(frag);
  host.onkeydown = (ev) => {
    if (SEL < 0 || !TRAJ.tracks) return;
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
function fillTokenChunk(lo, hi) {
  for (let i = lo; i < hi; i++) {
    const el = TK_ELS[i], t = TRAJ.tokens[i];
    if (el && t) { el.textContent = t.text.replace(/\n/g, "⏎\n"); el.classList.remove("pending"); }
  }
  paintTokens(lo, hi);
}
function applyMaskClasses(lo = 0, hi = Infinity) {
  if (!TRAJ.tracks) return;
  const end = Math.min(hi, TK_ELS.length);
  for (let i = lo; i < end; i++) {
    const el = TK_ELS[i];
    if (!el) continue;
    el.classList.toggle("obs", !TRAJ.tracks.mask[i]);
    el.classList.toggle("uncov", !TRAJ.tracks.covered[i]);
  }
}
function paintTokens(lo = 0, hi = Infinity) {
  if (!TRAJ || !TRAJ.tracks) return;
  const track = $("#t-track").value, vals = TRAJ.tracks[track] || [];
  const q98 = quantile(vals, 0.98) || 1, q02 = Math.min(0, quantile(vals, 0.02));
  const end = Math.min(hi, TK_ELS.length);
  for (let i = lo; i < end; i++) {
    const el = TK_ELS[i];
    if (!el) continue;
    const v = vals[i];
    el.style.background = v == null ? "transparent"
      : cmap((v - q02) / (q98 - q02 || 1)) + "55";
  }
}
function paintMinimap() {
  if (!TRAJ || !TRAJ.tracks) return;
  const cv = $("#t-minimap"), ctx = cv.getContext("2d");
  const n = TRAJ.n;
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
    if (TK_ELS[i]) TK_ELS[i].scrollIntoView({ block: "center" });
  };
}
let SEL_SEQ = 0;
async function selectPos(i, push) {
  SEL = i;
  const seq = ++SEL_SEQ;
  document.querySelectorAll("#t-tokens .tk.sel").forEach((e) => e.classList.remove("sel"));
  const el = TK_ELS[i];
  if (el) el.classList.add("sel");
  $("#t-tokens").focus({ preventScroll: true });
  paintMinimap();
  const [step, hash] = TRAJ_KEY.split(":");
  if (push) history.replaceState(null, "",
    `#/traj?${new URLSearchParams({ step, hash, track: $("#t-track").value, pos: i })}`);
  try {
    const p = await api(`/api/position?step=${step}&hash=${hash}&pos=${i}`, `pos ${i}`);
    if (seq !== SEL_SEQ) return;  // a newer selection superseded this response
    renderInspector(p);
  } catch { /* status bar already shows it */ }
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
function retryTraj() { TRAJ_KEY = ""; viewTraj(window._retryQ || {}); }
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
    }), "检索扫描");
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
    // task list is a full-dir scan — fetched lazily, only for this view
    const tasks = await api("/api/tasks", "任务列表");
    tasks.forEach((t) => taskSel.add(new Option(t.slice(0, 60), t)));
    taskSel.onchange = mSel.onchange = () =>
      go("evolution", { task: taskSel.value, metric: mSel.value });
  }
  const task = q.task || taskSel.value;
  if (!task) return;
  taskSel.value = task;
  const data = await api(`/api/evolution?` + new URLSearchParams({
    task, metric: q.metric || mSel.value }), "演化格(首次跨步扫描较慢)");
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
  try {
    META = await api("/api/meta", "连接后端");
  } catch (err) {
    $("#meta").innerHTML = `<span class="err-inline">无法连接后端:${esc(String(err.message || err))}
      — 确认 serve 进程与 SSH 隧道后<a href="javascript:location.reload()">刷新</a></span>`;
    return;
  }
  $("#meta").textContent =
    `${META.steps.length} steps · ${META.obs_dir}` +
    (META.tokenizer ? "" : " · ⚠ 无 tokenizer(仅显示 id)");
  route();
})();
