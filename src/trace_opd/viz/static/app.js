// TRACE-OPD observatory — hash-routed, zero-dependency SPA.
// Every view is addressable (deep-link) and every aggregate ships provenance.

import { lineChart, chartLegend } from "/static/chart.js";
import { renderMarkdown } from "/static/md.js";

const view = document.getElementById("view");
const crumbs = document.getElementById("crumbs");
const esc = (s) => String(s ?? "")
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  .replace(/"/g, "&quot;");
const pct = (v) => (v == null ? "—" : (v * 100).toFixed(1) + "%");
const num = (v) => (v == null ? "—" : typeof v === "number" && !Number.isInteger(v) ? v.toFixed(2) : v);
const when = (ts) => (ts ? new Date(ts * 1000).toLocaleString() : "—");

async function api(path) {
  const resp = await fetch(path);
  if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).error || resp.statusText);
  return resp.json();
}

function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.hidden = false;
  setTimeout(() => (t.hidden = true), 1800);
}

function copyText(text) {
  navigator.clipboard?.writeText(text).then(() => toast("copied"), () => toast("copy failed"));
}

// ------------------------------------------------------------------ routing

function parseHash() {
  const [path, query] = location.hash.replace(/^#\/?/, "").split("?");
  return { parts: path.split("/").filter(Boolean).map(decodeURIComponent),
           params: new URLSearchParams(query || "") };
}

const link = {
  runs: () => "#/runs",
  run: (r) => `#/run/${encodeURIComponent(r)}`,
  step: (r, s, extra = "") => `#/run/${encodeURIComponent(r)}/step/${s}${extra}`,
  rollout: (r, f) => `#/run/${encodeURIComponent(r)}/rollout?file=${encodeURIComponent(f)}`,
  research: (r, d) => `#/run/${encodeURIComponent(r)}/research?dir=${encodeURIComponent(d)}`,
  task: (r, t) => `#/run/${encodeURIComponent(r)}/task?task_id=${encodeURIComponent(t)}`,
};

async function route() {
  const { parts, params } = parseHash();
  view.innerHTML = '<p class="muted">loading…</p>';
  try {
    if (!parts.length || parts[0] === "runs") return await viewRuns();
    if (parts[0] === "run" && parts.length >= 2) {
      const run = parts[1];
      if (parts.length === 2) return await viewRun(run);
      if (parts[2] === "step") return await viewStep(run, parts[3], params);
      if (parts[2] === "rollout") return await viewRollout(run, params.get("file"));
      if (parts[2] === "research") return await viewResearch(run, params.get("dir"));
      if (parts[2] === "task") return await viewTask(run, params.get("task_id"));
      if (parts[2] === "microscope") return await viewMicroscope(run, params);
    }
    view.innerHTML = '<p class="bad">unknown route</p>';
  } catch (err) {
    view.innerHTML = `<p class="bad">error: ${esc(err.message)}</p>`;
  }
}

function setCrumbs(items) {
  crumbs.innerHTML = items
    .map((c) => (c.href ? `<a href="${c.href}">${esc(c.label)}</a>` : esc(c.label)))
    .join(" <span class='muted'>›</span> ");
}

// -------------------------------------------------------------------- views

async function viewRuns() {
  setCrumbs([{ label: "runs" }]);
  const { runs } = await api("/api/runs");
  view.innerHTML = `
    <h1>Runs</h1>
    <div class="panel"><table>
      <tr><th>run</th><th>samples</th><th>last step</th><th>train score</th>
          <th>last data</th><th>last scan</th></tr>
      ${runs.map((r) => `
        <tr class="click" data-href="${link.run(r.run)}">
          <td><b>${esc(r.run)}</b></td><td>${num(r.samples)}</td>
          <td>${num(r.last_step)}</td><td>${pct(r.train_score)}</td>
          <td class="muted">${when(r.last_ts)}</td>
          <td class="muted">${when(r.last_scan)}</td>
        </tr>`).join("")}
    </table>${runs.length ? "" : '<p class="muted">no runs indexed yet</p>'}</div>`;
  bindRowLinks();
}

const CURVE_GROUPS = [
  { title: "success — sampled train batch vs doc-privileged research",
    sub: "gap ≈ privileged-doc gain; converging gap = internalization",
    keys: ["score", "research_final_success"] },
  { title: "env steps (stratified)",
    sub: "success/failure split — aggregate medians mislead",
    keys: ["env_steps_success", "env_steps_failed"] },
  { title: "protocol health", sub: "format failures per episode / truncation",
    keys: ["invalid_format", "truncated"] },
  { title: "degeneration watch", sub: "longest action repeat / invalid-step share",
    keys: ["max_repeat", "invalid_rate"] },
  { title: "research loop", sub: "doc missing / research errors",
    keys: ["doc_missing", "research_error"] },
  { title: "doc size", sub: "mean words per document", keys: ["doc_words"] },
];

async function viewRun(run) {
  setCrumbs([{ label: "runs", href: link.runs() }, { label: run }]);
  const [ov, curves, val] = await Promise.all([
    api(`/api/run/${run}/overview`),
    api(`/api/run/${run}/curves?mode=train&keys=` +
        CURVE_GROUPS.flatMap((g) => g.keys).join(",")),
    api(`/api/run/${run}/curves?mode=val&keys=score`),
  ]);
  const train = ov.steps.filter((s) => !s.validate);
  const short = train.filter((s) => s.short);
  const latest = train[train.length - 1];
  view.innerHTML = `
    <h1>${esc(run)}</h1>
    <div class="panel kpis">
      <div class="kpi"><div class="v">${num(latest?.step)}</div><div class="k">latest train step</div></div>
      <div class="kpi"><div class="v">${latest ? pct(latest.n_success / latest.n) : "—"}</div><div class="k">latest step success</div></div>
      <div class="kpi"><div class="v">${num(ov.expected_per_step)}</div><div class="k">expected dumps/step</div></div>
      <div class="kpi"><div class="v ${short.length ? "bad" : "ok"}">${short.length}</div><div class="k">short steps (integrity)</div></div>
      <div class="kpi"><div class="v">${num(ov.research.n)}</div><div class="k">research sessions</div></div>
      <div class="kpi"><div class="v">${pct(ov.research.final_success)}</div><div class="k">doc final success</div></div>
      <div class="kpi"><div class="v muted">${when(ov.scan.last_scan)}</div><div class="k">last scan</div></div>
    </div>
    ${short.length ? `<div class="panel"><b class="bad">integrity:</b> steps with missing dumps:
      ${short.map((s) => `<a href="${link.step(run, s.step)}">${s.step} (${s.n}/${ov.expected_per_step})</a>`).join(", ")}
      <span class="muted">— may just be the step currently in flight</span></div>` : ""}
    <p><a href="#/run/${encodeURIComponent(run)}/microscope">→ distillation microscope</a>
       <span class="muted">(per-token KL depth terrain &amp; sampled trajectories — needs offline collect)</span></p>
    <h2>Train curves <span class="muted">(from per-sample dumps; click a point to open the step)</span></h2>
    <div class="grid charts" id="charts"></div>
    <h2>Validation (greedy, per split)</h2>
    <div class="grid charts" id="valcharts"></div>
    <h2>Steps</h2>
    <div class="panel" id="steps"></div>`;

  const charts = document.getElementById("charts");
  for (const group of CURVE_GROUPS) {
    const series = group.keys
      .filter((k) => curves.series[k])
      .map((k) => ({ name: curves.series[k].label, points: curves.series[k].points }));
    const card = document.createElement("div");
    card.className = "panel";
    card.innerHTML = `<div class="chart-title">${esc(group.title)}</div>
                      <div class="chart-sub">${esc(group.sub)}</div>`;
    card.appendChild(lineChart(series, { onPoint: (s) => (location.hash = link.step(run, s)) }));
    card.appendChild(chartLegend(series));
    charts.appendChild(card);
  }
  const vals = Object.entries(val.series.score?.splits || {})
    .map(([split, points]) => ({ name: split, points }));
  const vcard = document.createElement("div");
  vcard.className = "panel";
  vcard.innerHTML = `<div class="chart-title">val score by split</div>
                     <div class="chart-sub">seen vs unseen gap reads memorization vs strategy</div>`;
  vcard.appendChild(lineChart(vals, { onPoint: (s) => (location.hash = link.step(run, s, "?validate=1")) }));
  vcard.appendChild(chartLegend(vals));
  document.getElementById("valcharts").appendChild(vcard);

  document.getElementById("steps").innerHTML = ov.steps.map((s) => `
    <div class="step-card">
      <div class="step-head" data-href="${link.step(run, s.step, s.validate ? "?validate=1" : "")}">
        <span class="t">step ${s.step}${s.validate ? '<span class="tag dim">val</span>' : ""}</span>
        <span>${pct(s.n_success / s.n)} success</span>
        <span class="muted">${s.n} rollouts${s.short ? ' · <span class="bad">short</span>' : ""}</span>
      </div>
    </div>`).join("");
  view.querySelectorAll(".step-head").forEach((n) =>
    n.addEventListener("click", () => (location.hash = n.dataset.href)));
}

async function viewStep(run, step, params) {
  const validate = params.get("validate") === "1";
  setCrumbs([{ label: "runs", href: link.runs() }, { label: run, href: link.run(run) },
             { label: `step ${step}${validate ? " (val)" : ""}` }]);
  const qs = new URLSearchParams(params);
  const data = await api(`/api/run/${run}/step/${step}/rollouts?${qs}`);
  const st = data.stats;
  const filterChip = (key, value, label) => {
    const on = params.get(key) === value;
    const next = new URLSearchParams(params);
    on ? next.delete(key) : next.set(key, value);
    return `<span class="chip click ${on ? "on" : ""}"
      data-href="${link.step(run, step, "?" + next)}">${esc(label)}</span>`;
  };
  const failTotal = st.fail_labels.reduce((a, d) => a + d.count, 0) || 1;
  const stackColors = ["#d73027", "#b8860b", "#8b5cf6", "#67718a", "#0f9bb8"];
  view.innerHTML = `
    <h1>step ${esc(step)} ${validate ? '<span class="tag dim">val</span>' : ""}</h1>
    <div class="panel kpis">
      <div class="kpi"><div class="v">${pct(st.success_rate)}</div><div class="k">success (${st.n_success}/${st.n})</div></div>
      <div class="kpi"><div class="v">${num(st.env_steps_success_median)}</div><div class="k">median steps · success</div></div>
      <div class="kpi"><div class="v">${num(st.env_steps_failed_median)}</div><div class="k">median steps · failed</div></div>
      ${st.n_expert ? `<div class="kpi"><div class="v">${st.n_expert}</div><div class="k">expert rows (excluded)</div></div>` : ""}
    </div>
    <div class="panel">
      <b>failure stages</b> <span class="muted">(click to filter)</span>
      <div class="stack">${st.fail_labels.map((d, i) =>
        `<div style="width:${(d.count / failTotal) * 100}%;background:${stackColors[i % stackColors.length]}"
             title="${esc(d.value)}: ${d.count}"></div>`).join("")}</div>
      <div>${st.fail_labels.map((d) => filterChip("fail_label", d.value, `${d.value} · ${d.count}`)).join("")}</div>
      <div style="margin-top:8px"><b>families</b>
        ${st.families.map((d) => filterChip("family", d.value, `${d.value} · ${d.count}`)).join("")}</div>
      <div style="margin-top:8px"><b>result</b>
        ${filterChip("won", "1", "success")} ${filterChip("won", "0", "failed")}</div>
      <div class="provenance">
        <code class="mono">${esc(data.provenance.scope)}</code>
        <span>${esc(data.provenance.criteria)}</span>
        <button class="small" id="cite">copy citation</button>
      </div>
    </div>
    <div class="panel"><table>
      <tr><th>task</th><th>#</th><th>result</th><th>stage</th><th>steps</th>
          <th>fmt✗</th><th>repeat</th><th>tags</th></tr>
      ${data.rollouts.map((r) => `
        <tr class="click" data-href="${link.rollout(run, r.file)}">
          <td class="mono">${esc(r.task_id)}${r.sample_role === "sft"
                ? '<span class="tag warn">expert</span>' : ""}</td>
          <td>${r.sample_idx}</td>
          <td>${r.won ? '<span class="ok">✓</span>' : `<span class="bad">${esc(r.outcome)}</span>`}</td>
          <td>${esc(r.fail_label)}</td>
          <td>${r.env_steps}${r.truncated ? '<span class="tag warn">trunc</span>' : ""}</td>
          <td>${r.invalid_format || ""}</td>
          <td>${r.max_repeat > 2 ? `<span class="warn">${r.max_repeat}</span>` : r.max_repeat}</td>
          <td>${(r.tags || "").split(",").filter(Boolean)
                .map((t) => `<span class="tag warn">${esc(t)}</span>`).join("")}</td>
        </tr>`).join("")}
    </table></div>`;
  document.getElementById("cite").addEventListener("click", () =>
    copyText(data.provenance.citation));
  view.querySelectorAll(".chip.click").forEach((n) =>
    n.addEventListener("click", () => (location.hash = n.dataset.href)));
  bindRowLinks();
}

async function viewRollout(run, file) {
  const d = await api(`/api/run/${run}/rollout?file=${encodeURIComponent(file)}`);
  const p = d.payload;
  setCrumbs([{ label: "runs", href: link.runs() }, { label: run, href: link.run(run) },
             { label: `step ${p.global_step}`,
               href: link.step(run, p.global_step, p.validate ? "?validate=1" : "") },
             { label: "rollout" }]);
  const tagCls = { invalid: "bad", no_action: "warn", repeat: "warn" };
  view.innerHTML = `
    <h1 class="mono" style="font-size:15px">${esc(d.task_id)}</h1>
    <div class="panel kpis">
      <div class="kpi"><div class="v ${d.won ? "ok" : "bad"}">${esc(p.outcome)}</div><div class="k">outcome</div></div>
      <div class="kpi"><div class="v">${p.env_steps}</div><div class="k">env steps</div></div>
      <div class="kpi"><div class="v">${p.invalid_format}</div><div class="k">format failures</div></div>
      <div class="kpi"><div class="v">${p.truncated ? "yes" : "no"}</div><div class="k">truncated</div></div>
      <div class="kpi"><div class="v">${esc(d.family)}</div><div class="k">family</div></div>
    </div>
    <div class="toolbar">
      ${d.research_archive
        ? `<a class="small" style="padding:4px 10px;border:1px solid var(--line);border-radius:6px"
             href="${link.research(run, d.research_archive)}">→ research doc that taught this step</a>`
        : '<span class="muted">no research archive linked</span>'}
      <a class="small" style="padding:4px 10px;border:1px solid var(--line);border-radius:6px"
         href="${link.task(run, d.task_id)}">→ this task across steps</a>
      <button class="small" id="citefile">copy file path</button>
      ${d.traj_tags.map((t) => `<span class="tag warn">${esc(t)}</span>`).join("")}
    </div>
    <div class="panel"><b>initial observation</b>
      <div class="obs mono" style="white-space:pre-wrap;margin-top:6px">${esc(d.initial_obs)}</div></div>
    ${d.steps.map((s, i) => {
      const cls = s.tags.includes("no_action") ? "noact"
        : s.tags.includes("invalid") ? "invalid"
        : d.won && i === d.steps.length - 1 ? "winstep" : "";
      return `
      <div class="turn ${cls}">
        <div class="head">
          <span class="no">[${s.t}]</span>
          <span class="act">${esc(s.action_norm || "(no action)")}</span>
          ${s.tags.map((t) => `<span class="tag ${tagCls[t] || "dim"}">${esc(t)}</span>`).join("")}
        </div>
        <details><summary>thought (${(s.thought || "").length} chars)</summary>
          <p>${esc(s.thought)}</p></details>
        <div class="obs">${esc(s.observation)}</div>
      </div>`;
    }).join("")}`;
  document.getElementById("citefile").addEventListener("click", () =>
    copyText(`runs/${run}/${file}`));
}

async function viewResearch(run, dir) {
  const d = await api(`/api/run/${run}/research?dir=${encodeURIComponent(dir)}`);
  const m = d.meta;
  setCrumbs([{ label: "runs", href: link.runs() }, { label: run, href: link.run(run) },
             { label: "research" }]);
  view.innerHTML = `
    <h1 class="mono" style="font-size:15px">${esc(m.task_id)} <span class="tag dim">v${esc(m.policy_version)}</span></h1>
    <div class="panel kpis">
      <div class="kpi"><div class="v">${num(m.experiments_used)}</div><div class="k">experiments</div></div>
      <div class="kpi"><div class="v">${num(m.episodes_consumed)}</div><div class="k">episodes</div></div>
      <div class="kpi"><div class="v">${m.last_exp_successes}/${m.last_exp_attempts}</div><div class="k">final experiment</div></div>
      <div class="kpi"><div class="v">${num(m.doc_words)}</div><div class="k">doc words</div></div>
      <div class="kpi"><div class="v">${esc(m.researcher_model || "?")}</div><div class="k">researcher</div></div>
      <div class="kpi"><div class="v ${m.session_error ? "bad" : "ok"}">${m.session_error ? "error" : "clean"}</div><div class="k">session</div></div>
    </div>
    <div class="toolbar">
      <a href="${link.task(run, m.task_id)}">→ this task across steps</a>
      <button class="small" id="citedir">copy archive path</button>
    </div>
    <div class="grid cols2">
      <div class="panel"><h2 style="margin-top:0">document</h2>
        ${d.doc ? `<div class="doc">${renderMarkdown(d.doc)}</div>`
                : '<p class="muted">no doc.md (doc_missing)</p>'}</div>
      <div class="panel"><h2 style="margin-top:0">rollouts taught by this session</h2>
        <table><tr><th>step</th><th>result</th><th>stage</th><th>steps</th></tr>
        ${d.consumers.map((c) => `
          <tr class="click" data-href="${link.rollout(run, c.file)}">
            <td>${c.step}${c.validate ? " (val)" : ""}</td>
            <td>${c.won ? '<span class="ok">✓</span>' : `<span class="bad">${esc(c.outcome)}</span>`}</td>
            <td>${esc(c.fail_label)}</td><td>${c.env_steps}</td>
          </tr>`).join("")}
        </table>
        ${m.guidance_log?.length
          ? `<h2>guidance evolution</h2><ol>${m.guidance_log
              .map((g) => `<li class="muted">${esc(String(g).slice(0, 160))}</li>`).join("")}</ol>`
          : ""}
      </div>
    </div>`;
  document.getElementById("citedir").addEventListener("click", () =>
    copyText(`runs/${run}/${dir}`));
  bindRowLinks();
}

async function viewTask(run, taskId) {
  const d = await api(`/api/run/${run}/task?task_id=${encodeURIComponent(taskId)}`);
  setCrumbs([{ label: "runs", href: link.runs() }, { label: run, href: link.run(run) },
             { label: "task history" }]);
  view.innerHTML = `
    <h1 class="mono" style="font-size:15px">${esc(taskId)}
      <span class="tag dim">${esc(d.family)}</span></h1>
    <h2>Across steps <span class="muted">(success/failure always stratified)</span></h2>
    ${d.history.map((h) => `
      <div class="panel">
        <div class="step-head">
          <span class="t"><a href="${link.step(run, h.step, h.validate ? "?validate=1" : "")}">step ${h.step}</a>
            ${h.validate ? '<span class="tag dim">val</span>' : ""}</span>
          <span>${pct(h.success_rate)} (${h.n_success}/${h.n})</span>
          <span class="muted">steps: ✓ ${num(h.env_steps_success_median)} / ✗ ${num(h.env_steps_failed_median)}</span>
          <span>${h.fail_labels.map((f) => `<span class="chip">${esc(f.value)} · ${f.count}</span>`).join("")}</span>
        </div>
        <div style="margin-top:6px">${h.rollouts.map((r) => `
          <a class="chip" href="${link.rollout(run, r.file)}"
             title="${esc(r.fail_label)} · ${r.env_steps} steps">#${r.sample_idx}
             ${r.won ? "✓" : "✗"}</a>`).join("")}</div>
      </div>`).join("")}
    <h2>Research sessions</h2>
    <div class="panel"><table>
      <tr><th>policy ver</th><th>doc</th><th>words</th><th>experiments</th>
          <th>episodes</th><th>final success</th><th>error</th></tr>
      ${d.research_sessions.map((s) => `
        <tr class="click" data-href="${link.research(run, s.dir)}">
          <td>${num(s.policy_version)}</td>
          <td>${s.doc_present ? "✓" : '<span class="bad">missing</span>'}</td>
          <td>${s.doc_words}</td><td>${s.experiments_used}</td>
          <td>${s.episodes_consumed}</td><td>${pct(s.final_success)}</td>
          <td>${s.session_error ? `<span class="bad">${esc(s.session_error)}</span>` : ""}</td>
        </tr>`).join("")}
    </table></div>`;
  bindRowLinks();
}

// Viridis-ish 5-stop interpolation for the terrain cells (sequential, CB-safe).
const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
function viridis(x) {
  x = Math.max(0, Math.min(1, x)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(x)), f = x - i;
  const c = VIRIDIS[i].map((v, k) => Math.round(v + f * (VIRIDIS[i + 1][k] - v)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

async function viewMicroscope(run, params) {
  setCrumbs([{ label: "runs", href: link.runs() }, { label: run, href: link.run(run) },
             { label: "microscope" }]);
  const d = await api(`/api/run/${run}/microscope`);
  if (!d.available) {
    view.innerHTML = `<h1>Distillation microscope</h1>
      <div class="panel"><p class="muted">No collect outputs found for this run.
      Produce them offline with:</p>
      <p class="mono">python -m trace_opd.analysis.collect --run-dir runs/${esc(run)}</p></div>`;
    return;
  }
  const gmax = Math.max(...d.matrix.flat().filter((v) => v != null), 1e-6);
  const cw = Math.max(6, Math.min(18, Math.floor(760 / d.turns.length)));
  const ch = Math.max(3, Math.min(10, Math.floor(420 / d.steps.length)));
  const cells = d.steps.flatMap((s, si) => d.turns.map((t, ti) => {
    const v = d.matrix[si][ti];
    return v == null ? "" :
      `<rect x="${ti * cw}" y="${(d.steps.length - 1 - si) * ch}" width="${cw - 0.5}"
        height="${ch - 0.5}" fill="${viridis(v / gmax)}"><title>step ${s} · turn ${t} · G=${v}</title></rect>`;
  })).join("");
  const selStep = params.get("step"), selTraj = params.get("traj");
  view.innerHTML = `
    <h1>Distillation microscope</h1>
    <div class="panel">
      <div class="chart-title">depth terrain — per-turn residual privileged info G<sub>τ</sub></div>
      <div class="chart-sub">x = turn depth, y = training step (up = later); bright = more unconsumed doc signal</div>
      <svg viewBox="0 0 ${d.turns.length * cw} ${d.steps.length * ch}"
           style="width:100%;max-height:440px" preserveAspectRatio="none">${cells}</svg>
      <div class="legend">G ∈ [0, ${gmax.toFixed(3)}] nats/token · steps ${d.steps[0]}–${d.steps[d.steps.length - 1]}</div>
    </div>
    <div class="grid cols2">
      <div class="panel"><h2 style="margin-top:0">sampled trajectories</h2>
        <table><tr><th>step</th><th>trajectory</th><th>turns</th><th>mean G</th></tr>
        ${d.trajs.map((t) => `
          <tr class="click ${String(t.step) === selStep && t.traj_id === selTraj ? "on" : ""}"
              data-href="#/run/${encodeURIComponent(run)}/microscope?step=${t.step}&traj=${encodeURIComponent(t.traj_id)}">
            <td>${t.step}</td><td class="mono">${esc(t.traj_id.split("/").pop())}</td>
            <td>${t.turns}</td><td>${t.g_mean}</td>
          </tr>`).join("")}
        </table></div>
      <div class="panel" id="track"><h2 style="margin-top:0">token strip</h2>
        <p class="muted">select a trajectory — color = per-token KL, hover for detail</p></div>
    </div>`;
  bindRowLinks();
  if (selStep && selTraj) {
    const track = await api(
      `/api/run/${run}/microscope_track?step=${selStep}&traj=${encodeURIComponent(selTraj)}`);
    const kmax = Math.max(...track.tokens.map((t) => t.kl), 1e-6);
    let lastTurn = null;
    const strip = track.tokens.map((t) => {
      const sep = t.turn !== lastTurn ? `<span class="turnsep">t${t.turn}</span>` : "";
      lastTurn = t.turn;
      const alpha = Math.min(1, t.kl / kmax);
      const label = t.token_text ? esc(t.token_text) : "·";
      return `${sep}<span class="tok" style="background:rgba(214,69,39,${(alpha * 0.85).toFixed(3)})"
        title="t=${t.t} turn=${t.turn} kl=${t.kl}${t.token_type ? " · " + esc(t.token_type) : ""}">${label}</span>`;
    }).join("");
    document.getElementById("track").innerHTML = `
      <h2 style="margin-top:0">token strip — step ${esc(selStep)}</h2>
      <div class="mono" style="font-size:11px;color:var(--muted);margin-bottom:6px">${esc(selTraj)}</div>
      <div class="tokstrip">${strip}</div>
      <div class="legend" style="margin-top:6px">red intensity = KL (max ${kmax.toFixed(3)});
        ${track.tokens.some((t) => t.token_type) ? "annotated" : "run the annotator for token text/types"}</div>`;
  }
}

function bindRowLinks() {
  view.querySelectorAll("tr.click[data-href]").forEach((n) =>
    n.addEventListener("click", () => (location.hash = n.dataset.href)));
}

// -------------------------------------------------------------------- theme

const themeBtn = document.getElementById("theme");
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  localStorage.setItem("viz-theme", t);
}
themeBtn.addEventListener("click", () =>
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
applyTheme(localStorage.getItem("viz-theme") ||
  (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));

addEventListener("hashchange", route);
route();
