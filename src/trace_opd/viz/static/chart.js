// Minimal SVG line chart — no external libs (CSP/self-contained by design).
// Handles multi-series step curves with nice ticks and clickable points.

const NS = "http://www.w3.org/2000/svg";
const PALETTE = ["#2f6fed", "#d73027", "#1a9850", "#b8860b", "#8b5cf6", "#0f9bb8"];

function el(name, attrs = {}, text) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function niceTicks(lo, hi, n = 4) {
  if (!isFinite(lo) || !isFinite(hi)) return [];
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  const span = hi - lo;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * step0)
    .find((s) => span / s <= n + 1) || step0 * 10;
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step)
    ticks.push(Number(t.toFixed(6)));
  return ticks;
}

const fmt = (v) =>
  Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3);

// Tick labels need just enough decimals to distinguish adjacent ticks.
const tickFmt = (v, step) => v.toFixed(Math.max(0, -Math.floor(Math.log10(step || 1))));

// series: [{name, color?, points: [{step, value, n}]}]; onPoint(step) optional.
export function lineChart(series, { onPoint } = {}) {
  const W = 420, H = 190, L = 44, R = 8, T = 10, B = 24;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart" });
  const pts = series.flatMap((s) => s.points);
  if (!pts.length) {
    svg.appendChild(el("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, "no data"));
    return svg;
  }
  const xs = pts.map((p) => p.step), ys = pts.map((p) => p.value);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys, 0 < Math.min(...ys) ? Math.min(...ys) : 0);
  let y1 = Math.max(...ys);
  if (y0 === y1) y1 = y0 + 1;
  const X = (v) => x1 === x0 ? (L + (W - L - R) / 2) : L + ((v - x0) / (x1 - x0)) * (W - L - R);
  const Y = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);

  const yTicks = niceTicks(y0, y1);
  const yStep = yTicks.length > 1 ? yTicks[1] - yTicks[0] : 1;
  for (const t of yTicks) {
    svg.appendChild(el("line", { x1: L, x2: W - R, y1: Y(t), y2: Y(t), class: "axis" }));
    svg.appendChild(el("text", { x: L - 5, y: Y(t) + 3, "text-anchor": "end" }, tickFmt(t, yStep)));
  }
  for (const t of niceTicks(x0, x1, 6).filter(Number.isInteger)) {
    svg.appendChild(el("text", { x: X(t), y: H - 7, "text-anchor": "middle" }, String(t)));
  }

  series.forEach((s, i) => {
    const color = s.color || PALETTE[i % PALETTE.length];
    const path = s.points
      .map((p, j) => `${j ? "L" : "M"}${X(p.step).toFixed(1)},${Y(p.value).toFixed(1)}`)
      .join("");
    svg.appendChild(el("path", { d: path, fill: "none", stroke: color, "stroke-width": 1.8 }));
    for (const p of s.points) {
      const dot = el("circle", { cx: X(p.step), cy: Y(p.value), r: 3, fill: color });
      dot.appendChild(el("title", {},
        `${s.name} · step ${p.step} = ${fmt(p.value)} (n=${p.n ?? "?"})`));
      if (onPoint) dot.addEventListener("click", () => onPoint(p.step));
      svg.appendChild(dot);
    }
  });
  return svg;
}

export function chartLegend(series) {
  const div = document.createElement("div");
  div.className = "legend";
  series.forEach((s, i) => {
    const span = document.createElement("span");
    span.style.marginRight = "12px";
    span.innerHTML =
      `<span style="color:${s.color || PALETTE[i % PALETTE.length]}">●</span> ${s.name}`;
    div.appendChild(span);
  });
  return div;
}
