/* ============================================================
   inve-STEM — client script
   ============================================================ */

// Presets for quick picks
const PRESETS = {
  "sp-dow":     [["^GSPC", 0.5], ["^DJI", 0.5]],
  "tech":       [["AAPL", 0.25], ["MSFT", 0.25], ["GOOGL", 0.25], ["NVDA", 0.25]],
  "defensive":  [["JNJ", 0.34], ["PG", 0.33], ["KO", 0.33]],
  "global":     [["^GSPC", 0.4], ["^FTSE", 0.3], ["^N225", 0.3]],
};

// ------- Session clock -------
function tick() {
  const d = new Date();
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  const el = document.getElementById("session-time");
  if (el) el.textContent = `${hh}:${mm}:${ss}`;
}
setInterval(tick, 1000);
tick();

// ------- Assets (tickers + weights) -------
const assetsEl = document.getElementById("assets");

function makeAssetRow(ticker = "", weight = "") {
  const row = document.createElement("div");
  row.className = "asset-row";
  row.innerHTML = `
    <input type="text" class="ticker" placeholder="TICKER" value="${ticker}"
           autocomplete="off" spellcheck="false">
    <input type="number" class="weight" placeholder="weight"
           value="${weight}" min="0" max="1" step="0.01">
    <button type="button" class="remove-btn" title="Remove">×</button>
  `;
  row.querySelector(".remove-btn").addEventListener("click", () => {
    if (assetsEl.children.length > 1) row.remove();
  });
  return row;
}

function setAssets(pairs) {
  assetsEl.innerHTML = "";
  pairs.forEach(([t, w]) => assetsEl.appendChild(makeAssetRow(t, w)));
}

// Default pair: S&P + Dow (same as the default CONFIG in simulation.py)
setAssets([["^GSPC", 0.5], ["^DJI", 0.5]]);

document.getElementById("add-asset").addEventListener("click", () => {
  assetsEl.appendChild(makeAssetRow("", ""));
});

document.getElementById("normalize").addEventListener("click", () => {
  const rows = [...assetsEl.querySelectorAll(".asset-row")];
  const weights = rows.map(r => parseFloat(r.querySelector(".weight").value) || 0);
  const total = weights.reduce((a, b) => a + b, 0);
  if (total <= 0) {
    const eq = (1 / rows.length).toFixed(4);
    rows.forEach(r => r.querySelector(".weight").value = eq);
  } else {
    rows.forEach((r, i) => {
      r.querySelector(".weight").value = (weights[i] / total).toFixed(4);
    });
  }
});

// Preset chips
document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    const preset = PRESETS[chip.dataset.preset];
    if (preset) setAssets(preset);
  });
});

// ------- Paths slider readout -------
const pathsInput = document.getElementById("paths");
const pathsReadout = document.getElementById("paths-readout");
function updatePathsReadout() {
  const v = parseInt(pathsInput.value, 10);
  pathsReadout.textContent = v.toLocaleString();
}
pathsInput.addEventListener("input", updatePathsReadout);
updatePathsReadout();

// ------- State switching -------
const stateIdle    = document.getElementById("state-idle");
const stateError   = document.getElementById("state-error");
const stateResults = document.getElementById("state-results");

function showOnly(which) {
  [stateIdle, stateError, stateResults].forEach(el => el.hidden = true);
  which.hidden = false;
}

// ------- Run button -------
const runBtn = document.getElementById("run-btn");

runBtn.addEventListener("click", async () => {
  // Collect form values
  const rows = [...assetsEl.querySelectorAll(".asset-row")];
  const tickers = [];
  const weights = [];
  for (const r of rows) {
    const t = r.querySelector(".ticker").value.trim();
    const w = parseFloat(r.querySelector(".weight").value);
    if (!t) continue;
    if (isNaN(w) || w < 0) {
      showError(`Invalid weight for ${t}. Weights must be non-negative numbers.`);
      return;
    }
    tickers.push(t);
    weights.push(w);
  }
  if (tickers.length === 0) {
    showError("Add at least one ticker to the portfolio.");
    return;
  }
  if (weights.reduce((a, b) => a + b, 0) <= 0) {
    showError("At least one weight must be greater than zero.");
    return;
  }

  const payload = {
    tickers, weights,
    start_date: document.getElementById("start_date").value,
    end_date:   document.getElementById("end_date").value,
    paths:      parseInt(document.getElementById("paths").value, 10),
    df_tails:   parseInt(document.getElementById("df_tails").value, 10),
    vol_window: parseInt(document.getElementById("vol_window").value, 10),
    return_type: document.getElementById("return_type").value,
    max_daily_return: document.getElementById("max_daily_return").value,
  };

  runBtn.disabled = true;
  const runLabel = runBtn.querySelector(".btn-run-label");
  runLabel.textContent = "Running simulation…";

  try {
    const res = await fetch("/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      showError(data.error || "Unknown server error.");
      return;
    }
    renderResults(data);
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    runBtn.disabled = false;
    runLabel.textContent = "Execute Simulation";
  }
});

function showError(msg) {
  document.getElementById("error-text").textContent = msg;
  showOnly(stateError);
}

// ------- Dashboard -------
const C = {
  cream: "#ede6d6", dim: "#a9a396", faint: "#6c6860", line: "#2a2a32",
  amber: "#d99a3f", amberHot: "#f2b44a", red: "#d8634c", green: "#6aa36a", violet: "#9684c5",
};
const FONT = { family: "JetBrains Mono, ui-monospace, monospace", size: 11, color: C.dim };
const PLOT_CONFIG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
  toImageButtonOptions: { format: "png", scale: 2, filename: "investem_chart" },
};

let R = null;
const view = { day: 1, conf: 95, units: "ret", sample: true, actual: true };

const horizonEl = document.getElementById("horizon");
const confEl = document.getElementById("confidence");

function fmtPct(v) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

function baseLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: FONT,
    margin: { l: 58, r: 18, t: 10, b: 44 },
    hovermode: "x unified",
    hoverlabel: { bgcolor: "#1c1c22", bordercolor: C.line, font: { family: FONT.family, size: 11, color: C.cream } },
    legend: { orientation: "h", x: 0, y: 1.08, font: { size: 10, color: C.dim } },
    xaxis: { gridcolor: C.line, zerolinecolor: C.line, linecolor: C.line, tickfont: { size: 10 } },
    yaxis: { gridcolor: C.line, zerolinecolor: C.faint, linecolor: C.line, tickfont: { size: 10 } },
  }, extra);
}

// Quantile at a given level (0.5 ... 99.5) for every day.
function q(level) {
  const i = Math.min(R.levels.length - 1, Math.max(0, Math.round(level * 2) - 1));
  return R.quantiles[i];
}

// CVaR approximated as the mean of the quantiles inside the lower tail.
function cvarSeries(tail) {
  const n = Math.max(1, Math.round(tail * 2));
  const out = new Array(R.n_days).fill(0);
  for (let k = 0; k < n; k++) {
    const row = R.quantiles[k];
    for (let d = 0; d < R.n_days; d++) out[d] += row[d] / n;
  }
  return out;
}

// The real series can be a few days shorter than the simulation (market
// holidays), so use the last observed value on or before the chosen day.
function actualAt(d) {
  if (!R.actual) return null;
  for (let i = Math.min(d, R.actual.length - 1); i >= 0; i--) {
    if (R.actual[i] !== null && R.actual[i] !== undefined) return { v: R.actual[i], i };
  }
  return null;
}

function toUnits(arr) {
  if (view.units === "ret") return arr;
  return arr.map(v => (v === null ? null : R.s0 * (1 + v / 100)));
}

function drawPaths() {
  const x = R.dates;
  const tail = 100 - view.conf;
  const lo = toUnits(q(tail)), hi = toUnits(q(view.conf));
  const q25 = toUnits(q(25)), q75 = toUnits(q(75));
  const traces = [];

  if (view.sample) {
    const sx = [], sy = [];
    R.sample.forEach(path => {
      const p = toUnits(path);
      for (let d = 0; d < p.length; d++) { sx.push(x[d]); sy.push(p[d]); }
      sx.push(null); sy.push(null);
    });
    traces.push({ x: sx, y: sy, mode: "lines", name: "sample paths",
      line: { color: "rgba(169,163,150,0.16)", width: 1 }, hoverinfo: "skip" });
  }

  traces.push(
    { x, y: lo, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x, y: hi, mode: "lines", fill: "tonexty", fillcolor: "rgba(217,154,63,0.13)",
      line: { width: 0 }, name: `${tail.toFixed(1)}–${view.conf.toFixed(1)}% band`, hoverinfo: "skip" },
    { x, y: q25, mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x, y: q75, mode: "lines", fill: "tonexty", fillcolor: "rgba(217,154,63,0.26)",
      line: { width: 0 }, name: "25–75% band", hoverinfo: "skip" },
    { x, y: lo, mode: "lines", name: `VaR ${view.conf}%`, line: { color: C.amber, width: 1, dash: "dot" } },
    { x, y: toUnits(q(50)), mode: "lines", name: "median", line: { color: C.amberHot, width: 2 } },
    { x, y: toUnits(R.mean), mode: "lines", name: "mean", line: { color: C.green, width: 1.5, dash: "dash" } },
  );

  if (view.actual && R.actual) {
    traces.push({ x: x.slice(0, R.actual.length), y: toUnits(R.actual), mode: "lines",
      name: "actual", line: { color: C.red, width: 2.2 } });
  }

  const layout = baseLayout({
    yaxis: Object.assign(baseLayout().yaxis, {
      title: { text: view.units === "ret" ? "return since start (%)" : "portfolio value", font: { size: 10 } },
      ticksuffix: view.units === "ret" ? "%" : "",
    }),
  });
  Plotly.react("chart-paths", traces, layout, PLOT_CONFIG).then(() => moveMarker("chart-paths"));
}

function drawDist() {
  const d = view.day;
  const counts = R.hist.counts[d];
  const lo = R.hist.lo[d], hi = R.hist.hi[d];
  const w = (hi - lo) / counts.length;
  const centers = counts.map((_, i) => lo + (i + 0.5) * w);
  const varD = q(100 - view.conf)[d];
  const colors = centers.map(c => (c <= varD ? "rgba(216,99,76,0.85)" : "rgba(217,154,63,0.55)"));

  const lines = [
    { x: varD, color: C.red, label: `VaR ${fmtPct(varD)}` },
    { x: R.mean[d], color: C.green, label: `mean ${fmtPct(R.mean[d])}` },
  ];
  const a = view.actual ? actualAt(d) : null;
  if (a) lines.push({ x: a.v, color: C.cream, label: `actual ${fmtPct(a.v)}` });

  const layout = baseLayout({
    hovermode: "closest",
    bargap: 0.04,
    showlegend: false,
    xaxis: Object.assign(baseLayout().xaxis, { ticksuffix: "%", title: { text: "return since start", font: { size: 10 } } }),
    yaxis: Object.assign(baseLayout().yaxis, { title: { text: "paths", font: { size: 10 } } }),
    shapes: lines.map(l => ({ type: "line", x0: l.x, x1: l.x, yref: "paper", y0: 0, y1: 1,
      line: { color: l.color, width: 1.5, dash: "dash" } })),
    annotations: lines.map((l, i) => ({ x: l.x, yref: "paper", y: 1 - i * 0.09, text: l.label,
      showarrow: false, xanchor: "left", xshift: 4, bgcolor: "rgba(22,22,27,0.9)", borderpad: 2,
      font: { size: 10, color: l.color } })),
  });
  Plotly.react("chart-dist", [{
    type: "bar", x: centers, y: counts, width: w, marker: { color: colors },
    hovertemplate: "%{x:.2f}%<br>%{y} paths<extra></extra>",
  }], layout, PLOT_CONFIG);

  document.getElementById("dist-title").textContent =
    `Return distribution on ${R.dates[d]} (day ${d})`;
}

function drawRisk() {
  const tail = 100 - view.conf;
  const x = R.dates;
  const layout = baseLayout({
    yaxis: Object.assign(baseLayout().yaxis, { ticksuffix: "%", title: { text: "return", font: { size: 10 } } }),
    yaxis2: { overlaying: "y", side: "right", range: [0, 100], dtick: 25, ticksuffix: "%", showgrid: false,
      tickfont: { size: 10, color: C.violet }, title: { text: "P(loss)", font: { size: 10, color: C.violet } } },
    margin: { l: 58, r: 52, t: 10, b: 44 },
  });
  Plotly.react("chart-risk", [
    { x, y: q(tail), mode: "lines", name: `VaR ${view.conf}%`, line: { color: C.amber, width: 2 } },
    { x, y: cvarSeries(tail), mode: "lines", name: `CVaR ${view.conf}%`, line: { color: C.red, width: 2 } },
    { x, y: R.p_loss, mode: "lines", name: "P(loss)", yaxis: "y2", line: { color: C.violet, width: 1.5, dash: "dot" } },
  ], layout, PLOT_CONFIG).then(() => moveMarker("chart-risk"));
}

function updateMetrics() {
  const d = view.day, tail = 100 - view.conf;
  document.getElementById("m-mean").textContent = fmtPct(R.mean[d]);
  document.getElementById("m-median").textContent = fmtPct(q(50)[d]);
  document.getElementById("m-var").textContent = fmtPct(q(tail)[d]);
  document.getElementById("m-cvar").textContent = fmtPct(cvarSeries(tail)[d]);
  document.getElementById("m-loss").textContent = `${R.p_loss[d].toFixed(1)}%`;
  document.getElementById("m-var-label").textContent = `VaR ${view.conf}%`;
  document.getElementById("m-cvar-label").textContent = `CVaR ${view.conf}%`;
  document.getElementById("m-var-sub").textContent = `${tail.toFixed(1)}th percentile`;

  const wrap = document.getElementById("m-actual-wrap");
  const a = actualAt(d);
  if (a) {
    document.getElementById("m-actual").textContent = fmtPct(a.v);
    document.getElementById("m-actual-sub").textContent = `observed on ${R.dates[a.i]}`;
    wrap.hidden = false;
  } else {
    wrap.hidden = true;
  }

  document.getElementById("horizon-readout").textContent = `day ${d} · ${R.dates[d]}`;
  document.getElementById("confidence-readout").textContent = `${view.conf}%`;
}

// The horizon marker is a plain element laid over the chart, so dragging the
// slider moves it without making Plotly redraw thousands of points.
function moveMarker(id) {
  const gd = document.getElementById(id);
  const fl = gd && gd._fullLayout;
  if (!fl || !fl.xaxis) return;
  let m = gd.parentElement.querySelector(`.marker[data-for="${id}"]`);
  if (!m) {
    m = document.createElement("div");
    m.className = "marker";
    m.dataset.for = id;
    gd.parentElement.appendChild(m);
  }
  const ax = fl.xaxis, size = fl._size;
  const px = ax.l2p(ax.d2l(R.dates[view.day]));
  m.hidden = px < 0 || px > size.w;
  m.style.left = `${gd.offsetLeft + size.l + px}px`;
  m.style.top = `${gd.offsetTop + size.t}px`;
  m.style.height = `${size.h}px`;
}

// Redraw at most once per frame while a slider is being dragged.
let pending = null;
function schedule(what) {
  pending = Object.assign(pending || {}, what);
  if (pending.queued) return;
  pending.queued = true;
  requestAnimationFrame(() => {
    const p = pending; pending = null;
    if (!R) return;
    updateMetrics();
    if (p.paths) drawPaths();
    if (p.dist) drawDist();
    if (p.risk) drawRisk();
    if (p.marker && !p.paths) moveMarker("chart-paths");
    if (p.marker && !p.risk) moveMarker("chart-risk");
  });
}

horizonEl.addEventListener("input", () => {
  view.day = parseInt(horizonEl.value, 10);
  schedule({ dist: true, marker: true });
});
confEl.addEventListener("input", () => {
  view.conf = parseFloat(confEl.value);
  schedule({ paths: true, dist: true, risk: true });
});
document.querySelectorAll(".seg").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll(".seg").forEach(b => b.classList.toggle("active", b === btn));
  view.units = btn.dataset.units;
  schedule({ paths: true });
}));
document.getElementById("show-sample").addEventListener("change", e => {
  view.sample = e.target.checked; schedule({ paths: true });
});
document.getElementById("show-actual").addEventListener("change", e => {
  view.actual = e.target.checked; schedule({ paths: true, dist: true });
});

window.addEventListener("resize", () => {
  if (R) requestAnimationFrame(() => { moveMarker("chart-paths"); moveMarker("chart-risk"); });
});

function renderResults(data) {
  R = data;
  view.day = R.n_days - 1;
  horizonEl.max = String(R.n_days - 1);
  horizonEl.value = String(view.day);

  const tickerSummary = data.tickers
    .map((t, i) => `${t} ${(data.weights[i] * 100).toFixed(0)}%`)
    .join(" · ");
  document.getElementById("run-summary").textContent =
    `RUN · ${data.n_paths.toLocaleString()} PATHS · ${data.n_days - 1} DAYS · ${tickerSummary}`;

  showOnly(stateResults);
  updateMetrics();
  drawPaths();
  drawDist();
  drawRisk();

  ["chart-paths", "chart-risk"].forEach(id => {
    const gd = document.getElementById(id);
    gd.removeAllListeners && gd.removeAllListeners("plotly_relayout");
    gd.on("plotly_relayout", () => moveMarker(id));
  });

  const pathsDiv = document.getElementById("chart-paths");
  pathsDiv.removeAllListeners && pathsDiv.removeAllListeners("plotly_click");
  pathsDiv.on("plotly_click", ev => {
    if (!ev.points || !ev.points.length) return;
    const i = R.dates.indexOf(String(ev.points[0].x).slice(0, 10));
    if (i > 0) {
      view.day = i;
      horizonEl.value = String(i);
      schedule({ dist: true, marker: true });
    }
  });
}
