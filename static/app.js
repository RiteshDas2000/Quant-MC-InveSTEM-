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

// Default pair: S&P + Dow (matches the user's default CONFIG)
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
    await renderResults(data, payload);
  } catch (e) {
    showError(e.message || String(e));
  } finally {
    runBtn.disabled = false;
  }
});

// Resolve when an <img> has finished loading (or errored)
function waitForImage(imgEl, src) {
  return new Promise((resolve) => {
    imgEl.onload = () => resolve();
    imgEl.onerror = () => resolve();
    imgEl.src = src;
    // If already cached and complete, resolve immediately
    if (imgEl.complete && imgEl.naturalWidth > 0) resolve();
  });
}

function showError(msg) {
  document.getElementById("error-text").textContent = msg;
  showOnly(stateError);
}

// ------- Render results -------
function fmtPct(v) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

function renderResults(data, payload) {
  const m = data.metrics;

  document.getElementById("m-mean").textContent = fmtPct(m.mean_return);
  document.getElementById("m-var").textContent  = fmtPct(m.var95);
  document.getElementById("m-cvar").textContent = fmtPct(m.cvar95);
  document.getElementById("m-loss").textContent = `${m.chance_of_loss.toFixed(1)}%`;
  document.getElementById("m-bw").textContent   =
    `${fmtPct(m.best_case)} / ${fmtPct(m.worst_case)}`;

  const actualWrap = document.getElementById("m-actual-wrap");
  if (m.actual_return !== null && m.actual_return !== undefined) {
    document.getElementById("m-actual").textContent = fmtPct(m.actual_return);
    actualWrap.hidden = false;
  } else {
    actualWrap.hidden = true;
  }

  const tickerSummary = data.tickers
    .map((t, i) => `${t} ${(data.weights[i] * 100).toFixed(0)}%`)
    .join(" · ");
  document.getElementById("run-summary").textContent =
    `RUN · ${data.n_paths.toLocaleString()} PATHS · ${data.n_days} DAYS · ${tickerSummary}`;

  // Wait for both chart images to finish loading before swapping panels.
  const imgPaths = document.getElementById("chart-paths");
  const imgDist  = document.getElementById("chart-dist");
  return Promise.all([
    waitForImage(imgPaths, data.paths_img),
    waitForImage(imgDist,  data.dist_img),
  ]).then(() => {
    showOnly(stateResults);
  });
}
