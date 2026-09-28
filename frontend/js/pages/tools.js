/* Educational financial calculators. Inputs describe hypothetical scenarios. */
function renderTools() {
  document.getElementById("view-tools").innerHTML = `
    <div class="tools-tabs">
      <button class="tool-tab active" data-tool="compound">Compound Interest</button>
      <button class="tool-tab" data-tool="kelly">Kelly Criterion</button>
      <button class="tool-tab" data-tool="rr">Risk / Reward</button>
      <button class="tool-tab" data-tool="mc">Monte Carlo</button>
      <button class="tool-tab" data-tool="frontier">Efficient Frontier</button>
      
      

    </div>

    <div id="tool-compound"  class="tool-panel active">${buildCompound()}</div>
    <div id="tool-kelly"     class="tool-panel">${buildKelly()}</div>
    <div id="tool-rr"        class="tool-panel">${buildRR()}</div>
    <div id="tool-mc"        class="tool-panel">${buildMC()}</div>
    <div id="tool-frontier"  class="tool-panel">${buildFrontier()}</div>
    
    
    `;

  bindTabSwitching();
  bindLiveInputs();

  /* Initialise all calculators so the first tab shows a result. */
  calcCompound();
  calcKelly();
  calcRR();
  calcMC();
}

/* Tab switching */
function bindTabSwitching() {
  document.querySelectorAll(".tool-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document
        .querySelectorAll(".tool-tab")
        .forEach((t) => t.classList.remove("active"));
      document
        .querySelectorAll(".tool-panel")
        .forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      document
        .getElementById("tool-" + tab.dataset.tool)
        ?.classList.add("active");
      /* Initialise charts once their panel is visible. */
      if (tab.dataset.tool === "frontier") calcFrontier();
    });
  });
}

/* Delegated handler for range inputs */
function bindLiveInputs() {
  document.getElementById("view-tools").oninput = (e) => {
    const id = e.target.id;
    if (id?.startsWith("ci-")) calcCompound();
    if (id?.startsWith("kl-")) calcKelly();
    if (id?.startsWith("rr-")) calcRR();
    if (id?.startsWith("mc-")) calcMC();
    if (id?.startsWith("ef-")) calcFrontier();
  };
  document.getElementById("mc-run")?.addEventListener("click", calcMC);
  document.getElementById("ef-run")?.addEventListener("click", calcFrontier);
}

/* Tab 1: compound versus simple interest. */
// Render the compound interest calculator.
function buildCompound() {
  return `<div class="tool-grid">
    <div class="tool-form-card">
      <label class="field-label">Initial investment ($)</label>
      <input class="field-input" id="ci-principal" type="number" value="10000" />
      <label class="field-label">Monthly contribution ($)</label>
      <input class="field-input" id="ci-monthly" type="number" value="500" />
      <label class="field-label">Annual return (%)</label>
      <input class="field-range" id="ci-rate" type="range" min="1" max="30" value="10" step="0.5" />
      <div class="range-labels"><span>1%</span><span id="ci-rate-lbl">10%</span><span>30%</span></div>
      <label class="field-label">Time horizon (years)</label>
      <input class="field-range" id="ci-years" type="range" min="1" max="40" value="20" />
      <div class="range-labels"><span>1 yr</span><span id="ci-years-lbl">20 yrs</span><span>40 yrs</span></div>
      <div class="tool-result">
        <div class="result-label">Final portfolio value</div>
        <div class="result-value" id="ci-result">-</div>
        <div class="result-sub" id="ci-gain">-</div>
      </div>
      <div class="result-grid">
        <div class="result-cell"><div class="result-cell-label">Total invested</div><div class="result-cell-val" id="ci-invested">-</div></div>
        <div class="result-cell"><div class="result-cell-label">Interest earned</div><div class="result-cell-val pos" id="ci-interest">-</div></div>
      </div>
    </div>
    <div class="tool-chart-card">
      <div class="chart-title" style="margin-bottom:12px">Wealth over time</div>
      <div style="position:relative;height:320px"><canvas id="chart-compound"></canvas></div>
      <div style="font-size:11px;color:var(--txt-3);margin-top:10px">
        Demonstrates the exponential effect of compounding - each year's gain
        earns its own return in subsequent years.
      </div>
    </div>
  </div>`;
}

function calcCompound() {
  const P = toolNumber("ci-principal", 10000);
  const M = toolNumber("ci-monthly", 500);
  const rate = toolNumber("ci-rate", 10);
  const years = toolNumber("ci-years", 20);
  const r = rate / 100 / 12;

  setText("ci-rate-lbl", rate + "%");
  setText("ci-years-lbl", years + " yrs");

  const labels = ["Year 0"];
  const compound = [P];
  const flat = [P];
  let val = P;

  for (let y = 1; y <= years; y++) {
    for (let m = 0; m < 12; m++) val = (val + M) * (1 + r);
    compound.push(Math.round(val));
    flat.push(Math.round(P + M * 12 * y));
    labels.push(`Yr ${y}`);
  }

  const invested = P + M * 12 * years;
  const interest = Math.round(val - invested);

  setText("ci-result", "$" + Math.round(val).toLocaleString());
  setText(
    "ci-gain",
    `${invested > 0 ? ((val / invested - 1) * 100).toFixed(1) + "% total gain" : "No contributions"}`,
  );
  setText("ci-invested", "$" + invested.toLocaleString());
  setText("ci-interest", "$" + interest.toLocaleString());

  compoundChart("chart-compound", labels, flat, compound);
}

/* Tab 2: Kelly criterion, f* = (bp - q) / b, with half-Kelly shown. */
// Render the Kelly criterion calculator.
function buildKelly() {
  return `<div class="tool-grid">
    <div class="tool-form-card">
      <label class="field-label">Account size ($)</label>
      <input class="field-input" id="kl-account" type="number" value="${state.budget || 10000}" />
      <label class="field-label">Win rate (%)</label>
      <input class="field-range" id="kl-winrate" type="range" min="30" max="80" value="55" step="1" />
      <div class="range-labels"><span>30%</span><span id="kl-wr-lbl">55%</span><span>80%</span></div>
      <label class="field-label">Average win ($)</label>
      <input class="field-input" id="kl-win" type="number" value="300" />
      <label class="field-label">Average loss ($)</label>
      <input class="field-input" id="kl-loss" type="number" value="150" />

      <div class="tool-result">
        <div class="result-label">Full Kelly position</div>
        <div class="result-value" id="kl-full">-</div>
        <div class="result-sub" id="kl-full-pct">-</div>
      </div>
      <div class="result-grid">
        <div class="result-cell"><div class="result-cell-label">Half-Kelly (recommended)</div><div class="result-cell-val pos" id="kl-half">-</div></div>
        <div class="result-cell"><div class="result-cell-label">Expected value per trade</div><div class="result-cell-val" id="kl-ev">-</div></div>
      </div>
      <div style="font-size:11px;color:var(--txt-3);padding:10px 0;border-top:1px solid var(--border)">
        f* = (W × p − L × q) / W  where p = win rate, q = 1 − p, W = avg win, L = avg loss.
        Half-Kelly reduces drawdown risk at the cost of ~25% growth rate.
      </div>
    </div>
    <div class="tool-chart-card">
      <div class="chart-title" style="margin-bottom:8px">Position size vs win rate</div>
      <div style="font-size:11.5px;color:var(--txt-2);margin-bottom:14px">
        How the Kelly fraction changes as win rate varies (holding R:R constant)
      </div>
      <div style="position:relative;height:300px"><canvas id="chart-kelly"></canvas></div>
    </div>
  </div>`;
}

function calcKelly() {
  const account = toolNumber("kl-account", 10000);
  const wr = toolNumber("kl-winrate", 55);
  const avgWin = toolNumber("kl-win", 300);
  const avgLoss = toolNumber("kl-loss", 150);

  setText("kl-wr-lbl", wr + "%");

  const p = wr / 100;
  const q = 1 - p;
  const b = avgWin / avgLoss; // reward-to-risk ratio
  const f = Math.max(0, (b * p - q) / b); // Kelly fraction
  const fH = f / 2;

  const fullPos = Math.round(account * f);
  const halfPos = Math.round(account * fH);
  const ev = p * avgWin - q * avgLoss;

  const evEl = document.getElementById("kl-ev");
  if (evEl) {
    evEl.textContent = `$${ev.toFixed(2)}`;
    evEl.style.color = ev >= 0 ? "var(--green)" : "var(--red)";
  }
  setText("kl-full", "$" + fullPos.toLocaleString());
  setText("kl-full-pct", `${(f * 100).toFixed(1)}% of account`);
  setText(
    "kl-half",
    "$" + halfPos.toLocaleString() + ` (${(fH * 100).toFixed(1)}%)`,
  );

  /* Kelly fraction across win rates of 30-80% */
  const wrRange = Array.from({ length: 51 }, (_, i) => 30 + i);
  const kellyVals = wrRange.map((w) => {
    const pp = w / 100;
    const qq = 1 - pp;
    return Math.max(0, (b * pp - qq) / b) * 100;
  });
  const halfVals = kellyVals.map((v) => v / 2);

  drawChart("chart-kelly", {
    type: "line",
    data: {
      labels: wrRange.map((w) => w + "%"),
      datasets: [
        {
          label: "Full Kelly %",
          data: kellyVals,
          borderColor: "#F59E0B",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4,
          fill: false,
        },
        {
          label: "Half Kelly %",
          data: halfVals,
          borderColor: "#00C896",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.4,
          fill: false,
          borderDash: [5, 3],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
          align: "end",
          labels: { boxWidth: 10, usePointStyle: true },
        },
        tooltip: {
          backgroundColor: "#0B1220",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          callbacks: {
            label: (ctx) =>
              ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#475569", maxTicksLimit: 10 },
        },
        y: {
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#475569", callback: (v) => v.toFixed(0) + "%" },
          min: 0,
        },
      },
    },
  });
}

/* Tab 3: break-even win rate and expected value for a reward-to-risk ratio. */
function buildRR() {
  return `<div class="tool-grid">
    <div class="tool-form-card">
      <label class="field-label">Entry price ($)</label>
      <input class="field-input" id="rr-entry" type="number" value="100" step="0.01" />
      <label class="field-label">Stop loss ($)</label>
      <input class="field-input" id="rr-stop" type="number" value="95" step="0.01" />
      <label class="field-label">Target price ($)</label>
      <input class="field-input" id="rr-target" type="number" value="115" step="0.01" />
      <label class="field-label">Actual win rate (%)</label>
      <input class="field-range" id="rr-winrate" type="range" min="20" max="90" value="50" />
      <div class="range-labels"><span>20%</span><span id="rr-wr-lbl">50%</span><span>90%</span></div>

      <div class="tool-result">
        <div class="result-label">Risk-to-reward ratio</div>
        <div class="result-value" id="rr-ratio">-</div>
        <div class="result-sub" id="rr-bewin">-</div>
      </div>
      <div class="result-grid">
        <div class="result-cell"><div class="result-cell-label">Risk ($)</div><div class="result-cell-val neg" id="rr-risk">-</div></div>
        <div class="result-cell"><div class="result-cell-label">Reward ($)</div><div class="result-cell-val pos" id="rr-reward">-</div></div>
        <div class="result-cell"><div class="result-cell-label">Expected value</div><div class="result-cell-val" id="rr-ev">-</div></div>
        <div class="result-cell"><div class="result-cell-label">Verdict</div><div class="result-cell-val" id="rr-verdict">-</div></div>
      </div>
    </div>
    <div class="tool-chart-card">
      <div class="chart-title" style="margin-bottom:8px">Expected value vs win rate</div>
      <div style="font-size:11.5px;color:var(--txt-2);margin-bottom:14px">
        The shaded region shows where a trade with this R:R has positive expected value
      </div>
      <div style="position:relative;height:300px"><canvas id="chart-rr"></canvas></div>
    </div>
  </div>`;
}

function calcRR() {
  const entry = toolNumber("rr-entry", 100);
  const stop = toolNumber("rr-stop", 95);
  const target = toolNumber("rr-target", 115);
  const wr = toolNumber("rr-winrate", 50);
  setText("rr-wr-lbl", wr + "%");

  const risk = Math.abs(entry - stop);
  const reward = Math.abs(target - entry);
  const rr = risk > 0 ? reward / risk : 0;
  const beWin = risk > 0 ? 1 / (1 + rr) : 0;
  const p = wr / 100;
  const ev = p * reward - (1 - p) * risk;

  setText("rr-ratio", rr.toFixed(2) + ":1");
  setText("rr-bewin", `Break-even win rate: ${(beWin * 100).toFixed(1)}%`);
  setText("rr-risk", "-$" + risk.toFixed(2));
  setText("rr-reward", "+$" + reward.toFixed(2));

  const evEl = document.getElementById("rr-ev");
  if (evEl) {
    evEl.textContent = (ev >= 0 ? "+$" : "-$") + Math.abs(ev).toFixed(2);
    evEl.style.color = ev >= 0 ? "var(--green)" : "var(--red)";
  }
  const vEl = document.getElementById("rr-verdict");
  if (vEl) {
    vEl.textContent =
      ev >= 0 ? (p > beWin ? "Take trade " : "Marginal") : "Skip ";
    vEl.style.color = ev >= 0 ? "var(--green)" : "var(--red)";
  }

  /* Expected value across win rates */
  const wrs = Array.from({ length: 71 }, (_, i) => 20 + i);
  const evs = wrs.map((w) => (w / 100) * reward - (1 - w / 100) * risk);
  const zeroCross = wrs.findIndex(
    (w) => (w / 100) * reward - (1 - w / 100) * risk >= 0,
  );

  drawChart("chart-rr", {
    type: "line",
    data: {
      labels: wrs.map((w) => w + "%"),
      datasets: [
        {
          label: "EV ($)",
          data: evs,
          borderColor: "#3B82F6",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.2,
          fill: true,
          backgroundColor: (ctx) => {
            const g = ctx.chart.ctx.createLinearGradient(0, 0, 0, 300);
            g.addColorStop(0, "rgba(0,200,150,0.15)");
            g.addColorStop(1, "rgba(244,63,94,0.10)");
            return g;
          },
        },
        {
          label: "Break-even",
          data: wrs.map(() => 0),
          borderColor: "rgba(255,255,255,0.2)",
          borderWidth: 1,
          borderDash: [4, 3],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0B1220",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          callbacks: {
            label: (ctx) =>
              ` EV: ${ctx.parsed.y >= 0 ? "+" : ""}$${ctx.parsed.y.toFixed(2)}`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: { color: "#475569", maxTicksLimit: 10 },
        },
        y: {
          grid: { color: "rgba(255,255,255,0.05)" },
          ticks: {
            color: "#475569",
            callback: (v) => (v >= 0 ? "+$" : "-$") + Math.abs(v).toFixed(0),
          },
        },
      },
    },
  });
}

/* Tab 4: Monte Carlo simulation of 1,000 geometric Brownian motion paths. */
function buildMC() {
  return `<div class="tool-grid">
    <div class="tool-form-card">
      <label class="field-label">Starting capital ($)</label>
      <input class="field-input" id="mc-capital" type="number" value="${state.budget || 10000}" />
      <label class="field-label">Expected annual return (%)</label>
      <input class="field-range" id="mc-return" type="range" min="-10" max="40" value="10" step="0.5" />
      <div class="range-labels"><span>-10%</span><span id="mc-ret-lbl">10%</span><span>40%</span></div>
      <label class="field-label">Annual volatility (%)</label>
      <input class="field-range" id="mc-vol" type="range" min="5" max="60" value="20" step="1" />
      <div class="range-labels"><span>5%</span><span id="mc-vol-lbl">20%</span><span>60%</span></div>
      <label class="field-label">Time horizon (years)</label>
      <input class="field-range" id="mc-years" type="range" min="1" max="30" value="10" />
      <div class="range-labels"><span>1 yr</span><span id="mc-yr-lbl">10 yrs</span><span>30 yrs</span></div>

      <div class="tool-result">
        <div class="result-label">Median final value (50th pct)</div>
        <div class="result-value" id="mc-median">-</div>
      </div>
      <div class="result-grid">
        <div class="result-cell"><div class="result-cell-label">10th pct (pessimistic)</div><div class="result-cell-val neg" id="mc-p10">-</div></div>
        <div class="result-cell"><div class="result-cell-label">90th pct (optimistic)</div><div class="result-cell-val pos" id="mc-p90">-</div></div>
        <div class="result-cell"><div class="result-cell-label">Prob. of loss</div><div class="result-cell-val" id="mc-ploss">-</div></div>
        <div class="result-cell"><div class="result-cell-label">Prob. of doubling</div><div class="result-cell-val pos" id="mc-pdouble">-</div></div>
      </div>
      <button class="btn-secondary" id="mc-run" style="width:100%;margin-top:4px">Re-run simulation</button>
    </div>
    <div class="tool-chart-card">
      <div class="chart-title" style="margin-bottom:4px">Fan chart - 1,000 simulated paths</div>
      <div style="font-size:11px;color:var(--txt-3);margin-bottom:12px">10th / 50th / 90th percentile bands shown</div>
      <div style="position:relative;height:200px;margin-bottom:14px"><canvas id="chart-mc-fan"></canvas></div>
      <div class="chart-title" style="margin-bottom:4px">Final value distribution</div>
      <div style="font-size:11px;color:var(--txt-3);margin-bottom:12px">Histogram of terminal portfolio values across 1,000 paths</div>
      <div style="position:relative;height:140px"><canvas id="chart-mc-hist"></canvas></div>
    </div>
  </div>`;
}

function calcMC() {
  const capital = toolNumber("mc-capital", 10000);
  const annRet = toolNumber("mc-return", 10);
  const annVol = toolNumber("mc-vol", 20);
  const years = toolNumber("mc-years", 10);

  setText("mc-ret-lbl", annRet + "%");
  setText("mc-vol-lbl", annVol + "%");
  setText("mc-yr-lbl", years + " yrs");

  const N = 1000;
  const days = years * 252;
  const mu = annRet / 100 / 252;
  const sig = annVol / 100 / Math.sqrt(252);

  const paths = [];
  const finals = [];

  /* GBM: dS = S(μ dt + σ dt ε)  where ε ~ N(0,1) */
  for (let n = 0; n < N; n++) {
    let val = capital;
    const trail = [val];
    for (let d = 0; d < days; d++) {
      val *= Math.exp(mu - 0.5 * sig * sig + sig * randn());
      trail.push(val);
    }
    paths.push(trail);
    finals.push(val);
  }

  /* Percentile fan at regular intervals */
  const step = Math.max(1, Math.floor(days / 80));
  const labels = [];
  const p10 = [],
    p50 = [],
    p90 = [];

  for (let i = 0; i <= days; i += step) {
    const col = paths.map((p) => p[i]).sort((a, b) => a - b);
    p10.push(Math.round(col[Math.floor(N * 0.1)]));
    p50.push(Math.round(col[Math.floor(N * 0.5)]));
    p90.push(Math.round(col[Math.floor(N * 0.9)]));
    if (i === 0) labels.push("Start");
    else if (i >= days) labels.push(`Yr ${years}`);
    else labels.push(i % (252 * 2) === 0 ? `Yr ${Math.round(i / 252)}` : "");
  }

  /* Summary statistics */
  const sorted = [...finals].sort((a, b) => a - b);
  const med = sorted[Math.floor(N * 0.5)];
  const p10v = sorted[Math.floor(N * 0.1)];
  const p90v = sorted[Math.floor(N * 0.9)];
  const ploss = finals.filter((v) => v < capital).length / N;
  const pdbl = finals.filter((v) => v >= capital * 2).length / N;

  setText("mc-median", "$" + Math.round(med).toLocaleString());
  setText("mc-p10", "$" + Math.round(p10v).toLocaleString());
  setText("mc-p90", "$" + Math.round(p90v).toLocaleString());
  setText("mc-ploss", (ploss * 100).toFixed(1) + "%");
  setText("mc-pdouble", (pdbl * 100).toFixed(1) + "%");

  monteCarloChart("chart-mc-fan", labels, { p10, p50, p90 });

  /* Histogram of terminal values in 20 bins */
  const minV = Math.min(...finals);
  const maxV = Math.max(...finals);
  const bins = 20;
  const bw = (maxV - minV) / bins;
  const counts = Array(bins).fill(0);
  const midpoints = Array.from(
    { length: bins },
    (_, i) => minV + (i + 0.5) * bw,
  );
  finals.forEach((v) => {
    const b = Math.min(bins - 1, Math.floor((v - minV) / bw));
    counts[b]++;
  });
  histogramChart(
    "chart-mc-hist",
    midpoints.map((m) => Math.round(m)),
    counts,
  );
}

/* Tab 5: two-asset efficient frontier with the maximum Sharpe portfolio. */
function buildFrontier() {
  return `<div class="tool-grid">
    <div class="tool-form-card">
      <div class="section-title" style="margin-bottom:8px">Asset A</div>
      <label class="field-label">Expected annual return (%)</label>
      <input class="field-range" id="ef-ra" type="range" min="2" max="40" value="12" step="0.5" />
      <div class="range-labels"><span>2%</span><span id="ef-ra-lbl">12%</span><span>40%</span></div>
      <label class="field-label">Annual volatility (%)</label>
      <input class="field-range" id="ef-va" type="range" min="5" max="60" value="18" step="1" />
      <div class="range-labels"><span>5%</span><span id="ef-va-lbl">18%</span><span>60%</span></div>

      <div class="section-title" style="margin-bottom:8px;margin-top:4px">Asset B</div>
      <label class="field-label">Expected annual return (%)</label>
      <input class="field-range" id="ef-rb" type="range" min="2" max="40" value="8" step="0.5" />
      <div class="range-labels"><span>2%</span><span id="ef-rb-lbl">8%</span><span>40%</span></div>
      <label class="field-label">Annual volatility (%)</label>
      <input class="field-range" id="ef-vb" type="range" min="5" max="60" value="10" step="1" />
      <div class="range-labels"><span>5%</span><span id="ef-vb-lbl">10%</span><span>60%</span></div>

      <label class="field-label">Correlation between assets</label>
      <input class="field-range" id="ef-corr" type="range" min="-100" max="100" value="30" step="5" />
      <div class="range-labels"><span>-1.0</span><span id="ef-corr-lbl">0.30</span><span>+1.0</span></div>

      <div class="tool-result" style="margin-top:4px">
        <div class="result-label">Max Sharpe portfolio</div>
        <div class="result-value" id="ef-sharpe-val">-</div>
        <div class="result-sub" id="ef-weights">-</div>
      </div>
      <button class="btn-secondary" id="ef-run" style="width:100%;margin-top:4px">Recalculate frontier</button>
    </div>
    <div class="tool-chart-card">
      <div class="chart-title" style="margin-bottom:4px">Efficient Frontier</div>
      <div style="font-size:11px;color:var(--txt-3);margin-bottom:12px">
        Each point is a random portfolio weight combination. The star marks the
        maximum Sharpe ratio (Markowitz, 1952; Sharpe, 1966).
      </div>
      <div style="position:relative;height:340px"><canvas id="chart-frontier"></canvas></div>
    </div>
  </div>`;
}

function calcFrontier() {
  const Ra = toolNumber("ef-ra", 12);
  const Va = toolNumber("ef-va", 18);
  const Rb = toolNumber("ef-rb", 8);
  const Vb = toolNumber("ef-vb", 10);
  const corr = toolNumber("ef-corr", 30);

  setText("ef-ra-lbl", Ra + "%");
  setText("ef-va-lbl", Va + "%");
  setText("ef-rb-lbl", Rb + "%");
  setText("ef-vb-lbl", Vb + "%");
  setText("ef-corr-lbl", (corr / 100).toFixed(2));

  const ra = Ra / 100;
  const va = Va / 100;
  const rb = Rb / 100;
  const vb = Vb / 100;
  const rho = corr / 100;
  const rf = 0.02; // 2% risk-free rate

  /* Risk and return for 501 evenly spaced weights */
  const points = [];
  let bestSharpe = -Infinity;
  let bestPoint = null;
  let bestWa = 0;

  for (let i = 0; i <= 500; i++) {
    const wa = i / 500;
    const wb = 1 - wa;
    const ret = wa * ra + wb * rb;
    /* portfolio variance: σ²_p = w_a²σ_a² + w_b²σ_b² + 2w_a w_b ρ σ_a σ_b */
    const vol = Math.sqrt(
      wa * wa * va * va + wb * wb * vb * vb + 2 * wa * wb * rho * va * vb,
    );
    const sharpe = (ret - rf) / (vol + 1e-9);
    points.push({ x: vol, y: ret, sharpe });
    if (sharpe > bestSharpe) {
      bestSharpe = sharpe;
      bestPoint = { x: vol, y: ret };
      bestWa = wa;
    }
  }

  setText("ef-sharpe-val", `Sharpe ${bestSharpe.toFixed(2)}`);
  setText(
    "ef-weights",
    `A: ${(bestWa * 100).toFixed(0)}%  ·  B: ${((1 - bestWa) * 100).toFixed(0)}%`,
  );

  efficientFrontierChart("chart-frontier", points, bestPoint);
}

function toolNumber(id, fallback) {
  const text = document.getElementById(id)?.value;
  const value = Number(text);
  return text !== "" && text != null && Number.isFinite(value)
    ? value
    : fallback;
}

/* Box-Muller transform for the illustrative simulation. */
function randn() {
  let u = 0;
  while (u === 0) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}
