/*
 * Financial calculators — Tools page.
 *
 * Tabs:
 *   1. Compound interest   — wealth accumulation with monthly contributions
 *   2. Kelly Criterion     — mathematically optimal position sizing
 *   3. Risk / Reward       — break-even win rate and R:R visualisation
 *   4. Monte Carlo         — 1,000-path simulation with outcome histogram
 *   5. Efficient Frontier  — risk/return trade-off for a two-asset portfolio
 *   6. Model Backtest      — call the backend and show equity curves
 *
 * All calculations run client-side except the backtest which calls the API.
 */

function renderTools() {
  document.getElementById('view-tools').innerHTML = `
    <div class="tools-tabs">
      <button class="tool-tab active" data-tool="compound">Compound Interest</button>
      <button class="tool-tab" data-tool="kelly">Kelly Criterion</button>
      <button class="tool-tab" data-tool="rr">Risk / Reward</button>
      <button class="tool-tab" data-tool="mc">Monte Carlo</button>
      <button class="tool-tab" data-tool="frontier">Efficient Frontier</button>
      <button class="tool-tab" data-tool="backtest">Model Backtest</button>
      <button class="tool-tab" data-tool="validation">Validation</button>
    </div>

    <div id="tool-compound"  class="tool-panel active">${buildCompound()}</div>
    <div id="tool-kelly"     class="tool-panel">${buildKelly()}</div>
    <div id="tool-rr"        class="tool-panel">${buildRR()}</div>
    <div id="tool-mc"        class="tool-panel">${buildMC()}</div>
    <div id="tool-frontier"  class="tool-panel">${buildFrontier()}</div>
    <div id="tool-backtest"  class="tool-panel">${buildBacktest()}</div>
    <div id="tool-validation" class="tool-panel">${buildValidation()}</div>`;

  bindTabSwitching();
  bindLiveInputs();

  /* initialise all calculators so the first tab always shows a result */
  calcCompound();
  calcKelly();
  calcRR();
  calcMC();
}

/* ── Tab switching ───────────────────────────────────────────────────────── */
function bindTabSwitching() {
  document.querySelectorAll('.tool-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tool-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tool-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('tool-' + tab.dataset.tool)?.classList.add('active');
      /* lazy-initialise charts that require the panel to be visible first */
      if (tab.dataset.tool === 'frontier') calcFrontier();
    });
  });
}

/* single event delegation for all range inputs */
function bindLiveInputs() {
  document.addEventListener('input', e => {
    const id = e.target.id;
    if (id?.startsWith('ci-'))  calcCompound();
    if (id?.startsWith('kl-'))  calcKelly();
    if (id?.startsWith('rr-'))  calcRR();
    if (id?.startsWith('mc-'))  calcMC();
    if (id?.startsWith('ef-'))  calcFrontier();
  });
  document.getElementById('mc-run')?.addEventListener('click', calcMC);
  document.getElementById('bt-run')?.addEventListener('click', runBacktest);
  document.getElementById('ef-run')?.addEventListener('click', calcFrontier);
}

/* ══════════════════════════════════════════════════════════════════════════
   TAB 1 — Compound Interest
   Shows how compound interest outperforms simple interest over time.
   ══════════════════════════════════════════════════════════════════════════ */
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
        <div class="result-value" id="ci-result">—</div>
        <div class="result-sub" id="ci-gain">—</div>
      </div>
      <div class="result-grid">
        <div class="result-cell"><div class="result-cell-label">Total invested</div><div class="result-cell-val" id="ci-invested">—</div></div>
        <div class="result-cell"><div class="result-cell-label">Interest earned</div><div class="result-cell-val pos" id="ci-interest">—</div></div>
      </div>
    </div>
    <div class="tool-chart-card">
      <div class="chart-title" style="margin-bottom:12px">Wealth over time</div>
      <div style="position:relative;height:320px"><canvas id="chart-compound"></canvas></div>
      <div style="font-size:11px;color:var(--txt-3);margin-top:10px">
        Demonstrates the exponential effect of compounding — each year's gain
        earns its own return in subsequent years (Einstein's "8th wonder").
      </div>
    </div>
  </div>`;
}

function calcCompound() {
  const P     = parseFloat(document.getElementById('ci-principal')?.value) || 10000;
  const M     = parseFloat(document.getElementById('ci-monthly')?.value)   || 500;
  const rate  = parseFloat(document.getElementById('ci-rate')?.value)      || 10;
  const years = parseInt(document.getElementById('ci-years')?.value)       || 20;
  const r     = rate / 100 / 12;

  setText('ci-rate-lbl',  rate + '%');
  setText('ci-years-lbl', years + ' yrs');

  const labels   = ['Year 0'];
  const compound = [P];
  const flat     = [P];
  let val = P;

  for (let y = 1; y <= years; y++) {
    for (let m = 0; m < 12; m++) val = (val + M) * (1 + r);
    compound.push(Math.round(val));
    flat.push(Math.round(P + M * 12 * y));
    labels.push(`Yr ${y}`);
  }

  const invested = P + M * 12 * years;
  const interest = Math.round(val - invested);

  setText('ci-result',   '$' + Math.round(val).toLocaleString());
  setText('ci-gain',     `${((val/invested - 1)*100).toFixed(1)}% total gain`);
  setText('ci-invested', '$' + invested.toLocaleString());
  setText('ci-interest', '$' + interest.toLocaleString());

  compoundChart('chart-compound', labels, flat, compound);
}

/* ══════════════════════════════════════════════════════════════════════════
   TAB 2 — Kelly Criterion
   Calculates the mathematically optimal fraction of capital to risk per trade.
   Formula: f* = (bp - q) / b   where b = odds, p = win rate, q = 1 - p
   Half-Kelly (f* / 2) is the practical recommendation to reduce ruin risk.
   ══════════════════════════════════════════════════════════════════════════ */
function buildKelly() {
  return `<div class="tool-grid">
    <div class="tool-form-card">
      <label class="field-label">Account size ($)</label>
      <input class="field-input" id="kl-account" type="number" value="${state.budget||10000}" />
      <label class="field-label">Win rate (%)</label>
      <input class="field-range" id="kl-winrate" type="range" min="30" max="80" value="55" step="1" />
      <div class="range-labels"><span>30%</span><span id="kl-wr-lbl">55%</span><span>80%</span></div>
      <label class="field-label">Average win ($)</label>
      <input class="field-input" id="kl-win" type="number" value="300" />
      <label class="field-label">Average loss ($)</label>
      <input class="field-input" id="kl-loss" type="number" value="150" />

      <div class="tool-result">
        <div class="result-label">Full Kelly position</div>
        <div class="result-value" id="kl-full">—</div>
        <div class="result-sub" id="kl-full-pct">—</div>
      </div>
      <div class="result-grid">
        <div class="result-cell"><div class="result-cell-label">Half-Kelly (recommended)</div><div class="result-cell-val pos" id="kl-half">—</div></div>
        <div class="result-cell"><div class="result-cell-label">Expected value per trade</div><div class="result-cell-val" id="kl-ev">—</div></div>
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
  const account = parseFloat(document.getElementById('kl-account')?.value) || 10000;
  const wr      = parseFloat(document.getElementById('kl-winrate')?.value) || 55;
  const avgWin  = parseFloat(document.getElementById('kl-win')?.value)     || 300;
  const avgLoss = parseFloat(document.getElementById('kl-loss')?.value)    || 150;

  setText('kl-wr-lbl', wr + '%');

  const p  = wr / 100;
  const q  = 1 - p;
  const b  = avgWin / avgLoss;   // reward-to-risk ratio
  const f  = Math.max(0, (b * p - q) / b);   // Kelly fraction
  const fH = f / 2;

  const fullPos = Math.round(account * f);
  const halfPos = Math.round(account * fH);
  const ev      = p * avgWin - q * avgLoss;

  const evEl = document.getElementById('kl-ev');
  if (evEl) { evEl.textContent = `$${ev.toFixed(2)}`; evEl.style.color = ev >= 0 ? 'var(--green)' : 'var(--red)'; }
  setText('kl-full',    '$' + fullPos.toLocaleString());
  setText('kl-full-pct', `${(f*100).toFixed(1)}% of account`);
  setText('kl-half',    '$' + halfPos.toLocaleString() + ` (${(fH*100).toFixed(1)}%)`);

  /* draw a curve showing Kelly fraction across win rates 30-80% */
  const wrRange    = Array.from({length:51}, (_,i) => 30 + i);
  const kellyVals  = wrRange.map(w => {
    const pp = w/100; const qq = 1-pp;
    return Math.max(0, (b*pp - qq)/b) * 100;
  });
  const halfVals   = kellyVals.map(v => v/2);

  safeChart('chart-kelly', {
    type: 'line',
    data: {
      labels: wrRange.map(w => w + '%'),
      datasets: [
        { label: 'Full Kelly %', data: kellyVals, borderColor: '#F59E0B', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false },
        { label: 'Half Kelly %', data: halfVals,  borderColor: '#00C896', borderWidth: 2, pointRadius: 0, tension: 0.4, fill: false, borderDash: [5,3] },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: 'top', align: 'end', labels: { boxWidth: 10, usePointStyle: true } },
                 tooltip: { backgroundColor: '#0B1220', borderColor: 'rgba(255,255,255,0.1)', borderWidth:1,
                   callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%` } } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#475569', maxTicksLimit: 10 } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#475569', callback: v => v.toFixed(0)+'%' }, min: 0 },
      },
    },
  });
}

/* ══════════════════════════════════════════════════════════════════════════
   TAB 3 — Risk / Reward
   Shows the break-even win rate for a given R:R ratio and calculates
   expected value. A trade is only worth taking when EV > 0.
   ══════════════════════════════════════════════════════════════════════════ */
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
        <div class="result-value" id="rr-ratio">—</div>
        <div class="result-sub" id="rr-bewin">—</div>
      </div>
      <div class="result-grid">
        <div class="result-cell"><div class="result-cell-label">Risk ($)</div><div class="result-cell-val neg" id="rr-risk">—</div></div>
        <div class="result-cell"><div class="result-cell-label">Reward ($)</div><div class="result-cell-val pos" id="rr-reward">—</div></div>
        <div class="result-cell"><div class="result-cell-label">Expected value</div><div class="result-cell-val" id="rr-ev">—</div></div>
        <div class="result-cell"><div class="result-cell-label">Verdict</div><div class="result-cell-val" id="rr-verdict">—</div></div>
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
  const entry  = parseFloat(document.getElementById('rr-entry')?.value)   || 100;
  const stop   = parseFloat(document.getElementById('rr-stop')?.value)    || 95;
  const target = parseFloat(document.getElementById('rr-target')?.value)  || 115;
  const wr     = parseFloat(document.getElementById('rr-winrate')?.value) || 50;
  setText('rr-wr-lbl', wr + '%');

  const risk   = Math.abs(entry - stop);
  const reward = Math.abs(target - entry);
  const rr     = risk > 0 ? reward / risk : 0;
  const beWin  = risk > 0 ? 1 / (1 + rr) : 0;
  const p      = wr / 100;
  const ev     = p * reward - (1 - p) * risk;

  setText('rr-ratio',  rr.toFixed(2) + ':1');
  setText('rr-bewin',  `Break-even win rate: ${(beWin*100).toFixed(1)}%`);
  setText('rr-risk',   '-$' + risk.toFixed(2));
  setText('rr-reward', '+$' + reward.toFixed(2));

  const evEl = document.getElementById('rr-ev');
  if (evEl) { evEl.textContent = (ev >= 0 ? '+$' : '-$') + Math.abs(ev).toFixed(2); evEl.style.color = ev >= 0 ? 'var(--green)' : 'var(--red)'; }
  const vEl = document.getElementById('rr-verdict');
  if (vEl) {
    vEl.textContent = ev >= 0 ? (p > beWin ? 'Take trade ✓' : 'Marginal') : 'Skip ✗';
    vEl.style.color = ev >= 0 ? 'var(--green)' : 'var(--red)';
  }

  /* EV curve across win rates */
  const wrs  = Array.from({length:71}, (_,i) => 20 + i);
  const evs  = wrs.map(w => (w/100)*reward - (1-w/100)*risk);
  const zeroCross = wrs.findIndex(w => (w/100)*reward - (1-w/100)*risk >= 0);

  safeChart('chart-rr', {
    type: 'line',
    data: {
      labels: wrs.map(w => w + '%'),
      datasets: [
        { label: 'EV ($)', data: evs, borderColor: '#3B82F6', borderWidth: 2, pointRadius: 0,
          tension: 0.2, fill: true,
          backgroundColor: ctx => {
            const g = ctx.chart.ctx.createLinearGradient(0,0,0,300);
            g.addColorStop(0, 'rgba(0,200,150,0.15)'); g.addColorStop(1, 'rgba(244,63,94,0.10)');
            return g;
          }},
        { label: 'Break-even', data: wrs.map(() => 0), borderColor: 'rgba(255,255,255,0.2)',
          borderWidth: 1, borderDash: [4,3], pointRadius: 0, fill: false },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false },
                 tooltip: { backgroundColor: '#0B1220', borderColor: 'rgba(255,255,255,0.1)', borderWidth:1,
                   callbacks: { label: ctx => ` EV: ${ctx.parsed.y >= 0 ? '+' : ''}$${ctx.parsed.y.toFixed(2)}` } } },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#475569', maxTicksLimit: 10 } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#475569', callback: v => (v>=0?'+$':'-$')+Math.abs(v).toFixed(0) } },
      },
    },
  });
}

/* ══════════════════════════════════════════════════════════════════════════
   TAB 4 — Monte Carlo Simulation
   Runs 1,000 portfolio paths using a Geometric Brownian Motion model.
   GBM is the foundation of the Black-Scholes options model and assumes
   log-normally distributed returns — a reasonable approximation for
   index-level instruments over short horizons.
   ══════════════════════════════════════════════════════════════════════════ */
function buildMC() {
  return `<div class="tool-grid">
    <div class="tool-form-card">
      <label class="field-label">Starting capital ($)</label>
      <input class="field-input" id="mc-capital" type="number" value="${state.budget||10000}" />
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
        <div class="result-value" id="mc-median">—</div>
      </div>
      <div class="result-grid">
        <div class="result-cell"><div class="result-cell-label">10th pct (pessimistic)</div><div class="result-cell-val neg" id="mc-p10">—</div></div>
        <div class="result-cell"><div class="result-cell-label">90th pct (optimistic)</div><div class="result-cell-val pos" id="mc-p90">—</div></div>
        <div class="result-cell"><div class="result-cell-label">Prob. of loss</div><div class="result-cell-val" id="mc-ploss">—</div></div>
        <div class="result-cell"><div class="result-cell-label">Prob. of doubling</div><div class="result-cell-val pos" id="mc-pdouble">—</div></div>
      </div>
      <button class="btn-secondary" id="mc-run" style="width:100%;margin-top:4px">Re-run simulation</button>
    </div>
    <div class="tool-chart-card">
      <div class="chart-title" style="margin-bottom:4px">Fan chart — 1,000 simulated paths</div>
      <div style="font-size:11px;color:var(--txt-3);margin-bottom:12px">10th / 50th / 90th percentile bands shown</div>
      <div style="position:relative;height:200px;margin-bottom:14px"><canvas id="chart-mc-fan"></canvas></div>
      <div class="chart-title" style="margin-bottom:4px">Final value distribution</div>
      <div style="font-size:11px;color:var(--txt-3);margin-bottom:12px">Histogram of terminal portfolio values across 1,000 paths</div>
      <div style="position:relative;height:140px"><canvas id="chart-mc-hist"></canvas></div>
    </div>
  </div>`;
}

function calcMC() {
  const capital = parseFloat(document.getElementById('mc-capital')?.value) || 10000;
  const annRet  = parseFloat(document.getElementById('mc-return')?.value)  || 10;
  const annVol  = parseFloat(document.getElementById('mc-vol')?.value)     || 20;
  const years   = parseInt(document.getElementById('mc-years')?.value)     || 10;

  setText('mc-ret-lbl', annRet + '%');
  setText('mc-vol-lbl', annVol + '%');
  setText('mc-yr-lbl',  years + ' yrs');

  const N    = 1000;
  const days = years * 252;
  const mu   = annRet  / 100 / 252;
  const sig  = annVol  / 100 / Math.sqrt(252);

  const paths   = [];
  const finals  = [];

  /* GBM: dS = S(μ dt + σ √dt ε)  where ε ~ N(0,1) */
  for (let n = 0; n < N; n++) {
    let val = capital;
    const trail = [val];
    for (let d = 0; d < days; d++) {
      val *= Math.exp((mu - 0.5 * sig * sig) + sig * randn());
      trail.push(val);
    }
    paths.push(trail);
    finals.push(val);
  }

  /* build percentile fan at regular intervals */
  const step   = Math.max(1, Math.floor(days / 80));
  const labels = [];
  const p10 = [], p50 = [], p90 = [];

  for (let i = 0; i <= days; i += step) {
    const col = paths.map(p => p[i]).sort((a,b) => a-b);
    p10.push(Math.round(col[Math.floor(N*0.10)]));
    p50.push(Math.round(col[Math.floor(N*0.50)]));
    p90.push(Math.round(col[Math.floor(N*0.90)]));
    if (i === 0)    labels.push('Start');
    else if (i >= days) labels.push(`Yr ${years}`);
    else             labels.push(i % (252*2) === 0 ? `Yr ${Math.round(i/252)}` : '');
  }

  /* summary statistics */
  const sorted = [...finals].sort((a,b) => a-b);
  const med    = sorted[Math.floor(N*0.5)];
  const p10v   = sorted[Math.floor(N*0.10)];
  const p90v   = sorted[Math.floor(N*0.90)];
  const ploss  = finals.filter(v => v < capital).length / N;
  const pdbl   = finals.filter(v => v >= capital*2).length / N;

  setText('mc-median',  '$' + Math.round(med).toLocaleString());
  setText('mc-p10',     '$' + Math.round(p10v).toLocaleString());
  setText('mc-p90',     '$' + Math.round(p90v).toLocaleString());
  setText('mc-ploss',   (ploss*100).toFixed(1) + '%');
  setText('mc-pdouble', (pdbl*100).toFixed(1) + '%');

  monteCarloChart('chart-mc-fan', labels, { p10, p50, p90 });

  /* histogram — bucket terminal values into 20 bins */
  const minV = Math.min(...finals);
  const maxV = Math.max(...finals);
  const bins = 20;
  const bw   = (maxV - minV) / bins;
  const counts   = Array(bins).fill(0);
  const midpoints = Array.from({length:bins}, (_,i) => minV + (i+0.5)*bw);
  finals.forEach(v => {
    const b = Math.min(bins-1, Math.floor((v - minV) / bw));
    counts[b]++;
  });
  histogramChart('chart-mc-hist', midpoints.map(m => Math.round(m)), counts);
}

/* ══════════════════════════════════════════════════════════════════════════
   TAB 5 — Efficient Frontier (2-asset)
   Demonstrates the Markowitz (1952) mean-variance framework by generating
   random two-asset portfolio weights and plotting risk vs return.
   The maximum Sharpe ratio portfolio is highlighted as the optimal choice.
   ══════════════════════════════════════════════════════════════════════════ */
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
        <div class="result-value" id="ef-sharpe-val">—</div>
        <div class="result-sub" id="ef-weights">—</div>
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
  const Ra   = parseFloat(document.getElementById('ef-ra')?.value)   || 12;
  const Va   = parseFloat(document.getElementById('ef-va')?.value)   || 18;
  const Rb   = parseFloat(document.getElementById('ef-rb')?.value)   || 8;
  const Vb   = parseFloat(document.getElementById('ef-vb')?.value)   || 10;
  const corr = parseFloat(document.getElementById('ef-corr')?.value) || 30;

  setText('ef-ra-lbl',   Ra + '%');
  setText('ef-va-lbl',   Va + '%');
  setText('ef-rb-lbl',   Rb + '%');
  setText('ef-vb-lbl',   Vb + '%');
  setText('ef-corr-lbl', (corr / 100).toFixed(2));

  const ra   = Ra / 100;
  const va   = Va / 100;
  const rb   = Rb / 100;
  const vb   = Vb / 100;
  const rho  = corr / 100;
  const rf   = 0.02;   // 2% risk-free rate

  /* generate 500 random weight combinations and compute portfolio risk/return */
  const points = [];
  let   bestSharpe = -Infinity;
  let   bestPoint  = null;
  let   bestWa     = 0;

  for (let i = 0; i <= 500; i++) {
    const wa = i / 500;
    const wb = 1 - wa;
    const ret = wa * ra + wb * rb;
    /* portfolio variance: σ²_p = w_a²σ_a² + w_b²σ_b² + 2w_a w_b ρ σ_a σ_b */
    const vol = Math.sqrt(wa*wa*va*va + wb*wb*vb*vb + 2*wa*wb*rho*va*vb);
    const sharpe = (ret - rf) / (vol + 1e-9);
    points.push({ x: vol, y: ret, sharpe });
    if (sharpe > bestSharpe) { bestSharpe = sharpe; bestPoint = { x: vol, y: ret }; bestWa = wa; }
  }

  setText('ef-sharpe-val', `Sharpe ${bestSharpe.toFixed(2)}`);
  setText('ef-weights', `A: ${(bestWa*100).toFixed(0)}%  ·  B: ${((1-bestWa)*100).toFixed(0)}%`);

  efficientFrontierChart('chart-frontier', points, bestPoint);
}

/* ══════════════════════════════════════════════════════════════════════════
   TAB 6 — Model Backtest
   Calls the backend /api/backtest endpoint and draws the equity curve.
   ══════════════════════════════════════════════════════════════════════════ */
function buildBacktest() {
  return `<div class="tool-grid">
    <div class="tool-form-card">
      <label class="field-label">Ticker symbol</label>
      <input class="field-input" id="bt-ticker" type="text" value="AAPL" placeholder="AAPL" style="text-transform:uppercase" />
      <label class="field-label">Starting capital ($)</label>
      <input class="field-input" id="bt-capital" type="number" value="${state.budget||10000}" />

      <div id="bt-metrics" style="display:none">
        <div class="tool-result">
          <div class="result-label">Strategy total return</div>
          <div class="result-value" id="bt-total-ret">—</div>
          <div class="result-sub" id="bt-vs-bench">—</div>
        </div>
        <div class="result-grid">
          <div class="result-cell"><div class="result-cell-label">Sharpe ratio</div><div class="result-cell-val" id="bt-sharpe">—</div></div>
          <div class="result-cell"><div class="result-cell-label">Max drawdown</div><div class="result-cell-val neg" id="bt-dd">—</div></div>
          <div class="result-cell"><div class="result-cell-label">Direction accuracy</div><div class="result-cell-val" id="bt-acc">—</div></div>
          <div class="result-cell"><div class="result-cell-label">Information coeff.</div><div class="result-cell-val" id="bt-ic">—</div></div>
        </div>
      </div>

      <button class="btn-primary" id="bt-run" style="width:100%;margin-top:4px">Run backtest</button>
      <div class="future-notice">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.5"/><path d="M7 4v4M7 9.5v.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
        Requires trained models — run the full pipeline first
      </div>
      <div style="font-size:11px;color:var(--txt-3);padding-top:8px;border-top:1px solid var(--border)">
        Out-of-sample test on the last 20% of historical data (dates the model never saw during training).
        Strategy: long when XGBoost predicts a positive return, flat otherwise.
      </div>
    </div>
    <div class="tool-chart-card">
      <div class="chart-title" style="margin-bottom:4px">Strategy vs Buy &amp; Hold</div>
      <div style="font-size:11px;color:var(--txt-3);margin-bottom:12px" id="bt-period">Run the backtest to see results</div>
      <div style="position:relative;height:340px">
        <canvas id="chart-bt"></canvas>
        <div id="bt-empty" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--txt-3);font-size:13px">
          Enter a ticker and click Run
        </div>
      </div>
    </div>
  </div>`;
}

async function runBacktest() {
  const ticker  = (document.getElementById('bt-ticker')?.value || 'AAPL').trim().toUpperCase();
  const capital = parseFloat(document.getElementById('bt-capital')?.value) || 10000;
  const btn     = document.getElementById('bt-run');

  if (btn) { btn.textContent = 'Running...'; btn.disabled = true; }
  document.getElementById('bt-empty').style.display = 'flex';

  try {
    const data = await api.backtest([ticker], capital, false);
    if (data.error) { toast(data.error, 'error'); return; }

    const m = data.metrics || {};
    const retEl = document.getElementById('bt-total-ret');
    const ret   = m.total_return || 0;
    if (retEl) { retEl.textContent = (ret>=0?'+':'') + (ret*100).toFixed(2)+'%'; retEl.style.color = ret>=0?'var(--green)':'var(--red)'; }

    const excess = m.excess_return || 0;
    setText('bt-vs-bench', `vs buy & hold: ${excess>=0?'+':''}${(excess*100).toFixed(2)}%`);

    const sharpe = m.sharpe_ratio || 0;
    const shrEl  = document.getElementById('bt-sharpe');
    if (shrEl) { shrEl.textContent = sharpe.toFixed(2); shrEl.style.color = sharpe>=1?'var(--green)':sharpe>=0?'var(--amber)':'var(--red)'; }

    setText('bt-dd',  (m.max_drawdown*100).toFixed(2)+'%');
    setText('bt-acc', (m.direction_accuracy*100).toFixed(1)+'%');
    setText('bt-ic',  (m.information_coefficient||0).toFixed(4));

    document.getElementById('bt-metrics').style.display = 'block';
    setText('bt-period', data.test_period || '');

    /* draw equity curves */
    if (data.equity_curve?.length) {
      document.getElementById('bt-empty').style.display = 'none';
      const labels = data.equity_curve.map(p => p.date);
      const strat  = data.equity_curve.map(p => p.value);
      const bench  = data.benchmark_curve?.map(p => p.value) || [];
      backtestChart('chart-bt', labels, strat, bench, capital);
    }
  } catch (err) {
    toast('Backtest failed: ' + (err.detail || err.message || 'Unknown error'), 'error');
  } finally {
    if (btn) { btn.textContent = 'Run backtest'; btn.disabled = false; }
  }
}

/* Box-Muller transform — generates normally distributed random variable */
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ══════════════════════════════════════════════════════════════════════════
   TAB 7 — Validation Results
   Shows evidence that the models actually work:
     - XGBoost cross-validation AUC across 5 folds
     - Sentiment ablation (direction accuracy with vs without FinBERT)
     - Backtest Sharpe per ticker vs buy-and-hold
     - LSTM vs naive momentum baseline
   ══════════════════════════════════════════════════════════════════════════ */
function buildValidation() {
  return `<div style="margin-bottom:16px;display:flex;align-items:center;justify-content:space-between">
    <div>
      <div class="chart-title">Model validation results</div>
      <div style="font-size:12px;color:var(--txt-2);margin-top:4px">
        Run <code style="background:var(--bg-2);padding:2px 6px;border-radius:4px;font-size:11px">python scripts/evaluate_models.py</code> to generate results
      </div>
    </div>
    <button class="btn-secondary" id="val-run-btn">Run evaluation</button>
  </div>
  <div id="val-content">
    <div class="empty-state" style="padding:48px">
      <div class="empty-icon">🧪</div>
      <div class="empty-text">No results yet — click Run evaluation or run the script from the terminal</div>
    </div>
  </div>`;
}

document.addEventListener('click', async e => {
  if (e.target.id !== 'val-run-btn') return;
  e.target.textContent = 'Running…';
  e.target.disabled    = true;
  try {
    await api.runEvaluation();
    toast('Evaluation started — results appear in ~60 seconds', 'info');
    setTimeout(loadValidationResults, 65_000);
  } catch {
    toast('Could not start evaluation — check the server is running', 'error');
    e.target.textContent = 'Run evaluation';
    e.target.disabled    = false;
  }
});

/* called when the Validation tab becomes visible */
const _origTabSwitch = bindTabSwitching;
document.addEventListener('click', e => {
  if (e.target.classList.contains('tool-tab') && e.target.dataset.tool === 'validation') {
    loadValidationResults();
  }
});

async function loadValidationResults() {
  const el = document.getElementById('val-content');
  if (!el) return;
  try {
    const data = await api.validationResults();
    if (data.error) {
      el.innerHTML = `<div class="empty-state" style="padding:32px"><div class="empty-text">${data.error}</div></div>`;
      return;
    }
    renderValidationResults(data, el);
  } catch {
    el.innerHTML = `<div class="empty-state" style="padding:32px"><div class="empty-text">Could not load results — start the backend first</div></div>`;
  }
}

function renderValidationResults(data, el) {
  const xgb  = data.xgboost_cv          || {};
  const sa   = data.sentiment_ablation   || {};
  const bt   = data.backtest             || {};
  const lstm = data.lstm_vs_baseline     || {};
  const gen  = data.generated_at ? new Date(data.generated_at).toLocaleString() : '';

  const foldBars = (xgb.fold_aucs || []).map((auc, i) => {
    const pct = Math.round(auc * 100);
    const col = auc >= 0.55 ? 'var(--green)' : auc >= 0.5 ? 'var(--amber)' : 'var(--red)';
    return `<div style="display:flex;align-items:center;gap:10px;padding:4px 0">
      <span style="font-size:11px;color:var(--txt-2);min-width:40px">Fold ${i+1}</span>
      <div style="flex:1;height:4px;background:var(--bg-0);border-radius:2px">
        <div style="width:${Math.max(pct-40,0)*3.33}%;height:100%;background:${col};border-radius:2px"></div>
      </div>
      <span style="font-family:var(--mono);font-size:12px">${auc}</span>
    </div>`;
  }).join('');

  const btRows = Object.entries(bt.per_ticker || {}).map(([ticker, m]) => {
    const beats = m.total_return > m.benchmark_total_return;
    return `<tr>
      <td class="mono" style="font-weight:700">${ticker}</td>
      <td class="mono" style="color:${m.sharpe_ratio>=1?'var(--green)':m.sharpe_ratio>=0?'var(--amber)':'var(--red)'}">${m.sharpe_ratio.toFixed(2)}</td>
      <td class="mono" style="color:${m.total_return>=0?'var(--green)':'var(--red)'}">${(m.total_return*100).toFixed(2)}%</td>
      <td class="mono">${(m.benchmark_total_return*100).toFixed(2)}%</td>
      <td class="mono" style="color:${beats?'var(--green)':'var(--red)'}">${beats?'✓':'-'}</td>
      <td class="mono">${(m.direction_accuracy*100).toFixed(1)}%</td>
    </tr>`;
  }).join('');

  el.innerHTML = `
    <div style="font-size:11px;color:var(--txt-3);margin-bottom:16px">Generated: ${gen}</div>

    <!-- summary cards -->
    <div class="stat-row" style="grid-template-columns:repeat(4,1fr);margin-bottom:20px">
      <div class="stat-card">
        <div class="stat-label">XGBoost CV AUC</div>
        <div class="stat-value" style="font-size:22px;color:${(xgb.mean_auc||0)>=0.55?'var(--green)':'var(--amber)'}">
          ${xgb.mean_auc ?? '--'}
        </div>
        <div class="stat-change">±${xgb.std_auc ?? '--'} across 5 folds</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Direction accuracy</div>
        <div class="stat-value" style="font-size:22px">${xgb.mean_dir_acc ?? '--'}</div>
        <div class="stat-change">Avg over test folds</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Sentiment lift</div>
        <div class="stat-value" style="font-size:22px;color:${(sa.sentiment_delta||0)>0?'var(--green)':'var(--red)'}">
          ${sa.sentiment_delta != null ? (sa.sentiment_delta >= 0 ? '+' : '') + sa.sentiment_delta : '--'}
        </div>
        <div class="stat-change">Dir. acc with vs without FinBERT</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Beats buy-and-hold</div>
        <div class="stat-value" style="font-size:22px;color:${(bt.pct_beats_bh||0)>50?'var(--green)':'var(--amber)'}">
          ${bt.pct_beats_bh != null ? bt.pct_beats_bh + '%' : '--'}
        </div>
        <div class="stat-change">of backtested tickers</div>
      </div>
    </div>

    <div class="two-col" style="margin-bottom:16px">
      <!-- XGBoost fold scores -->
      <div class="chart-box">
        <div class="chart-title" style="margin-bottom:14px">XGBoost — per-fold AUC (TimeSeriesSplit)</div>
        ${foldBars || '<div style="color:var(--txt-3);font-size:12px">No fold data</div>'}
        <div style="font-size:11px;color:var(--txt-3);margin-top:10px">
          AUC > 0.5 = model has predictive power over a random classifier.
          Consistent scores across folds confirm generalisation (no memorisation).
        </div>
      </div>

      <!-- Sentiment ablation -->
      <div class="chart-box">
        <div class="chart-title" style="margin-bottom:14px">Sentiment ablation</div>
        ${sa.dir_acc_with_sentiment != null ? `
          <div style="display:flex;flex-direction:column;gap:12px">
            ${[
              ['With FinBERT sentiment', sa.dir_acc_with_sentiment, 'var(--green)'],
              ['Without sentiment',      sa.dir_acc_without_sentiment, 'var(--txt-3)'],
            ].map(([label, val, col]) => `
              <div>
                <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                  <span style="font-size:12px;color:var(--txt-2)">${label}</span>
                  <span style="font-family:var(--mono);font-size:13px;font-weight:700;color:${col}">${val}</span>
                </div>
                <div style="height:6px;background:var(--bg-0);border-radius:3px">
                  <div style="width:${Math.round(val*200)}%;height:100%;background:${col};border-radius:3px"></div>
                </div>
              </div>`).join('')}
            <div style="font-size:11px;color:var(--txt-3);margin-top:4px">
              Delta: <strong style="color:${(sa.sentiment_delta||0)>0?'var(--green)':'var(--red)'}">${sa.sentiment_delta >= 0 ? '+' : ''}${sa.sentiment_delta}</strong>
              — direct measure of FinBERT's contribution to predictive accuracy
            </div>
          </div>` : '<div style="color:var(--txt-3);font-size:12px">Run evaluation to see ablation results</div>'}
      </div>
    </div>

    <!-- LSTM vs baseline -->
    ${lstm.avg_lstm_dir_acc != null ? `
    <div class="chart-box" style="margin-bottom:16px">
      <div class="chart-title" style="margin-bottom:10px">LSTM vs naive baseline (predict yesterday\'s direction)</div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
        <div class="result-cell">
          <div class="result-cell-label">LSTM dir. accuracy</div>
          <div class="result-cell-val pos">${lstm.avg_lstm_dir_acc}</div>
        </div>
        <div class="result-cell">
          <div class="result-cell-label">Baseline dir. accuracy</div>
          <div class="result-cell-val">${lstm.avg_baseline_dir_acc}</div>
        </div>
        <div class="result-cell">
          <div class="result-cell-label">Improvement over baseline</div>
          <div class="result-cell-val ${lstm.lstm_improvement > 0 ? 'pos' : 'neg'}">${lstm.lstm_improvement >= 0 ? '+' : ''}${lstm.lstm_improvement}</div>
        </div>
      </div>
    </div>` : ''}

    <!-- backtest table -->
    ${btRows ? `
    <div class="card" style="padding:0;overflow:hidden">
      <div style="padding:14px 18px;border-bottom:1px solid var(--border)">
        <div class="chart-title">Walk-forward backtest — last 20% of data (out-of-sample)</div>
        <div style="font-size:11.5px;color:var(--txt-2);margin-top:4px">
          Avg Sharpe: <strong>${bt.avg_sharpe}</strong> &nbsp;·&nbsp;
          Strategy return: <strong>${bt.avg_strategy_return != null ? (bt.avg_strategy_return*100).toFixed(2)+'%' : '--'}</strong> &nbsp;·&nbsp;
          Buy &amp; hold: <strong>${bt.avg_bh_return != null ? (bt.avg_bh_return*100).toFixed(2)+'%' : '--'}</strong>
        </div>
      </div>
      <div style="overflow-x:auto">
        <table class="data-table">
          <thead><tr><th>Ticker</th><th>Sharpe</th><th>Strategy</th><th>B&H</th><th>Beats B&H</th><th>Dir. acc</th></tr></thead>
          <tbody>${btRows}</tbody>
        </table>
      </div>
    </div>` : ''}`;

  document.getElementById('val-run-btn').textContent = 'Re-run evaluation';
  document.getElementById('val-run-btn').disabled    = false;
}
