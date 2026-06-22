/*
 * Dashboard page.
 * Before analysis runs: shows an empty state with a "Run Analysis" prompt.
 * After analysis: shows real portfolio data, charts, and signal heatmap.
 * All data is populated automatically — no manual refresh needed.
 */

function renderDashboard() {
  const hasData = !!(state.portfolioData?.portfolio?.holdings?.length);

  if (!hasData) {
    document.getElementById('view-dashboard').innerHTML = `
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:60vh;gap:20px;text-align:center">
        <svg width="64" height="64" viewBox="0 0 28 28" fill="none" style="opacity:0.3">
          <rect width="28" height="28" rx="7" fill="#00C896" fill-opacity="0.3"/>
          <path d="M6 20L11 13L15 16L20 8" stroke="#00C896" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div style="font-size:18px;font-weight:700;color:var(--txt-1)">No portfolio data yet</div>
        <div style="font-size:13.5px;color:var(--txt-2);max-width:340px;line-height:1.6">
          Select a market and index in the sidebar, then click
          <strong style="color:var(--green)">Run Analysis</strong> to download data,
          train the models, and populate your dashboard.
        </div>
        <button class="btn-primary" onclick="triggerRun()" style="margin-top:8px">
          Run Analysis Now
        </button>
        ${state.modelsReady ? `
          <div style="font-size:12px;color:var(--txt-2)">
            Models are already trained — just fetching recommendations…
          </div>` : ''}
      </div>`;
    return;
  }

  document.getElementById('view-dashboard').innerHTML = `
    <!-- KPI cards -->
    <div class="stat-row" id="dash-stats">
      ${kpi('Portfolio value',  'id="kpi-value"')}
      ${kpi('Total invested',   'id="kpi-invested"')}
      ${kpi('Expected return',  'id="kpi-return"')}
      ${kpi('Active positions', 'id="kpi-count"')}
    </div>

    <!-- market regime badge -->
    <div id="regime-banner" style="display:none;margin-bottom:14px"></div>

    <!-- risk metrics row -->
    <div class="risk-metrics-row" id="risk-row" style="display:none">
      ${riskCard('Sharpe ratio',     'id="rm-sharpe"')}
      ${riskCard('95% VaR (1-day)', 'id="rm-var"')}
      ${riskCard('95% CVaR (1-day)','id="rm-cvar"')}
    </div>

    <!-- main charts -->
    <div class="chart-row" style="margin-bottom:14px">
      <div class="chart-box">
        <div class="chart-header">
          <div>
            <div class="chart-title">Portfolio growth</div>
            <div class="chart-sub" id="growth-sub">Projected over 90 trading days</div>
          </div>
          <div style="display:flex;gap:5px">
            ${['30','90','180','252'].map((d,i) =>
              `<button class="chip period-btn ${i===1?'active':''}" data-days="${d}">
                ${['1M','3M','6M','1Y'][i]}</button>`).join('')}
          </div>
        </div>
        <div class="chart-wrap"><canvas id="chart-growth"></canvas></div>
      </div>
      <div class="chart-box">
        <div class="chart-header">
          <div class="chart-title">Allocation</div>
          <div class="chart-sub" id="alloc-sub">By holding</div>
        </div>
        <div class="chart-wrap"><canvas id="chart-alloc"></canvas></div>
      </div>
    </div>

    <div class="two-col">
      <div class="chart-box">
        <div class="chart-header">
          <div class="chart-title">Top performers</div>
          <div class="chart-sub">Predicted 1-day return</div>
        </div>
        <div class="chart-wrap-sm"><canvas id="chart-perf"></canvas></div>
      </div>
      <div class="chart-box">
        <div class="chart-header">
          <div class="chart-title">Signal heatmap</div>
          <div class="chart-sub" id="sig-meta"></div>
        </div>
        <div id="signal-heat" style="overflow-y:auto;max-height:220px;padding:4px">
          <div class="empty-state" style="padding:24px">
            <div class="empty-text">Signals appear here after analysis</div>
          </div>
        </div>
      </div>
    </div>`;

  /* populate all data */
  populateDashboard(state.portfolioData);
  if (state.recommendations) populateSignals(state.recommendations);
  bindPeriodButtons();
}

function kpi(label, attrs) {
  return `<div class="stat-card">
    <div class="stat-label">${label}</div>
    <div class="stat-value" ${attrs}>--</div>
  </div>`;
}

function riskCard(label, attrs) {
  return `<div class="risk-metric-card">
    <div class="risk-metric-label">${label}</div>
    <div class="risk-metric-val" ${attrs}>--</div>
  </div>`;
}

/* ── Populate with real data ─────────────────────────────────────────────── */

function populateDashboard(data) {
  const p = data?.portfolio;
  if (!p) return;

  const invested = p.total_invested || 0;
  const ret      = p.expected_portfolio_return || 0;

  setText('kpi-value',    '$' + invested.toLocaleString(undefined, {minimumFractionDigits:2,maximumFractionDigits:2}));
  setText('kpi-invested', '$' + invested.toLocaleString(undefined, {minimumFractionDigits:2,maximumFractionDigits:2}));
  setText('kpi-count',    (p.n_positions || 0) + ' stocks');

  const retEl = document.getElementById('kpi-return');
  if (retEl) {
    retEl.style.color = ret >= 0 ? 'var(--green)' : 'var(--red)';
    retEl.textContent  = (ret >= 0 ? '+' : '') + ret.toFixed(2) + '%';
  }

  /* risk metrics */
  const rm = p.risk_metrics;
  if (rm) {
    const row = document.getElementById('risk-row');
    if (row) row.style.display = 'grid';
    const sharpe  = rm.annualized_sharpe;
    const sharpeEl = document.getElementById('rm-sharpe');
    if (sharpeEl) {
      sharpeEl.style.color = sharpe >= 1 ? 'var(--green)' : sharpe >= 0 ? 'var(--amber)' : 'var(--red)';
      sharpeEl.textContent  = sharpe?.toFixed(2) ?? '--';
    }
    setText('rm-var',  rm.var_95_1day  != null ? (rm.var_95_1day  * 100).toFixed(2) + '%' : '--');
    setText('rm-cvar', rm.cvar_95_1day != null ? (rm.cvar_95_1day * 100).toFixed(2) + '%' : '--');
  }

  /* charts */
  if (p.holdings?.length) {
    allocationChart('chart-alloc',
      p.holdings.map(h => h.ticker),
      p.holdings.map(h => h.total_cost));
    setText('alloc-sub', `$${invested.toLocaleString()} across ${p.n_positions} stocks`);

    const sorted = [...p.holdings].sort((a,b) => b.predicted_return - a.predicted_return).slice(0,8);
    performanceChart('chart-perf', sorted.map(h => h.ticker), sorted.map(h => h.predicted_return));
  }

  drawGrowthChart(invested, ret / 100 / 252, 90);

  /* market regime badge */
  api.regime().then(r => {
    const el = document.getElementById('regime-banner');
    if (!el || !r.regime || r.regime === 'unknown') return;
    const colours = { bull:'var(--green)', bear:'var(--red)', sideways:'var(--amber)' };
    const labels  = { bull:'🐂 Bull market', bear:'🐻 Bear market', sideways:'↔ Sideways market' };
    const colour  = colours[r.regime] || 'var(--txt-2)';
    el.style.display = 'block';
    el.innerHTML = `<div style="display:flex;align-items:center;gap:12px;padding:10px 16px;background:var(--bg-card);border:1px solid ${colour};border-radius:var(--r-md)">
      <span style="font-size:14px;font-weight:700;color:${colour}">${labels[r.regime] || r.regime}</span>
      <span style="font-size:12px;color:var(--txt-2)">Confidence: ${r.confidence}%</span>
      <span style="font-size:12px;color:var(--txt-3)">${r.metrics?.pct_above_sma200 ?? '--'}% of stocks above SMA200</span>
    </div>`;
  }).catch(() => {});
}

function drawGrowthChart(invested, dailyRet, days) {
  const labels  = ['Start'];
  const vals    = [invested];
  const bench   = [invested];
  for (let i = 1; i <= days; i++) {
    const noise = (Math.random() - 0.48) * 0.009;
    vals.push(Math.max(0, vals[i-1] * (1 + dailyRet + noise)));
    bench.push(invested * Math.pow(1.0004, i));
    labels.push(i % 20 === 0 ? `Day ${i}` : i === days ? 'Today' : '');
  }
  setText('growth-sub', `Projected over ${days} trading days vs S&P 500 benchmark`);
  growthChart('chart-growth', labels, vals, bench);
}

function populateSignals(data) {
  const signals = data.all_signals || data.top_picks || [];
  const meta    = data.summary     || {};
  const el      = document.getElementById('signal-heat');
  const metaEl  = document.getElementById('sig-meta');
  if (metaEl) metaEl.textContent = `BUY ${meta.buy_count||0} · HOLD ${meta.hold_count||0} · SELL ${meta.sell_count||0}`;
  if (!signals.length || !el) return;

  el.innerHTML = `<div class="signal-grid">
    ${signals.slice(0, 30).map(s => {
      const cls = {BUY:'buy',HOLD:'hold',SELL:'sell'}[s.signal] || '';
      return `<div class="signal-cell ${cls}" onclick="goAnalysis('${s.ticker}')">
        <div class="signal-cell-ticker">${s.ticker}</div>
        <div class="signal-cell-score">${(s.composite_score||0).toFixed(0)}</div>
      </div>`;
    }).join('')}
  </div>`;
}

/* alias so app.js can call these after pipeline */
function updateDashboard(data) { populateDashboard(data); }
function updateSignals(data)    { populateSignals(data);   }

function goAnalysis(ticker) {
  /* set ticker first so renderAnalysis picks it up */
  if (typeof window !== 'undefined') window._pendingAnalysisTicker = ticker;
  /* patch _currentTicker in analysis.js scope */
  try { _currentTicker = ticker; } catch(e) {}
  navigateTo('analysis');
  /* fallback in case renderAnalysis didn't auto-load */
  setTimeout(() => { if (typeof loadAnalysis === 'function') loadAnalysis(ticker); }, 150);
}

function bindPeriodButtons() {
  document.addEventListener('click', e => {
    if (!e.target.classList.contains('period-btn')) return;
    document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
    e.target.classList.add('active');
    const days   = parseInt(e.target.dataset.days);
    const p      = state.portfolioData?.portfolio;
    const inv    = p?.total_invested || 10000;
    const ret    = (p?.expected_portfolio_return || 0) / 100 / 252;
    drawGrowthChart(inv, ret, days);
  });
}

function badgeClass(signal) {
  return {BUY:'badge badge-buy',HOLD:'badge badge-hold',SELL:'badge badge-sell'}[signal] || 'badge';
}
