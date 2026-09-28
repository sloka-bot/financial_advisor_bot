/* Saved portfolio values, illustrative growth and historical correlation. */
const ILLUSTRATIVE_GROWTH_RATE = 0.05;
function renderDashboard() {
  const p = state.portfolioData?.portfolio;
  const holdings = p?.holdings || [];
  const cost = holdings.reduce((s, h) => s + (h.total_cost || 0), 0);
  const value = p?.total_invested ?? 0;
  const gain = cost > 0 ? value - cost : null;
  const stale = holdings.some((h) => h.valuation_estimated);
  document.getElementById("view-dashboard").innerHTML = `
    <div class="page-intro"><div><span class="section-kicker">Your money at a glance</span><h1>Portfolio overview</h1><p>Your key numbers and how your money is split.</p></div><button class="btn-primary" onclick="navigateTo('portfolio')">Go to portfolio <span aria-hidden="true">\u2197</span></button></div>
    <div class="stat-row dashboard-stats">
      ${dashboardStat("Total portfolio value", p ? money(p.total_value) : "Not available", "Holdings plus cash", true)}
      ${dashboardStat("Available cash", p ? money(p.cash_remaining) : "Not available", "Not invested")}
      ${dashboardStat("Unrealised change", money(gain), gain === null ? "Available after you save holdings" : `${((gain / cost) * 100).toFixed(2)}% vs what you paid`)}
      ${dashboardStat("Open positions", String(holdings.length), `${escapeHtml(state.risk)} risk`)}
    </div>
    ${stale ? '<div class="notice">Some prices are estimated, so values are indicative.</div>' : ""}
    <div class="card"><div class="section-kicker">Where your money sits</div><h2>Current allocation</h2><div class="chart-wrap"><canvas id="dashboard-allocation"></canvas></div>${holdings.length ? "" : '<p class="muted">No saved positions yet. Import or build a portfolio to begin.</p>'}</div>
    <div class="card" style="margin-top:16px"><div class="section-kicker">Projected growth</div><h2>If your portfolio compounds</h2>
      <div class="muted" style="font-size:12px;margin-bottom:8px">An illustration starting from ${money(p?.total_value)} with an assumed ${(ILLUSTRATIVE_GROWTH_RATE * 100).toFixed(0)}% annual rate, compounded monthly for 10 years. This rate is a scenario input, not a model forecast or a return promised by your risk profile. Contributions, fees and market swings are excluded.</div>
      <div style="position:relative;height:240px"><canvas id="dashboard-growth"></canvas></div></div>
    <div class="card" style="margin-top:16px"><div class="section-kicker">Diversification</div><h2>How your holdings move together</h2>
      <div class="muted" style="font-size:12px;margin-bottom:10px">Correlation of daily returns between your holdings. Green (low/negative) means they diversify each other; red (near +1) means they tend to rise and fall together, so add less diversification.</div>
      <div id="dashboard-heatmap"><p class="muted">Add two or more holdings to see how they move together.</p></div></div>`;
  if (p)
    allocationChart(
      "dashboard-allocation",
      [...holdings.map((h) => h.ticker), "Cash"],
      [...holdings.map((h) => h.current_value ?? h.total_cost), p.cash_remaining || 0],
    );
  if (Number.isFinite(p?.total_value)) drawGrowthProjection(p.total_value);
  if (holdings.length >= 2) loadCorrelationHeatmap();
}

// Illustrative compound-growth projection of the current portfolio value.
function drawGrowthProjection(startVal) {
  const rate = ILLUSTRATIVE_GROWTH_RATE;
  const r = rate / 12;
  const labels = ["Now"];
  const flat = [Math.round(startVal)];
  const compounded = [Math.round(startVal)];
  let v = startVal;
  for (let m = 1; m <= 120; m++) {
    v *= 1 + r;
    if (m % 12 === 0) {
      labels.push(`Yr ${m / 12}`);
      flat.push(Math.round(startVal));
      compounded.push(Math.round(v));
    }
  }
  compoundChart("dashboard-growth", labels, flat, compounded);
}

// Correlation heatmap of the user's holdings (data from the backend).
async function loadCorrelationHeatmap() {
  const el = document.getElementById("dashboard-heatmap");
  if (!el) return;
  try {
    const data = await api.correlation(state.userId);
    const tickers = data.tickers || [];
    const matrix = data.matrix || [];
    if (tickers.length < 2 || matrix.length !== tickers.length) {
      el.innerHTML = '<p class="muted">Not enough shared price history to calculate correlation.</p>';
      return;
    }
    const cell = (v) => {
      if (v === null || v === undefined) return "background:var(--bg-2);color:var(--txt-3)";
      const t = (v + 1) / 2; // -1..1 -> 0..1
      const red = Math.round(230 * t + 20);
      const green = Math.round(230 * (1 - t) + 20);
      return `background:rgba(${red},${green},90,0.85);color:#0b1220;font-weight:600`;
    };
    let html = '<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:12px"><tr><th></th>';
    tickers.forEach((t) => (html += `<th style="padding:6px 8px">${escapeHtml(t)}</th>`));
    html += "</tr>";
    matrix.forEach((row, i) => {
      html += `<tr><th style="padding:6px 8px;text-align:right">${escapeHtml(tickers[i])}</th>`;
      row.forEach((v) => {
        html += `<td style="padding:8px 10px;text-align:center;${cell(v)}">${v === null ? "-" : v.toFixed(2)}</td>`;
      });
      html += "</tr>";
    });
    html += "</table></div>";
    el.innerHTML = html;
  } catch (e) {
    el.innerHTML = '<p class="muted">Correlation is unavailable right now.</p>';
  }
}
function dashboardStat(label, value, note, primary = false) {
  return `<div class="stat-card ${primary ? "primary-stat" : ""}"><div class="stat-label">${label}</div><div class="stat-value">${value}</div><div class="stat-note">${note}</div></div>`;
}
function goAnalysis(ticker) {
  predictorTicker = ticker;
  navigateTo("predictor");
}
