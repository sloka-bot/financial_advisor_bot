/* Dashboard: portfolio stats and allocation only. */
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
    <div class="card"><div class="section-kicker">Where your money sits</div><h2>Current allocation</h2><div class="chart-wrap"><canvas id="dashboard-allocation"></canvas></div>${holdings.length ? "" : '<p class="muted">No saved positions yet. Import or build a portfolio to begin.</p>'}</div>`;
  if (p)
    allocationChart(
      "dashboard-allocation",
      [...holdings.map((h) => h.ticker), "Cash"],
      [...holdings.map((h) => h.current_value ?? h.total_cost), p.cash_remaining || 0],
    );
}
function dashboardStat(label, value, note, primary = false) {
  return `<div class="stat-card ${primary ? "primary-stat" : ""}"><div class="stat-label">${label}</div><div class="stat-value">${value}</div><div class="stat-note">${note}</div></div>`;
}
function goAnalysis(ticker) {
  predictorTicker = ticker;
  navigateTo("predictor");
}
