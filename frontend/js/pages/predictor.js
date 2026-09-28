/* Latest LSTM estimates and a separately fitted historical XGBoost comparison. */
let predictorTicker = "AAPL";
let predictorHorizon = 21;
let predictorCache = null;
let comparisonCache = null;
let predictorBusy = false;
let comparisonBusy = false;

function forecastPrice(close, returnPct) {
  return Number.isFinite(close) && close > 0 && Number.isFinite(returnPct)
    ? close * (1 + returnPct / 100)
    : null;
}

function renderPredictor() {
  document.getElementById("view-predictor").innerHTML = `
    <div class="page-intro"><div><span class="section-kicker">From estimate to evidence</span><h1>Stock Predictor</h1><p>Explore a price estimate, then compare historical forecasts with observed prices.</p></div></div>
    <form class="card predictor-search" onsubmit="event.preventDefault(); loadPrediction()"><label for="predictor-ticker">Stock ticker</label><input id="predictor-ticker" class="field-input" value="${escapeHtml(predictorTicker)}" maxlength="12" placeholder="e.g. AAPL" autocomplete="off"><label for="predictor-horizon">Horizon</label><select id="predictor-horizon" class="styled-select styled-select-sm"><option value="1"${predictorHorizon===1?" selected":""}>1 day</option><option value="5"${predictorHorizon===5?" selected":""}>5 days</option><option value="21"${predictorHorizon===21?" selected":""}>21 days</option></select><button class="btn-primary" id="predictor-run" ${predictorBusy ? "disabled" : ""}>${predictorBusy ? "Loading..." : "View forecast"}</button><span class="muted">Saved US equity data · 1 / 5 / 21-session forecast</span></form>
    <div id="predictor-result"><div class="card empty-state">Enter a ticker to see its latest available forecast.</div></div>
    <div class="card comparison-card"><span class="section-kicker">Check what actually happened</span><h2>Historical forecast comparison</h2><p><strong>What this does:</strong> pick a past start and end date, and it plots what the model predicted the price would be against what the price actually did over that period - so you can see how accurate the model was. It fits a fresh model on data before the chosen period (not a replay of the LSTM forecast above), so it can take a few minutes.</p>
      <form class="comparison-form" onsubmit="event.preventDefault(); loadComparison()"><div><label for="comparison-start">Forecast dates from</label><input id="comparison-start" type="date" class="field-input" value="2023-01-01" required></div><div><label for="comparison-end">Through</label><input id="comparison-end" type="date" class="field-input" value="2023-12-31" required></div><button class="btn-secondary" id="comparison-run" ${comparisonBusy ? "disabled" : ""}>${comparisonBusy ? "Comparing..." : "Compare past forecasts"}</button></form>
      <div id="comparison-result" class="muted">Choose a period with enough earlier data to fit the model. This can take a few minutes.</div>
    </div>
    <div id="predictor-charts"></div>`;
  if (predictorCache?.ticker === predictorTicker && predictorCache?.horizon === predictorHorizon)
    drawPrediction(predictorCache);
  if (comparisonCache?.ticker === predictorTicker)
    drawComparison(comparisonCache);
  updateHorizonAvailability();
}

// Disable horizon options without a trained model.
async function updateHorizonAvailability() {
  try {
    const status = await api.modelStatus();
    const hz = status.horizons || {};
    const sel = document.getElementById("predictor-horizon");
    if (!sel) return;
    [...sel.options].forEach((opt) => {
      const a = hz[opt.value] || {};
      const known = Object.prototype.hasOwnProperty.call(hz, opt.value);
      const ok = a.xgboost || a.lstm;
      opt.disabled = known ? !ok : false;
      const base = opt.textContent.replace(/ \(unavailable\)$/, "");
      opt.textContent = known && !ok ? base + " (unavailable)" : base;
    });
  } catch (e) {
    /* ignore an unreachable status endpoint */
  }
}

function selectedTicker() {
  const ticker = document
    .getElementById("predictor-ticker")
    .value.trim()
    .toUpperCase();
  if (!/^[A-Z][A-Z0-9.-]{0,11}$/.test(ticker))
    throw new Error("Enter a stock ticker such as AAPL or BRK-B.");
  return ticker;
}

async function loadPrediction() {
  if (predictorBusy) return;
  const button = document.getElementById("predictor-run");
  try {
    const ticker = selectedTicker();
    const horizon = Number(document.getElementById("predictor-horizon")?.value) || 21;
    predictorHorizon = horizon;
    predictorBusy = true;
    button.disabled = true;
    button.textContent = "Loading...";
    const [analysis, prices] = await Promise.all([
      api.analysis(ticker, state.risk, horizon),
      api.stockHistory(ticker, 160),
    ]);
    predictorTicker = ticker;
    predictorCache = { ticker, horizon, analysis, history: prices.history || [] };
    if (comparisonCache?.ticker !== ticker) {
      comparisonCache = null;
      setText(
        "comparison-result",
        "Run a historical comparison for this ticker.",
      );
    }
    drawPrediction(predictorCache);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    predictorBusy = false;
    const b = document.getElementById("predictor-run");
    if (b) {
      b.disabled = false;
      b.textContent = "View forecast";
    }
  }
}

function drawPrediction({ ticker, analysis, history }) {
  const prediction = analysis.prediction || {};
  const h = prediction.horizon || predictorHorizon || 21;
  const close = history.at(-1)?.close ?? analysis.close;
  const asOf = history.at(-1)?.date || "Date unavailable";
  const estimate = forecastPrice(close, prediction.expected_return_pct);
  const probability = Number.isFinite(prediction.prob_up_pct)
    ? `${prediction.prob_up_pct.toFixed(1)}%`
    : "Unavailable";
  const signal = analysis.signal || "HOLD";
  const sigConf = Number.isFinite(analysis.support_score) ? analysis.support_score : null;
  const sigConfText = sigConf === null ? "Unavailable" : `${sigConf}/100`;
  const sigProfile = analysis.risk_profile || state.risk || "moderate";
  const sigWords = { BUY: "The model suggests buying this stock", HOLD: "The model suggests holding - no strong edge either way", SELL: "The model suggests selling this stock" }[signal] || "";
  document.getElementById("predictor-result").innerHTML = `
    <div class="card" style="display:flex;flex-direction:column;gap:8px">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <span class="${badgeClass(signal)}" style="font-size:15px;padding:6px 14px">${escapeHtml(signal)}</span>
        <span class="rec-score ${confClass(sigConf)}">Support score: ${sigConfText}</span>
        <span class="muted">for a ${escapeHtml(sigProfile)} risk profile</span>
      </div>
      <div class="muted">${sigWords}. Support score is a heuristic combining model, sentiment and technical signals - not a probability; the calibrated probability of an increase is shown below.</div>
    </div>
    <div class="stat-row predictor-stats">${dashboardStat(`${escapeHtml(ticker)} · last adjusted close`, money(close), `As of ${escapeHtml(asOf)}`)}${dashboardStat("Estimated adjusted price", money(estimate), `LSTM · ${h} trading sessions ahead`, true)}${dashboardStat("Probability of an increase", probability, "XGBoost direction classifier")}</div>
    <div class="notice">The future outcome is not known yet. A forecast can be wrong. Prices below are adjusted historical closes, not live quotes or executable prices.</div>`;
  const chartsEl = document.getElementById("predictor-charts");
  if (chartsEl) chartsEl.innerHTML = `
    <div class="card" style="margin-top:16px"><span class="section-kicker">Observed market data</span><h2>${escapeHtml(ticker)} price &amp; indicators</h2>
      <div class="muted" style="font-size:12px;margin-bottom:8px">The stock's price with its 20- and 50-day moving averages and Bollinger bands. The bands widen when the stock is more volatile; price near the upper band is relatively high, near the lower band relatively low.</div>
      <div style="height:280px"><canvas id="predictor-price"></canvas></div></div>
    <div class="card" style="margin-top:16px"><span class="section-kicker">Technical indicators</span>
      <div class="muted" style="font-size:12px;margin:6px 0 10px">These summarise recent price behaviour: RSI flags overbought/oversold, MACD shows momentum direction, and volume shows how much trading is happening.</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:10px">
        <div><div class="chart-title">RSI (14)</div><div style="height:150px"><canvas id="predictor-rsi"></canvas></div><div class="muted" style="font-size:11px;margin-top:4px">Above 70 overbought · below 30 oversold</div></div>
        <div><div class="chart-title">MACD (12, 26, 9)</div><div style="height:150px"><canvas id="predictor-macd"></canvas></div><div class="muted" style="font-size:11px;margin-top:4px">Histogram above zero = bullish momentum</div></div>
      </div>
      <div style="margin-top:16px"><div class="chart-title">Volume</div><div style="height:120px"><canvas id="predictor-volume"></canvas></div><div class="muted" style="font-size:11px;margin-top:4px">Green = volume &gt; 1.5x its 20-day average</div></div>
    </div>`;
  const dts = history.map((r) => r.date);
  priceChart("predictor-price", dts, history.map((r) => r.close), history.map((r) => r.sma20), history.map((r) => r.sma50), history.map((r) => r.bb_upper), history.map((r) => r.bb_lower));
  rsiChart("predictor-rsi", dts, history.map((r) => r.rsi));
  macdChart("predictor-macd", dts, history.map((r) => r.macd), history.map((r) => r.macd_signal), history.map((r) => r.macd_hist));
  volumeChart("predictor-volume", dts, history.map((r) => r.volume), history.map((r) => r.volume_ratio));
}

async function loadComparison() {
  if (comparisonBusy) return;
  try {
    const ticker = selectedTicker();
    const horizon = Number(document.getElementById("predictor-horizon")?.value) || predictorHorizon || 21;
    predictorHorizon = horizon;
    const start = document.getElementById("comparison-start").value;
    const end = document.getElementById("comparison-end").value;
    if (!start || !end || start > end)
      throw new Error("Choose a start date on or before the end date.");
    comparisonBusy = true;
    const button = document.getElementById("comparison-run");
    button.disabled = true;
    button.textContent = "Comparing...";
    setText(
      "comparison-result",
      `Fitting the historical model for ${ticker}. Saved holdings will not change.`,
    );
    const result = await api.backtestRange([ticker], start, end, horizon, "both");
    if (result.error) throw new Error(result.error);
    comparisonCache = { ticker, ...result };
    drawComparison(comparisonCache);
  } catch (error) {
    setText("comparison-result", error.message);
  } finally {
    comparisonBusy = false;
    const button = document.getElementById("comparison-run");
    if (button) {
      button.disabled = false;
      button.textContent = "Compare past forecasts";
    }
  }
}

function drawComparison(result) {
  const rows = (result.rows || []).filter(
    (r) =>
      r.known &&
      r.target_date &&
      Number.isFinite(r.predicted_price) &&
      Number.isFinite(r.actual_price),
  );
  if (!rows.length) {
    setText(
      "comparison-result",
      "No dated, observed outcomes are available for this period.",
    );
    return;
  }
  const mae =
    rows.reduce(
      (sum, r) => sum + Math.abs(r.predicted_price - r.actual_price),
      0,
    ) / rows.length;
  document.getElementById("comparison-result").innerHTML = `
    <div class="comparison-summary"><strong>${escapeHtml(result.ticker)}</strong><span>Model fitted using data through ${escapeHtml(result.trained_until)}</span><span>${rows.length} matched outcomes</span><span>Mean absolute price error: ${money(mae)}</span></div>
    <p class="muted">Both lines refer to the same target date, ${result.horizon || predictorHorizon} sessions after each forecast. Overlapping forecasts are not independent observations.</p><div class="chart-wrap"><canvas id="predictor-comparison"></canvas></div>
    <details class="forecast-table"><summary>View the latest 10 observed outcomes</summary><div class="table-scroll"><table class="data-table"><thead><tr><th>Forecast date</th><th>Target date</th><th>Predicted price</th><th>Observed price</th><th>Absolute error</th></tr></thead><tbody>${rows
      .slice(-10)
      .map(
        (r) =>
          `<tr><td>${escapeHtml(r.date)}</td><td>${escapeHtml(r.target_date)}</td><td>${money(r.predicted_price)}</td><td>${money(r.actual_price)}</td><td>${money(Math.abs(r.predicted_price - r.actual_price))}</td></tr>`,
      )
      .join("")}</tbody></table></div></details>`;
  priceLines(
    "predictor-comparison",
    rows.map((r) => r.target_date),
    [
      {
        label: "Historical predicted price",
        values: rows.map((r) => r.predicted_price),
        color: "#a99ae6",
      },
      {
        label: "Observed target price",
        values: rows.map((r) => r.actual_price),
        color: "#73dcc8",
      },
    ],
  );
}

function priceLines(id, labels, series) {
  drawChart(id, {
    type: "line",
    data: {
      labels,
      datasets: series.map((s) => ({
        label: s.label,
        data: s.values,
        borderColor: s.color,
        borderWidth: 2,
        pointRadius: 0,
        tension: 0,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { position: "bottom" } },
      scales: {
        x: { ticks: { maxTicksLimit: 8 } },
        y: { ticks: { callback: (v) => money(v) } },
      },
    },
  });
}
