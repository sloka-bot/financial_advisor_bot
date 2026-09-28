/* Saved holdings and explicit approval of proposed transactions. */
const RETURN_PERIOD =
  "21 sessions"; /* horizon used by the recommendation model */

function renderPortfolio() {
  document.getElementById("view-portfolio").innerHTML = `
    <div class="page-intro"><div><span class="section-kicker">Review before you act</span><h1>Your portfolio</h1><p>Saved holdings and proposed changes, kept separate.</p></div><div style="display:flex;gap:8px"><button class="btn-secondary" data-action="import-portfolio">Import portfolio</button><button class="btn-primary" data-action="build-portfolio">Build portfolio</button></div></div>

    <div id="rec-panel" class="rec-panel"></div>

    <div class="portfolio-header">
      <div>
        <div class="page-heading">Holdings</div>
        <div class="page-sub" id="port-meta">Saved positions</div>
      </div>
      <div class="pf-toolbar">
        <select id="port-filter" class="styled-select styled-select-sm">
          <option value="all">All signals</option>
          <option value="BUY">BUY only</option>
          <option value="HOLD">HOLD only</option>
          <option value="SELL">SELL only</option>
        </select>
        <button class="btn-secondary" data-action="export-csv">Export CSV</button>
      </div>
    </div>

    <div class="stat-row kpi-5">
      ${pCard("Holdings value", "ps-invested")}
      ${pCard("Cash left", "ps-cash")}
      ${pCard("Positions", "ps-pos")}
      ${pCard("Historical annual return", "ps-ret")}
      ${pCard("Risk profile", "ps-risk")}
    </div>

    <div id="risk-panel" class="stat-row risk-3 hidden">
      ${pCard("Sharpe ratio", "rm-sharpe")}
      ${pCard("95% VaR (1-day)", "rm-var")}
      ${pCard("95% CVaR (1-day)", "rm-cvar")}
    </div>

    <div id="drift-panel" class="drift-panel hidden"></div>

    <div class="period-row">
      <span>Predicted return:</span>
      <span class="chip active">${RETURN_PERIOD}</span>
      <span class="muted">Model-based estimate, one horizon</span>
    </div>

    <div class="card holdings-card">
      <div id="holdings-wrap" class="holdings-wrap">
        <div class="empty-state">
          <div class="empty-text">No saved holdings yet</div>
          <button class="btn-primary" data-action="refresh-advice">Refresh advice</button>
        </div>
      </div>
    </div>`;

  bindPortfolioEvents();
  if (state.portfolioData) updatePortfolio(state.portfolioData);
  if (state.userId) loadPendingRecommendations();
}

function pCard(label, id) {
  return `<div class="stat-card"><div class="stat-label">${escapeHtml(label)}</div><div class="stat-value kpi-value" id="${id}">--</div></div>`;
}

/* One delegated handler for the whole page, bound once. */
function bindPortfolioEvents() {
  const view = document.getElementById("view-portfolio");
  if (view.dataset.bound) return;
  view.dataset.bound = "1";

  view.addEventListener("click", (event) => {
    const target = event.target.closest("[data-action], [data-ticker]");
    if (!target) return;
    const action = target.dataset.action;
    if (action === "generate-recs") {
      generateNewRecs();
      return;
    }
    if (action === "import-portfolio") {
      openImport(state.userId);
      return;
    }
    if (action === "build-portfolio") {
      buildPortfolio();
      return;
    }
    if (action === "refresh-recs") {
      generateNewRecs();
      return;
    }
    if (action === "refresh-advice") {
      refreshAdvice();
      return;
    }
    if (action === "export-csv") {
      exportCSV();
      return;
    }
    if (action === "approve") {
      approveRec(Number(target.dataset.rec));
      return;
    }
    if (action === "reject") {
      rejectRec(Number(target.dataset.rec));
      return;
    }
    if (action === "buy-more") {
      buyMoreHolding(target.dataset.ticker);
      return;
    }
    if (action === "sell") {
      sellHolding(target.dataset.ticker);
      return;
    }
    if (target.dataset.ticker) {
      goAnalysis(target.dataset.ticker);
    }
  });

  view.addEventListener("change", (event) => {
    if (event.target.id !== "port-filter") return;
    const value = event.target.value;
    document.querySelectorAll("#holdings-tbody tr").forEach((row) => {
      row.style.display =
        value === "all" || row.dataset.signal === value ? "" : "none";
    });
  });
}

/* Recommendations panel */

async function loadPendingRecommendations() {
  const userId = state.userId;
  if (!userId) return;
  try {
    const data = await api.getUserRecs(userId);
    const pending = data.pending || [];
    const panel = document.getElementById("rec-panel");
    if (!panel) return;

    const inPortfolio = new Set(
      (state.portfolioData?.portfolio?.holdings || []).map((h) => h.ticker),
    );

    if (!pending.length) {
      panel.innerHTML = `
        <div class="rec-empty">
          <div class="rec-empty-text">No pending recommendations</div>
          <div class="rec-empty-actions">
            <button class="btn-primary btn-xs" data-action="generate-recs">Generate new</button>
            <button class="btn-secondary btn-xs" data-action="refresh-recs">Refresh</button>
          </div>
        </div>`;
      return;
    }

    panel.innerHTML = `
      <div class="card rec-box">
        <div class="rec-box-head">
          <div>
            <div class="chart-title">Advisor recommendations</div>
            <div class="rec-box-sub">Showing ${Math.min(3, pending.length)} of ${pending.length} pending. Approve or dismiss to see the next recommendation.</div>
          </div>
          <button class="btn-secondary btn-xs" data-action="refresh-recs">Refresh</button>
        </div>
        <div class="rec-legend">BUY = model expects price to rise &nbsp;|&nbsp; SELL = model expects price to fall &nbsp;|&nbsp; Support score = a heuristic combining model, sentiment and technical signals; not a probability</div>
        <div class="rec-list">${pending.slice(0, 3).map((rec) => recCard(rec, inPortfolio)).join("")}</div>
      </div>`;
  } catch (error) {
    toast(error.message, "error");
  }
}

function confClass(value) {
  if (!Number.isFinite(value)) return "";
  return value >= 75 ? "conf-hi" : value >= 50 ? "conf-mid" : "conf-lo";
}
function barClass(value) {
  return value >= 70
    ? "bar-fill-hi"
    : value >= 50
      ? "bar-fill-mid"
      : "bar-fill-lo";
}

function recCard(rec, inPortfolio) {
  const alreadyHeld = inPortfolio.has(rec.ticker);
  const actionLabel =
    rec.action === "BUY" && alreadyHeld ? "ADD MORE" : rec.action;
  const conf = Number.isFinite(rec.confidence) ? rec.confidence : null;
  const confText = conf === null ? "Unavailable" : `${conf}/100`;
  const factors = rec.factors || {};

  const factorRows = Object.entries(factors)
    .map(([name, val]) => {
      const pct = Number.isFinite(Number(val))
        ? Math.max(0, Math.min(100, Number(val)))
        : 0;
      return `
      <div class="rec-factor">
        <div class="rec-factor-head"><span>${escapeHtml(name.replaceAll("_", " "))}</span><span>${pct}%</span></div>
        <div class="rec-bar"><div class="rec-bar-fill ${barClass(pct)}" style="width:${pct}%"></div></div>
      </div>`;
    })
    .join("");

  return `
    <div class="card rec-item">
      <div class="rec-item-row">
        <div class="rec-item-main">
          <div class="rec-title-row">
            <span class="rec-ticker">${escapeHtml(rec.ticker)}</span>
            <span class="${badgeClass(rec.action)}">${escapeHtml(actionLabel)}</span>
            ${alreadyHeld ? '<span class="rec-held">Already held</span>' : ""}
            <span class="rec-score ${confClass(conf)}">Support score: ${confText}</span>
          </div>
          <div class="rec-reason">${escapeHtml(rec.reason || "")}</div>
          <div class="rec-factors">${factorRows}</div>
        </div>
        <div class="rec-actions">
          <button class="btn-primary" data-action="approve" data-rec="${rec.id}">Approve</button>
          <button class="btn-secondary btn-danger" data-action="reject" data-rec="${rec.id}">Reject</button>
        </div>
      </div>
    </div>`;
}

async function generateNewRecs() {
  const panel = document.getElementById("rec-panel");
  if (panel)
    panel.innerHTML =
      '<div class="rec-loading">Generating recommendations from trained model...</div>';
  try {
    const res = await api.generateRecs(state.userId);
    await loadPendingRecommendations();
    const n =
      res && typeof res.generated === "number"
        ? res.generated
        : (res && res.pending && res.pending.length) || 0;
    if (n > 0)
      toast(`Generated ${n} recommendation${n === 1 ? "" : "s"}`, "success");
    else
      toast("No new eligible signals meet the current cash and risk limits", "info");
  } catch (error) {
    toast("Could not generate: " + (error.detail || error.message), "error");
    loadPendingRecommendations();
  }
}

async function approveRec(recId) {
  try {
    const approved = await api.approveRec(state.userId, recId);
    if (!approved.approved)
      throw new Error(approved.error || "Recommendation could not be applied");
    toast("Recommendation approved", "success");

    const port = await api.getUserPortfolio(state.userId);
    if (port) {
      state.portfolioData = state.portfolioData || {};
      state.portfolioData.portfolio = state.portfolioData.portfolio || {};
      const holdings = port.holdings || [];
      const cash =
        typeof port.cash === "number" ? port.cash : port.cash_remaining || 0;
      const invested =
        port.total_invested ??
        holdings.reduce(
          (sum, h) => sum + (h.current_value ?? h.total_cost ?? 0),
          0,
        );
      Object.assign(state.portfolioData.portfolio, port, {
        holdings,
        cash,
        cash_remaining: cash,
        total_invested: invested,
        n_positions: holdings.length,
      });
      updatePortfolio(state.portfolioData);
      if (typeof renderDashboard === "function") renderDashboard();
    }
    loadPendingRecommendations();
  } catch (error) {
    toast("Could not approve: " + (error.detail || error.message), "error");
  }
}

async function rejectRec(recId) {
  try {
    await api.rejectRec(state.userId, recId);
    toast("Recommendation rejected", "info");
    loadPendingRecommendations();
  } catch (error) {
    toast("Could not reject: " + (error.detail || error.message), "error");
  }
}

async function refreshBook() {
  const port = await api.getUserPortfolio(state.userId);
  if (!port) return;
  state.portfolioData = state.portfolioData || {};
  state.portfolioData.portfolio = state.portfolioData.portfolio || {};
  const holdings = port.holdings || [];
  const cash =
    typeof port.cash === "number" ? port.cash : port.cash_remaining || 0;
  const invested =
    port.total_invested ??
    holdings.reduce((sum, h) => sum + (h.current_value ?? h.total_cost ?? 0), 0);
  Object.assign(state.portfolioData.portfolio, port, {
    holdings,
    cash,
    cash_remaining: cash,
    total_invested: invested,
    n_positions: holdings.length,
  });
  updatePortfolio(state.portfolioData);
  if (typeof renderDashboard === "function") renderDashboard();
}

async function buyMoreHolding(ticker) {
  const entered = prompt(`How many additional whole shares of ${ticker}?`);
  if (entered === null) return;
  const shares = Number(entered);
  if (!Number.isSafeInteger(shares) || shares < 1) {
    toast("Enter a positive whole number of shares", "error");
    return;
  }
  try {
    const result = await api.buyMore(state.userId, ticker, shares);
    toast(`Added ${shares} shares of ${ticker}. Fees: ${money(result.fees)}`, "success");
    await refreshBook();
    await loadPendingRecommendations();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function sellHolding(ticker) {
  if (!ticker) return;
  if (!confirm(`Sell your entire ${ticker} position and credit the proceeds to cash?`))
    return;
  try {
    const res = await api.sellHolding(state.userId, ticker);
    toast(`Sold ${ticker} for ${money(res.proceeds)}`, "success");
    await refreshBook();
    loadPendingRecommendations();
  } catch (error) {
    toast("Could not sell: " + (error.detail || error.message), "error");
  }
}

async function buildPortfolio() {
  const userId = state.userId;
  if (!userId) {
    toast("Create a profile first", "error");
    return;
  }
  if (
    !confirm(
      "Build an allocation of up to five stocks from your available cash and add it to your current holdings? Fees and your whole-portfolio risk limits apply.",
    )
  )
    return;
  const panel = document.getElementById("rec-panel");
  try {
    if (panel)
      panel.innerHTML =
        '<div class="rec-loading">Building an allocation of up to five stocks...</div>';
    const res = await api.buildPortfolio(userId);
    await refreshBook();
    loadPendingRecommendations();
    if (res && res.built > 0)
      toast(
        `Added ${res.built} holding${res.built === 1 ? "" : "s"} to your portfolio`,
        "success",
      );
    else toast("No affordable names to add right now", "info");
  } catch (error) {
    toast("Could not build portfolio: " + (error.detail || error.message), "error");
    loadPendingRecommendations();
  }
}

/* Portfolio population */

function updatePortfolio(data) {
  const p = data?.portfolio;
  if (!p) return;

  const invested = p.total_invested || 0;
  const cash = p.cash_remaining || 0;
  const ret = p.expected_portfolio_return;

  setText("ps-invested", money(invested));
  setText("ps-cash", money(cash));
  setText("ps-pos", String(p.n_positions || 0));

  const retEl = document.getElementById("ps-ret");
  if (retEl) {
    const retFinite = Number.isFinite(ret);
    retEl.classList.toggle("val-up", retFinite && ret >= 0);
    retEl.classList.toggle("val-down", retFinite && ret < 0);
    retEl.textContent = Number.isFinite(ret)
      ? (ret >= 0 ? "+" : "") + ret.toFixed(2) + "%"
      : "Unavailable";
  }
  const risk = p.risk_profile || state.risk || "moderate";
  setText("ps-risk", risk.charAt(0).toUpperCase() + risk.slice(1));

  const metaEl = document.getElementById("port-meta");
  if (metaEl)
    metaEl.textContent = `${p.n_positions} positions · ${money(invested)} invested · Saved holdings`;

  const rm = p.risk_metrics;
  if (rm) {
    document.getElementById("risk-panel")?.classList.remove("hidden");
    const sharpe = rm.annualized_sharpe;
    const sEl = document.getElementById("rm-sharpe");
    if (sEl) {
      sEl.classList.toggle("conf-hi", sharpe >= 1);
      sEl.classList.toggle("conf-mid", sharpe >= 0 && sharpe < 1);
      sEl.classList.toggle("conf-lo", sharpe < 0);
      sEl.textContent = Number.isFinite(sharpe) ? sharpe.toFixed(2) : "--";
    }
    setText(
      "rm-var",
      rm.var_95_1day != null ? (rm.var_95_1day * 100).toFixed(2) + "%" : "--",
    );
    setText(
      "rm-cvar",
      rm.cvar_95_1day != null ? (rm.cvar_95_1day * 100).toFixed(2) + "%" : "--",
    );
  }

  if (data.drift_suggestions?.length) renderDriftPanel(data.drift_suggestions);

  renderHoldings(p.holdings || []);
}

function renderDriftPanel(suggestions) {
  const el = document.getElementById("drift-panel");
  if (!el) return;
  el.classList.remove("hidden");
  el.innerHTML = `
    <div class="card drift-box">
      <div class="drift-box-head">
        <div>
          <div class="chart-title">Rebalancing suggested</div>
          <div class="muted">${suggestions.length} position${suggestions.length > 1 ? "s" : ""} have drifted &gt;5% from target</div>
        </div>
      </div>
      ${suggestions
        .map(
          (s) => `
        <div class="drift-row">
          <span class="ticker">${escapeHtml(s.ticker)}</span>
          <span class="${s.drift > 0 ? "val-down" : "val-up"}">${s.drift > 0 ? "Overweight" : "Underweight"} ${Math.abs(s.drift).toFixed(1)}%</span>
          <span class="muted">${s.current_weight.toFixed(1)}% target ${s.target_weight.toFixed(1)}%</span>
        </div>`,
        )
        .join("")}
    </div>`;
}

/* Holdings table */

function renderHoldings(holdings) {
  const wrap = document.getElementById("holdings-wrap");
  if (!wrap) return;
  if (!holdings.length) {
    wrap.innerHTML =
      '<div class="empty-state">No saved holdings. Import holdings or approve a recommendation.</div>';
    return;
  }
  const maxCost = Math.max(...holdings.map((h) => h.total_cost), 1);

  wrap.innerHTML = `
    <table class="data-table">
      <thead><tr>
        <th>Ticker</th><th>Signal</th><th>Support</th><th>Batch rank</th>
        <th>Shares</th><th>Price</th><th>Cost</th>
        <th>Weight</th><th>Predicted (${RETURN_PERIOD})</th><th>Allocation</th><th>Action</th>
      </tr></thead>
      <tbody id="holdings-tbody">${holdings.map((h) => holdingRow(h, maxCost)).join("")}</tbody>
    </table>`;
}

function holdingRow(h, maxCost) {
  const ret = predictedReturn(h);
  const pct = Math.round((h.total_cost / maxCost) * 100);
  const conf = Number.isFinite(h.confidence) ? h.confidence : null;
  const confText = conf === null ? "--" : `${conf}/100`;

  return `<tr data-signal="${escapeHtml(h.signal)}" data-ticker="${escapeHtml(h.ticker)}" class="row-link" title="Click to view full analysis for ${escapeHtml(h.ticker)}">
    <td><span class="ticker">${escapeHtml(h.ticker)}</span></td>
    <td><span class="${badgeClass(h.signal)}">${escapeHtml(h.signal)}</span></td>
    <td class="mono ${confClass(conf)}">${confText}</td>
    <td class="mono">${h.composite_score != null ? h.composite_score.toFixed(0) : "--"}</td>
    <td class="mono">${h.shares}</td>
    <td class="mono">${money(h.price)}</td>
    <td class="mono">${money(h.total_cost)}</td>
    <td class="mono">${(h.weight_pct ?? 0).toFixed(1)}%</td>
    <td class="mono ${Number.isFinite(ret) ? (ret >= 0 ? "val-up" : "val-down") : ""}">${Number.isFinite(ret) ? (ret >= 0 ? "+" : "") + ret.toFixed(2) + "%" : "Unavailable"}</td>
    <td>
      <div class="pct-bar-wrap"><div class="pct-bar" style="width:${pct}%"></div></div>
      <div class="holding-detail">${(h.weight_pct ?? 0).toFixed(1)}%</div>
    </td>
    <td><button class="btn-secondary btn-xs" data-action="buy-more" data-ticker="${escapeHtml(h.ticker)}">Buy more</button> <button class="btn-secondary btn-xs" data-action="sell" data-ticker="${escapeHtml(h.ticker)}" title="Sell all ${escapeHtml(h.ticker)}">Sell</button></td>
  </tr>`;
}

/* Predicted return from the 21-session model. */
function predictedReturn(h) {
  return Number.isFinite(h.predicted_return) ? h.predicted_return : null;
}

function exportCSV() {
  const p = state.portfolioData?.portfolio;
  if (!p?.holdings?.length) {
    toast("No data to export", "error");
    return;
  }
  const rows = [
    [
      "Ticker",
      "Signal",
      "Support score",
      "Shares",
      "Price",
      "Cost",
      "Weight%",
      "Pred.Return%",
      "Sentiment",
    ],
  ];
  p.holdings.forEach((h) =>
    rows.push([
      h.ticker,
      h.signal,
      h.confidence ?? "--",
      h.shares,
      h.price,
      h.total_cost,
      h.weight_pct,
      h.predicted_return,
      h.sentiment,
    ]),
  );
  const csv = rows.map((r) => r.join(",")).join("\n");
  const link = document.createElement("a");
  link.href = "data:text/csv;charset=utf-8," + encodeURIComponent(csv);
  link.download = `portfolio_${new Date().toISOString().split("T")[0]}.csv`;
  link.click();
  toast("CSV exported", "success");
}

function badgeClass(signal) {
  return (
    {
      BUY: "badge badge-buy",
      HOLD: "badge badge-hold",
      SELL: "badge badge-sell",
    }[signal] || "badge"
  );
}
