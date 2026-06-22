/*
 * Portfolio page — holdings, recommendation approvals, drift detection, backtest.
 *
 * Key behaviours:
 *  - Stocks already in the portfolio show "ADD MORE" not "BUY" in rec panel
 *  - Backtest results open in a scrollable panel BELOW the holdings table
 *  - Return column switchable: 1D / 5D / 30D / 6M / 1Y / 5Y
 *  - Refresh button reloads pending recommendations from the server
 */

let _returnPeriod = '1D';   /* currently selected period for the return column */

function renderPortfolio() {
  document.getElementById('view-portfolio').innerHTML = `

    <!-- pending recommendations -->
    <div id="rec-panel" style="margin-bottom:18px"></div>

    <div class="portfolio-header">
      <div>
        <div class="page-heading" style="font-size:16px">Holdings</div>
        <div class="page-sub" id="port-meta">Run analysis to populate</div>
      </div>
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <select id="port-filter" class="styled-select styled-select-sm" style="max-width:130px">
          <option value="all">All signals</option>
          <option value="BUY">BUY only</option>
          <option value="HOLD">HOLD only</option>
          <option value="SELL">SELL only</option>
        </select>
        <button class="btn-secondary" id="bt-btn">
          <svg viewBox="0 0 14 14" fill="none" width="12" height="12">
            <path d="M1 11L4 7L7 9L11 3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
          Backtest
        </button>
        <button class="btn-secondary" id="export-btn">Export CSV</button>
      </div>
    </div>

    <!-- KPI row -->
    <div class="stat-row" style="grid-template-columns:repeat(5,1fr);margin-bottom:12px">
      ${pCard('Invested',    'id="ps-invested"')}
      ${pCard('Cash left',  'id="ps-cash"')}
      ${pCard('Positions',  'id="ps-pos"')}
      ${pCard('Exp. return','id="ps-ret"')}
      ${pCard('Risk profile','id="ps-risk"')}
    </div>

    <!-- risk metrics -->
    <div id="risk-panel" style="display:none;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px" class="stat-row">
      ${pCard('Sharpe ratio',      'id="rm-sharpe"')}
      ${pCard('95% VaR (1-day)',   'id="rm-var"')}
      ${pCard('95% CVaR (1-day)', 'id="rm-cvar"')}
    </div>

    <!-- drift alert -->
    <div id="drift-panel" style="display:none;margin-bottom:14px"></div>

    <!-- return period selector -->
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px">
      <span style="font-size:11.5px;color:var(--txt-2)">Predicted return:</span>
      ${['1D','5D','30D','6M','1Y','5Y'].map(p =>
        `<button class="chip period-chip ${p==='1D'?'active':''}" data-period="${p}">${p}</button>`
      ).join('')}
      <span style="font-size:11px;color:var(--txt-3);margin-left:4px">Model-based estimates</span>
    </div>

    <!-- holdings table -->
    <div class="card" style="padding:0;overflow:hidden;margin-bottom:14px">
      <div id="holdings-wrap" style="overflow-x:auto">
        <div class="empty-state" style="padding:48px">
          <div class="empty-icon">💼</div>
          <div class="empty-text">Run analysis to see portfolio holdings</div>
          <button class="btn-primary" style="margin-top:12px" onclick="triggerRun()">Run Analysis</button>
        </div>
      </div>
    </div>

    <!-- backtest results (below holdings, not replacing them) -->
    <div id="bt-panel" class="card" style="display:none;margin-bottom:14px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
        <div style="font-weight:600;font-size:13px">Out-of-sample backtest</div>
        <button class="btn-secondary" style="font-size:11px" onclick="document.getElementById('bt-panel').style.display='none'">✕ Close</button>
      </div>
      <div id="bt-content"></div>
    </div>`;

  if (state.portfolioData) updatePortfolio(state.portfolioData);
  loadPendingRecommendations();
  bindPortfolioEvents();
}

function pCard(label, attrs) {
  return `<div class="stat-card"><div class="stat-label">${label}</div><div class="stat-value" style="font-size:18px" ${attrs}>--</div></div>`;
}

/* ── Recommendations panel ───────────────────────────────────────────────── */

async function loadPendingRecommendations() {
  const userId = state.userId;
  if (!userId) return;
  try {
    const data    = await api.getUserRecs(userId);
    const pending = data.pending || [];
    const panel   = document.getElementById('rec-panel');
    if (!panel) return;

    if (!pending.length) {
      panel.innerHTML = '';
      return;
    }

    /* get current portfolio tickers to distinguish BUY vs ADD MORE */
    const inPortfolio = new Set(
      (state.portfolioData?.portfolio?.holdings || []).map(h => h.ticker)
    );

    panel.innerHTML = `
      <div class="card" style="border-color:var(--green)">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
          <div>
            <div class="chart-title">Advisor recommendations</div>
            <div style="font-size:12px;color:var(--txt-2);margin-top:3px">
              ${pending.length} pending — approve or reject each one
            </div>
          </div>
          <button class="btn-secondary" style="font-size:11.5px" onclick="loadPendingRecommendations()">
            ↻ Refresh
          </button>
        </div>
        <div style="display:flex;flex-direction:column;gap:10px">
          ${pending.map(rec => recCard(rec, inPortfolio)).join('')}
        </div>
      </div>`;
  } catch {}
}

function recCard(rec, inPortfolio) {
  const alreadyHeld = inPortfolio.has(rec.ticker);
  const actionLabel = rec.action === 'BUY' && alreadyHeld ? 'ADD MORE' : rec.action;
  const conf        = rec.confidence || 50;
  const confColor   = conf>=75?'var(--green)':conf>=50?'var(--amber)':'var(--red)';
  const factors     = rec.factors || {};

  return `
    <div class="card" style="padding:14px 16px;border-color:var(--border-hi)">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px">
        <div style="flex:1">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap">
            <span style="font-size:15px;font-weight:700;font-family:var(--mono)">${rec.ticker}</span>
            <span class="badge badge-${(rec.action||'hold').toLowerCase()}">${actionLabel}</span>
            ${alreadyHeld ? `<span style="font-size:11px;background:var(--blue-dim);color:var(--blue);padding:2px 8px;border-radius:4px;border:1px solid var(--blue)">Already held</span>` : ''}
            <span style="font-size:12px;color:${confColor};font-weight:600">Confidence: ${conf}%</span>
          </div>
          <div style="font-size:12.5px;color:var(--txt-2);margin-bottom:10px;line-height:1.55">${rec.reason||''}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
            ${Object.entries(factors).map(([name,val])=>`
              <div style="font-size:10.5px">
                <div style="display:flex;justify-content:space-between;margin-bottom:2px;color:var(--txt-3)">
                  <span>${name.replace('_',' ')}</span><span>${val}%</span>
                </div>
                <div style="height:3px;background:var(--bg-0);border-radius:2px">
                  <div style="width:${val}%;height:100%;background:${val>=70?'var(--green)':val>=50?'var(--amber)':'var(--red)'};border-radius:2px"></div>
                </div>
              </div>`).join('')}
          </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:8px;flex-shrink:0">
          <button class="btn-primary" style="padding:8px 16px" onclick="approveRec(${rec.id})">✓ Approve</button>
          <button class="btn-secondary" style="padding:8px 16px;color:var(--red);border-color:var(--red)" onclick="rejectRec(${rec.id})">✗ Reject</button>
        </div>
      </div>
    </div>`;
}

async function approveRec(recId) {
  try {
    await api.approveRec(state.userId, recId);
    toast('Recommendation approved — portfolio updated', 'success');
    loadPendingRecommendations();
    /* refresh portfolio display */
    const port = await api.getUserPortfolio(state.userId).catch(() => null);
    if (port?.holdings) {
      if (state.portfolioData?.portfolio) state.portfolioData.portfolio.holdings = port.holdings;
      renderHoldings(port.holdings);
    }
  } catch (e) { toast('Could not approve: ' + (e.detail||e.message), 'error'); }
}

async function rejectRec(recId) {
  try {
    await api.rejectRec(state.userId, recId);
    toast('Recommendation rejected', 'info');
    loadPendingRecommendations();
  } catch (e) { toast('Could not reject: ' + (e.detail||e.message), 'error'); }
}

/* ── Portfolio population ────────────────────────────────────────────────── */

function updatePortfolio(data) {
  const p = data?.portfolio;
  if (!p) return;

  const invested = p.total_invested  || 0;
  const cash     = p.cash_remaining  || 0;
  const ret      = p.expected_portfolio_return || 0;

  setText('ps-invested','$'+invested.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}));
  setText('ps-cash',    '$'+cash.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2}));
  setText('ps-pos',     p.n_positions||'0');

  const retEl = document.getElementById('ps-ret');
  if(retEl){retEl.style.color=ret>=0?'var(--green)':'var(--red)';retEl.textContent=(ret>=0?'+':'')+ret.toFixed(2)+'%';}
  setText('ps-risk',(p.risk_profile||state.risk||'moderate').charAt(0).toUpperCase()+(p.risk_profile||state.risk||'moderate').slice(1));

  const metaEl=document.getElementById('port-meta');
  if(metaEl) metaEl.textContent=`${p.n_positions} positions · $${invested.toLocaleString()} invested · Sharpe-optimised`;

  const rm=p.risk_metrics;
  if(rm){
    const rp=document.getElementById('risk-panel');
    if(rp)rp.style.display='grid';
    const sharpe=rm.annualized_sharpe;
    const sEl=document.getElementById('rm-sharpe');
    if(sEl){sEl.style.color=sharpe>=1?'var(--green)':sharpe>=0?'var(--amber)':'var(--red)';sEl.textContent=sharpe?.toFixed(2)??'--';}
    setText('rm-var',  rm.var_95_1day  !=null?(rm.var_95_1day*100).toFixed(2)+'%':'--');
    setText('rm-cvar', rm.cvar_95_1day !=null?(rm.cvar_95_1day*100).toFixed(2)+'%':'--');
  }

  /* drift */
  if(data.drift_suggestions?.length) renderDriftPanel(data.drift_suggestions);

  renderHoldings(p.holdings||[]);
}

function renderDriftPanel(suggestions) {
  const el=document.getElementById('drift-panel');
  if(!el)return;
  el.style.display='block';
  el.innerHTML=`
    <div class="card" style="border-color:var(--amber)">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
        <span style="font-size:16px">⚠️</span>
        <div>
          <div class="chart-title">Rebalancing suggested</div>
          <div style="font-size:12px;color:var(--txt-2)">${suggestions.length} position${suggestions.length>1?'s':''} have drifted &gt;5% from target</div>
        </div>
      </div>
      ${suggestions.map(s=>`
        <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border)">
          <span class="ticker">${s.ticker}</span>
          <span style="font-size:12px;color:${s.drift>0?'var(--red)':'var(--green)'}">
            ${s.drift>0?'▲ Overweight':'▼ Underweight'} ${Math.abs(s.drift).toFixed(1)}%
          </span>
          <span style="font-size:12px;color:var(--txt-2)">${s.current_weight.toFixed(1)}% → target ${s.target_weight.toFixed(1)}%</span>
        </div>`).join('')}
    </div>`;
}

/* ── Holdings table with period-switchable returns ───────────────────────── */

function renderHoldings(holdings) {
  if (!holdings.length) return;
  const wrap    = document.getElementById('holdings-wrap');
  const maxCost = Math.max(...holdings.map(h=>h.total_cost),1);

  wrap.innerHTML = `
    <table class="data-table">
      <thead><tr>
        <th>Ticker</th><th>Signal</th><th>Conf.</th><th>Score</th>
        <th>Shares</th><th>Price</th><th>Cost</th>
        <th>Weight</th><th id="ret-col-header">Predicted (${_returnPeriod})</th><th>Allocation</th>
      </tr></thead>
      <tbody id="holdings-tbody">
        ${holdings.map(h=>holdingRow(h,maxCost)).join('')}
      </tbody>
    </table>`;

  document.getElementById('port-filter')?.addEventListener('change',e=>{
    const val=e.target.value;
    document.querySelectorAll('#holdings-tbody tr').forEach(row=>{
      row.style.display=val==='all'||row.dataset.signal===val?'':'none';
    });
  });
}

function holdingRow(h, maxCost) {
  const ret      = predictedReturn(h, _returnPeriod);
  const retColor = ret>=0?'var(--green)':'var(--red)';
  const pct      = Math.round((h.total_cost/maxCost)*100);
  const conf     = h.confidence||50;
  const confCol  = conf>=75?'var(--green)':conf>=50?'var(--amber)':'var(--red)';

  return `<tr data-signal="${h.signal}">
    <td><span class="ticker">${h.ticker}</span></td>
    <td><span class="${badgeClass(h.signal)}">${h.signal}</span></td>
    <td class="mono" style="color:${confCol}">${conf}%</td>
    <td class="mono">${h.composite_score!=null?h.composite_score.toFixed(0):'--'}</td>
    <td class="mono">${h.shares}</td>
    <td class="mono">$${h.price.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</td>
    <td class="mono">$${h.total_cost.toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2})}</td>
    <td class="mono">${h.weight_pct.toFixed(1)}%</td>
    <td class="mono" style="color:${retColor}">${ret>=0?'+':''}${ret.toFixed(2)}%</td>
    <td style="min-width:110px">
      <div class="pct-bar-wrap"><div class="pct-bar" style="width:${pct}%"></div></div>
      <div class="holding-detail">${h.weight_pct.toFixed(1)}%</div>
    </td>
  </tr>`;
}

/* project 1-day predicted return to different horizons */
function predictedReturn(h, period) {
  const daily = h.predicted_return || 0;          /* 1-day model output */
  const ann   = state.portfolioData?.portfolio?.expected_portfolio_return || daily * 252;
  switch (period) {
    case '1D':  return daily;
    case '5D':  return daily * 5;
    case '30D': return daily * 21;
    case '6M':  return daily * 126;
    case '1Y':  return ann;
    case '5Y':  return (Math.pow(1 + ann/100, 5) - 1) * 100;
    default:    return daily;
  }
}

/* ── Backtest — appended BELOW holdings, never hides them ────────────────── */

async function runPortfolioBacktest() {
  const p = state.portfolioData?.portfolio;
  if (!p?.holdings?.length) { toast('Run analysis first', 'error'); return; }
  const tickers = p.holdings.map(h=>h.ticker).slice(0,5);
  const btn     = document.getElementById('bt-btn');
  if(btn){btn.textContent='Running…';btn.disabled=true;}
  try {
    const result = await api.backtest(tickers, state.budget||10000, true);
    renderBacktestResults(result);
    /* scroll to backtest panel without hiding holdings */
    document.getElementById('bt-panel')?.scrollIntoView({behavior:'smooth',block:'start'});
  } catch(err) {
    toast('Backtest failed: '+(err.detail||err.message||''), 'error');
  } finally {
    if(btn){btn.textContent='Backtest';btn.disabled=false;}
  }
}

function renderBacktestResults(data) {
  const panel   = document.getElementById('bt-panel');
  const content = document.getElementById('bt-content');
  if(!panel||!content)return;
  const individual = data.individual_metrics||{};
  const summary    = data.portfolio_summary||{};
  const sCol = v => v>=1?'var(--green)':v>=0?'var(--amber)':'var(--red)';

  content.innerHTML=`
    <div style="font-size:11px;color:var(--txt-3);margin-bottom:10px">
      Walk-forward test · last 20% of data · strategy = long when XGBoost predicts positive return
    </div>
    <div class="stat-row" style="grid-template-columns:repeat(3,1fr);margin-bottom:12px">
      ${pCard('Avg Sharpe',`style="color:${sCol(summary.avg_sharpe||0)}">${(summary.avg_sharpe||0).toFixed(2)}<span`)}
      ${pCard('Best ticker',`>${summary.best_ticker||'--'}<span`)}
      ${pCard('Avg dir. acc',`>${((summary.avg_direction_acc||0)*100).toFixed(1)}%<span`)}
    </div>
    <div style="overflow-x:auto">
      <table class="data-table">
        <thead><tr><th>Ticker</th><th>Sharpe</th><th>Strategy</th><th>Buy &amp; Hold</th><th>Max DD</th><th>Dir. acc</th></tr></thead>
        <tbody>${Object.entries(individual).map(([ticker,m])=>`
          <tr>
            <td class="mono" style="font-weight:700">${ticker}</td>
            <td class="mono" style="color:${sCol(m.sharpe_ratio||0)}">${(m.sharpe_ratio||0).toFixed(2)}</td>
            <td class="mono" style="color:${(m.total_return||0)>=0?'var(--green)':'var(--red)'}">${((m.total_return||0)*100).toFixed(2)}%</td>
            <td class="mono">${((m.benchmark_total_return||0)*100).toFixed(2)}%</td>
            <td class="mono" style="color:var(--red)">${((m.max_drawdown||0)*100).toFixed(2)}%</td>
            <td class="mono">${((m.direction_accuracy||0)*100).toFixed(1)}%</td>
          </tr>`).join('')}
        </tbody>
      </table>
    </div>`;

  panel.style.display='block';
}

/* ── Event binding ───────────────────────────────────────────────────────── */

function bindPortfolioEvents() {
  document.getElementById('export-btn')?.addEventListener('click', exportCSV);
  document.getElementById('bt-btn')?.addEventListener('click',     runPortfolioBacktest);

  /* period chips */
  document.addEventListener('click', e => {
    if (!e.target.classList.contains('period-chip')) return;
    document.querySelectorAll('.period-chip').forEach(c=>c.classList.remove('active'));
    e.target.classList.add('active');
    _returnPeriod = e.target.dataset.period;
    /* update header and re-render rows */
    const hdr = document.getElementById('ret-col-header');
    if(hdr) hdr.textContent = `Predicted (${_returnPeriod})`;
    const holdings = state.portfolioData?.portfolio?.holdings||[];
    if(holdings.length) {
      const maxCost = Math.max(...holdings.map(h=>h.total_cost),1);
      const tbody   = document.getElementById('holdings-tbody');
      if(tbody) tbody.innerHTML = holdings.map(h=>holdingRow(h,maxCost)).join('');
    }
  });
}

function exportCSV() {
  const p = state.portfolioData?.portfolio;
  if(!p?.holdings?.length){toast('No data to export','error');return;}
  const rows=[['Ticker','Signal','Confidence%','Shares','Price','Cost','Weight%','Pred.Return%','Sentiment']];
  p.holdings.forEach(h=>rows.push([h.ticker,h.signal,h.confidence||'--',h.shares,h.price,h.total_cost,h.weight_pct,h.predicted_return,h.sentiment]));
  const csv=rows.map(r=>r.join(',')).join('\n');
  const link=document.createElement('a');
  link.href='data:text/csv;charset=utf-8,'+encodeURIComponent(csv);
  link.download=`portfolio_${new Date().toISOString().split('T')[0]}.csv`;
  link.click();
  toast('CSV exported','success');
}

function badgeClass(signal) {
  return {BUY:'badge badge-buy',HOLD:'badge badge-hold',SELL:'badge badge-sell'}[signal]||'badge';
}
