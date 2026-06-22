/*
 * Analysis page — deep technical dive for a single stock.
 * Only re-renders when explicitly requested. Preserves ticker between tab switches.
 */

let _currentTicker  = null;
let _analysisReady  = false;

function renderAnalysis() {
  /* if already rendered and has a ticker loaded, just show it */
  if (_analysisReady && _currentTicker) return;

  _analysisReady = true;

  document.getElementById('view-analysis').innerHTML = `
    <div class="analysis-search">
      <input id="an-ticker" class="field-input"
             placeholder="Enter ticker (e.g. AAPL)"
             style="max-width:180px;text-transform:uppercase" />
      <select id="an-period" class="styled-select styled-select-sm" style="max-width:110px">
        <option value="60">60 days</option>
        <option value="120" selected>120 days</option>
        <option value="252">1 year</option>
      </select>
      <button class="btn-primary" id="an-load-btn">Load chart</button>
      ${state.portfolioData?.portfolio?.holdings?.length
        ? `<div style="display:flex;gap:6px;flex-wrap:wrap">
            ${state.portfolioData.portfolio.holdings.slice(0,8).map(h =>
              `<button class="chip" onclick="loadAnalysis('${h.ticker}')">${h.ticker}</button>`
            ).join('')}
          </div>` : ''}
    </div>

    <div class="analysis-grid" id="analysis-grid" style="display:none">
      <div class="analysis-charts">
        <div class="chart-box">
          <div class="chart-header">
            <div>
              <div class="chart-title" id="an-price-title">Price</div>
              <div class="chart-sub">Close · SMA 20 · SMA 50 · Bollinger Bands</div>
            </div>
          </div>
          <div class="chart-wrap-xl"><canvas id="chart-price"></canvas></div>
        </div>
        <div class="two-col">
          <div class="chart-box">
            <div class="chart-title" style="margin-bottom:12px">RSI (14)</div>
            <div class="chart-wrap-sm"><canvas id="chart-rsi"></canvas></div>
            <div style="font-size:11px;color:var(--txt-3);margin-top:6px">Above 70 = overbought · Below 30 = oversold</div>
          </div>
          <div class="chart-box">
            <div class="chart-title" style="margin-bottom:12px">MACD (12,26,9)</div>
            <div class="chart-wrap-sm"><canvas id="chart-macd"></canvas></div>
            <div style="font-size:11px;color:var(--txt-3);margin-top:6px">Histogram above zero = bullish momentum</div>
          </div>
        </div>
        <div class="chart-box">
          <div class="chart-title" style="margin-bottom:12px">Volume</div>
          <div style="position:relative;height:110px"><canvas id="chart-vol"></canvas></div>
          <div style="font-size:11px;color:var(--txt-3);margin-top:6px">Teal = volume &gt;1.5× 20-day average</div>
        </div>
      </div>

      <div class="analysis-sidebar">
        <div class="card" id="an-signal-card">
          <div class="chart-title" style="margin-bottom:12px">Model signal</div>
          <div style="text-align:center;padding:8px 0 16px">
            <div id="an-signal-badge" style="font-size:28px;font-weight:900;letter-spacing:-1px">--</div>
            <div id="an-signal-score" style="font-size:12px;color:var(--txt-2);margin-top:4px">Score: --</div>
          </div>
          <div id="an-regime" style="text-align:center;margin-bottom:14px"></div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px">
            <div class="result-cell"><div class="result-cell-label">XGBoost</div><div class="result-cell-val" id="an-pred-xgb">--</div></div>
            <div class="result-cell"><div class="result-cell-label">LSTM</div><div class="result-cell-val" id="an-pred-lstm">--</div></div>
          </div>
          <div class="result-cell" style="text-align:center">
            <div class="result-cell-label">Ensemble prediction</div>
            <div class="result-cell-val" id="an-pred-ens" style="font-size:18px">--</div>
          </div>
          <div id="an-confidence" style="margin-top:12px"></div>
        </div>

        <div class="card">
          <div class="chart-title" style="margin-bottom:12px">Indicator snapshot</div>
          <div id="an-indicators"><div style="font-size:12px;color:var(--txt-3)">Loading…</div></div>
        </div>

        <div class="card">
          <div class="chart-title" style="margin-bottom:10px">FinBERT sentiment</div>
          <div id="an-sentiment"><div style="font-size:12px;color:var(--txt-3)">Loading…</div></div>
        </div>

        <div class="card">
          <div class="chart-title" style="margin-bottom:10px">Top prediction drivers</div>
          <div class="future-notice" style="margin-bottom:10px">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.5"/><path d="M7 4v4M7 9.5v.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
            Requires trained XGBoost model
          </div>
          <div id="an-features"></div>
        </div>
      </div>
    </div>

    <div id="an-empty" style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:80px 0;gap:16px;text-align:center">
      <div style="font-size:32px;opacity:0.4">📈</div>
      <div style="font-size:15px;font-weight:600">Select a stock to analyse</div>
      <div style="font-size:13px;color:var(--txt-2);max-width:320px">Enter a ticker above or click one of the chips from your portfolio</div>
    </div>`;

  document.getElementById('an-load-btn')?.addEventListener('click', () => {
    const t = document.getElementById('an-ticker')?.value.trim().toUpperCase();
    if (t) loadAnalysis(t); else showAnalysisError(null, 'Enter a ticker first');
  });
  document.getElementById('an-ticker')?.addEventListener('keydown', e => {
    if (e.key === 'Enter') { const t = e.target.value.trim().toUpperCase(); if (t) loadAnalysis(t); }
  });
  document.getElementById('an-period')?.addEventListener('change', () => {
    if (_currentTicker) loadAnalysis(_currentTicker);
  });

  /* load ticker if one is remembered */
  if (_currentTicker) {
    setTimeout(() => loadAnalysis(_currentTicker), 80);
  }
}

async function loadAnalysis(ticker) {
  _currentTicker = ticker.toUpperCase();
  const days     = parseInt(document.getElementById('an-period')?.value || '120');
  const btn      = document.getElementById('an-load-btn');
  const tickerEl = document.getElementById('an-ticker');
  if (tickerEl) tickerEl.value = _currentTicker;
  if (btn) { btn.textContent = 'Loading…'; btn.disabled = true; }

  document.getElementById('analysis-grid').style.display = 'none';
  document.getElementById('an-empty').style.display      = 'none';

  try {
    const [history, analysis] = await Promise.all([
      api.stockHistory(_currentTicker, days),
      api.analysis(_currentTicker),
    ]);

    if (!history?.history?.length) throw new Error('No price history for ' + _currentTicker);

    drawAllCharts(history, _currentTicker);
    drawSignalCard(analysis);
    drawConfidence(analysis.confidence);
    drawIndicators(analysis.indicators);
    drawSentiment(analysis.sentiment);
    drawFeatureImportance(analysis.top_features);
    document.getElementById('analysis-grid').style.display = 'grid';

  } catch (err) {
    const isNet = err instanceof TypeError && err.message.includes('fetch');
    showAnalysisError(_currentTicker,
      isNet ? 'Server not responding — is uvicorn running on port 8000?'
            : (err.detail || err.message || 'No data for this ticker — run the pipeline first'));
  } finally {
    if (btn) { btn.textContent = 'Load chart'; btn.disabled = false; }
  }
}

function showAnalysisError(ticker, message) {
  const el = document.getElementById('an-empty');
  if (!el) return;
  el.style.display = 'flex';
  el.innerHTML = `
    <div style="font-size:32px;opacity:0.4">⚠️</div>
    <div style="font-size:15px;font-weight:600">${ticker ? 'Could not load ' + ticker : 'Error'}</div>
    <div style="font-size:13px;color:var(--txt-2);max-width:340px;line-height:1.6">${message}</div>
    ${ticker ? `<button class="btn-secondary" style="margin-top:8px" onclick="triggerRun()">Run pipeline first</button>` : ''}`;
}

function drawAllCharts(data, ticker) {
  const rows = data.history || [];
  if (!rows.length) return;
  const labels = rows.map(r => r.date);
  setText('an-price-title', `${ticker} — Price`);
  priceChart('chart-price', labels, rows.map(r=>r.close), rows.map(r=>r.sma20), rows.map(r=>r.sma50), rows.map(r=>r.bb_upper), rows.map(r=>r.bb_lower));
  rsiChart('chart-rsi',    labels, rows.map(r=>r.rsi));
  macdChart('chart-macd',  labels, rows.map(r=>r.macd), rows.map(r=>r.macd_signal), rows.map(r=>r.macd_hist));
  volumeChart('chart-vol', labels, rows.map(r=>r.volume), rows.map(r=>r.volume_ratio));
}

function drawSignalCard(a) {
  const sig  = a.signal || 'HOLD';
  const score = a.score  || 50;
  const pred  = a.prediction || {};
  const sigEl = document.getElementById('an-signal-badge');
  if (sigEl) { sigEl.textContent = sig; sigEl.style.color = sig==='BUY'?'var(--green)':sig==='SELL'?'var(--red)':'var(--amber)'; }
  setText('an-signal-score', `Score: ${score.toFixed(0)}/100  ·  $${a.close?.toFixed(2)??'--'}`);
  const regimeEl = document.getElementById('an-regime');
  if (regimeEl && a.regime) {
    const lbl = {trending_up:'↑ Trending up', trending_down:'↓ Trending down', ranging:'→ Ranging'};
    regimeEl.innerHTML = `<span class="regime-badge regime-${a.regime}">${lbl[a.regime]||a.regime}</span>`;
  }
  const colour = v => v>0?'var(--green)':'var(--red)';
  const fmt    = v => v!=null?`${v>=0?'+':''}${v.toFixed(3)}%`:'--';
  const xEl=document.getElementById('an-pred-xgb'); const lEl=document.getElementById('an-pred-lstm'); const eEl=document.getElementById('an-pred-ens');
  if(xEl){xEl.textContent=fmt(pred.xgboost); xEl.style.color=colour(pred.xgboost||0);}
  if(lEl){lEl.textContent=fmt(pred.lstm);    lEl.style.color=colour(pred.lstm||0);}
  if(eEl){eEl.textContent=fmt(pred.ensemble);eEl.style.color=colour(pred.ensemble||0);}
}

function drawConfidence(conf) {
  const el = document.getElementById('an-confidence');
  if (!el || !conf) return;
  const overall  = conf.overall || 50;
  const colour   = overall>=75?'var(--green)':overall>=50?'var(--amber)':'var(--red)';
  const factors  = conf.factors || {};
  el.innerHTML = `
    <div style="margin-bottom:8px;display:flex;justify-content:space-between;align-items:center">
      <span style="font-size:12px;color:var(--txt-2)">Model confidence</span>
      <span style="font-size:16px;font-weight:700;color:${colour}">${overall}%</span>
    </div>
    ${Object.entries(factors).map(([name,val])=>`
      <div style="font-size:10.5px;margin-bottom:5px">
        <div style="display:flex;justify-content:space-between;margin-bottom:2px;color:var(--txt-3)">
          <span>${name.replace('_',' ')}</span><span>${val}%</span>
        </div>
        <div style="height:3px;background:var(--bg-0);border-radius:2px">
          <div style="width:${val}%;height:100%;background:${val>=70?'var(--green)':val>=50?'var(--amber)':'var(--red)'};border-radius:2px"></div>
        </div>
      </div>`).join('')}`;
}

function drawIndicators(ind) {
  const el = document.getElementById('an-indicators');
  if (!el || !ind) return;
  const rows = [
    {name:'RSI (14)',        val:ind.rsi?.toFixed(1),   note:rsiNote(ind.rsi)},
    {name:'ADX (14)',        val:ind.adx?.toFixed(1),   note:adxNote(ind.adx)},
    {name:'Stochastic %K',  val:ind.stoch_k?.toFixed(1),note:stochNote(ind.stoch_k)},
    {name:'Bollinger %B',   val:ind.bb_pct?.toFixed(3), note:bbNote(ind.bb_pct)},
    {name:'ATR %',          val:ind.atr_pct!=null?(ind.atr_pct*100).toFixed(2)+'%':'--', note:''},
    {name:'52-week pos.',   val:ind.week52_pos!=null?(ind.week52_pos*100).toFixed(0)+'%':'--', note:''},
    {name:'Volume ratio',   val:ind.volume_ratio?.toFixed(2), note:(ind.volume_ratio||0)>1.5?'Elevated':'Normal'},
    {name:'10-day momentum',val:ind.momentum_10d!=null?(ind.momentum_10d>=0?'+':'')+ind.momentum_10d.toFixed(2)+'%':'--', note:''},
  ];
  el.innerHTML = rows.map(r=>`
    <div class="indicator-row">
      <span class="indicator-name">${r.name}</span>
      <div style="display:flex;align-items:center;gap:8px">
        ${r.note?`<span style="font-size:10.5px;color:var(--txt-3)">${r.note}</span>`:''}
        <span class="indicator-val">${r.val??'--'}</span>
      </div>
    </div>`).join('');
}

function drawSentiment(sent) {
  const el = document.getElementById('an-sentiment');
  if (!el || !sent) return;
  const score  = sent.score??0;
  const label  = sent.label??'neutral';
  const colour = label==='positive'?'var(--green)':label==='negative'?'var(--red)':'var(--amber)';
  el.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
      <span class="badge badge-${label}">${label.charAt(0).toUpperCase()+label.slice(1)}</span>
      <span style="font-size:13px;font-weight:700;color:${colour}">${score>=0?'+':''}${score.toFixed(4)}</span>
    </div>
    <div class="sentiment-bar-wrap"><div class="sentiment-bar" style="width:${Math.round(Math.abs(score)*100)}%;background:${colour}"></div></div>
    <div style="font-size:11px;color:var(--txt-3);margin-top:6px">${sent.count} articles · FinBERT (Araci, 2019)</div>`;
}

function drawFeatureImportance(features) {
  const el = document.getElementById('an-features');
  if (!el) return;
  if (!features || !Object.keys(features).length) { el.innerHTML='<div style="font-size:12px;color:var(--txt-3)">Train models to see feature importance</div>'; return; }
  const max = Math.max(...Object.values(features));
  el.innerHTML = Object.entries(features).slice(0,8).map(([name,val])=>`
    <div class="feature-importance-bar">
      <span class="fi-name">${name}</span>
      <div class="fi-bar-wrap"><div class="fi-bar" style="width:${Math.round(val/max*100)}%"></div></div>
      <span class="fi-val">${(val*100).toFixed(1)}%</span>
    </div>`).join('');
}

function rsiNote(rsi)   { if(rsi==null)return''; if(rsi>=70)return'⚠ Overbought'; if(rsi<=30)return'⚠ Oversold'; return rsi>=60?'Strong':rsi<=40?'Weak':'Neutral'; }
function adxNote(adx)   { if(adx==null)return''; if(adx>=40)return'Very strong'; if(adx>=25)return'Trending'; return'Ranging'; }
function stochNote(k)   { if(k==null)return''; if(k>=80)return'Overbought'; if(k<=20)return'Oversold'; return''; }
function bbNote(pct)    { if(pct==null)return''; if(pct>=0.9)return'Near upper band'; if(pct<=0.1)return'Near lower band'; return''; }
