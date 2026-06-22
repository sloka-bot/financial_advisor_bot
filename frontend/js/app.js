/*
 * App entry — boots after Clerk auth, loads user profile, shows onboarding
 * for new users, pre-fills sidebar for returning users.
 *
 * Clerk key is fetched from /api/config so it is never hardcoded here.
 * The user_id comes from Clerk and is used as the key for profile storage.
 */

const PIPELINE_STEPS = [
  { key: 'fetching',    label: 'Fetching tickers',                pct: 5  },
  { key: 'downloading', label: 'Downloading price history',        pct: 15 },
  { key: 'cleaning',    label: 'Cleaning data',                   pct: 22 },
  { key: 'features',    label: 'Building 38 technical indicators', pct: 34 },
  { key: 'news',        label: 'Collecting news headlines',        pct: 48 },
  { key: 'sentiment',   label: 'Running FinBERT sentiment',        pct: 60 },
  { key: 'fusion',      label: 'Merging features + sentiment',     pct: 72 },
  { key: 'xgboost',     label: 'Training XGBoost (5-fold CV)',     pct: 82 },
  { key: 'lstm',        label: 'Training LSTM sequences',          pct: 92 },
  { key: 'done',        label: 'Complete',                         pct: 100 },
];

let _allMarkets  = {};
let _clerk       = null;
let _currentUser = null;   /* Clerk user object, set after auth */

/* ── Boot ────────────────────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', async () => {
  try {
    const cfg = await api.config().catch(() => ({}));
    const key = cfg.clerk_key || '';

    if (key && !key.includes('YOUR_KEY')) {
      _clerk = new window.Clerk(key);
      await _clerk.load();

      document.getElementById('go-signup')?.addEventListener('click', e => {
        e.preventDefault(); showPage('signup');
      });
      document.getElementById('go-signin')?.addEventListener('click', e => {
        e.preventDefault(); showPage('signin');
      });

      if (!_clerk.user) {
        showPage('signin');
        mountClerk();
        return;
      }

      _currentUser = _clerk.user;
      setUserUI(_clerk.user);
      document.getElementById('logout-btn')?.addEventListener('click',
        () => _clerk.signOut().then(() => location.reload()));

      await bootWithProfile(_clerk.user.id);
    } else {
      /* dev mode — no Clerk */
      await bootWithProfile('dev-user');
    }
  } catch (e) {
    console.warn('Auth error:', e.message);
    await bootWithProfile('dev-user');
  }
});

function showPage(page) {
  document.getElementById('page-signin').style.display = page === 'signin' ? 'flex' : 'none';
  document.getElementById('page-signup').style.display = page === 'signup' ? 'flex' : 'none';
  document.getElementById('app-layout').style.display  = 'none';
}

function mountClerk() {
  if (!_clerk) return;
  const si = document.getElementById('clerk-signin-mount');
  const su = document.getElementById('clerk-signup-mount');
  if (si && !si.children.length) _clerk.mountSignIn(si,  { afterSignInUrl: window.location.href });
  if (su && !su.children.length) _clerk.mountSignUp(su, { afterSignUpUrl:  window.location.href });
}

function setUserUI(user) {
  const first    = user.firstName || '';
  const last     = user.lastName  || '';
  const email    = user.emailAddresses?.[0]?.emailAddress || '';
  const fullName = (first + ' ' + last).trim() || email.split('@')[0] || 'User';
  const initials = (first.charAt(0) + (last.charAt(0) || '')).toUpperCase() || fullName.charAt(0).toUpperCase();

  setText('user-name',     fullName);
  setText('user-email',    email);
  setText('user-avatar',   initials);
  setText('dropdown-name', fullName);
  setText('dropdown-email',email);

  document.getElementById('user-profile-btn')?.addEventListener('click', e => {
    e.stopPropagation();
    document.getElementById('user-dropdown')?.classList.toggle('open');
  });
  document.addEventListener('click', () =>
    document.getElementById('user-dropdown')?.classList.remove('open'));
}

/* ── Profile-aware boot ──────────────────────────────────────────────────── */

async function bootWithProfile(userId) {
  document.getElementById('page-signin').style.display = 'none';
  document.getElementById('page-signup').style.display = 'none';
  document.getElementById('app-layout').style.display  = 'flex';

  state.userId = userId;

  loadMarkets();
  checkModelStatus();
  renderDashboard();
  navigateTo('dashboard');
  bindNavLinks();
  bindRunButton();
  bindSidebarInputs();
  document.getElementById('refresh-btn')?.addEventListener('click', checkModelStatus);

  /* fetch user profile */
  try {
    const profile = await api.getUser(userId);

    if (profile.is_new) {
      /* new user — show onboarding questionnaire */
      initOnboarding(userId, true);
    } else {
      /* returning user — pre-fill sidebar with stored preferences */
      state.risk   = profile.risk_profile || 'moderate';
      state.budget = profile.budget       || 10000;

      const riskEl   = document.getElementById('sb-risk');
      const budgetEl = document.getElementById('sb-budget');
      if (riskEl)   riskEl.value   = state.risk;
      if (budgetEl) budgetEl.value = state.budget;

      toast(`Welcome back${profile.name ? ', ' + profile.name : ''} — your ${state.risk} profile is loaded`, 'success');

      /* silently try to reload last portfolio */
      tryReloadPortfolio(userId, profile);
    }
  } catch {
    /* server down or profile check failed — show onboarding anyway */
    initOnboarding(userId, true);
  }
}

async function tryReloadPortfolio(userId, profile) {
  try {
    const status = await api.pipelineStatus().catch(() => ({}));
    if (status.step !== 'done' || !status.processed_tickers?.length) return;

    const [recs, port] = await Promise.all([
      api.recommend('United States', 'NASDAQ 100', profile.risk_profile || 'moderate', 20).catch(() => null),
      api.portfolio('United States', 'NASDAQ 100', profile.risk_profile || 'moderate', profile.budget || 10000, 10).catch(() => null),
    ]);

    if (port?.portfolio?.holdings?.length) {
      state.recommendations = recs;
      state.portfolioData   = port;
      renderDashboard();
    }
  } catch {}
}

/* ── Navigation ──────────────────────────────────────────────────────────── */

function navigateTo(page) {
  state.currentPage = page;
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.getElementById(`view-${page}`)?.classList.add('active');
  document.querySelectorAll('.nav-item[data-page]').forEach(a =>
    a.classList.toggle('active', a.dataset.page === page));
  const titles = { dashboard:'Dashboard', analysis:'Stock Analysis',
                   portfolio:'Portfolio',  advisor:'AI Advisor', tools:'Tools' };
  setText('page-title', titles[page] || page);
  ({ dashboard: renderDashboard, analysis: renderAnalysis,
     portfolio: renderPortfolio, advisor: renderAdvisor,
     tools: renderTools })[page]?.();
}

function bindNavLinks() {
  document.querySelectorAll('.nav-item[data-page]').forEach(a =>
    a.addEventListener('click', () => navigateTo(a.dataset.page)));
}

/* ── Market selects ──────────────────────────────────────────────────────── */

async function loadMarkets() {
  try {
    const data  = await api.markets();
    _allMarkets = data.markets || {};

    const mSel = document.getElementById('sb-market');
    if (!mSel) return;
    mSel.innerHTML = '';
    Object.keys(_allMarkets).forEach(m => {
      const o = document.createElement('option');
      o.value = m; o.textContent = m; mSel.appendChild(o);
    });
    const saved  = state.market && _allMarkets[state.market] ? state.market : Object.keys(_allMarkets)[0];
    mSel.value   = saved;
    state.market = saved;
    fillIndexes(saved);
    mSel.addEventListener('change', () => { state.market = mSel.value; fillIndexes(mSel.value); });
  } catch {}
}

function fillIndexes(market) {
  const iSel    = document.getElementById('sb-index');
  if (!iSel) return;
  const indexes = _allMarkets[market] || [];
  iSel.innerHTML = '';
  indexes.forEach(idx => {
    const o = document.createElement('option');
    o.value = idx; o.textContent = idx; iSel.appendChild(o);
  });
  const saved  = state.index && indexes.includes(state.index) ? state.index : indexes[0] || '';
  iSel.value   = saved;
  state.index  = saved;
  iSel.onchange = () => { state.index = iSel.value; };
}

function bindSidebarInputs() {
  document.getElementById('sb-risk')?.addEventListener('change',  e => { state.risk   = e.target.value; });
  document.getElementById('sb-budget')?.addEventListener('input', e => { state.budget = parseFloat(e.target.value) || 10000; });
}

/* ── Run Analysis ─────────────────────────────────────────────────────────── */

function bindRunButton() {
  document.getElementById('run-btn')?.addEventListener('click', triggerRun);
}

async function triggerRun() {
  if (state.pipelineRunning) return;

  const market = document.getElementById('sb-market')?.value || state.market || 'United States';
  const index  = document.getElementById('sb-index')?.value  || state.index  || 'NASDAQ 100';
  const risk   = document.getElementById('sb-risk')?.value   || state.risk   || 'moderate';
  const budget = parseFloat(document.getElementById('sb-budget')?.value) || state.budget || 10000;
  const scope  = document.querySelector('input[name="scope"]:checked')?.value || 'sample';
  const userId = state.userId || '';

  state.market = market; state.index = index;
  state.risk   = risk;   state.budget = budget;
  state.pipelineRunning = true;

  showLoader();

  try {
    await api.runPipeline(market, index, risk, budget, scope, userId);
    await pollUntilDone(10 * 60 * 1000);

    updateLoaderStep('done', 'Fetching recommendations…', 97);

    const [recs, port] = await Promise.all([
      api.recommend(market, index, risk, 20),
      api.portfolio(market, index, risk, budget, 10),
    ]);

    state.recommendations = recs;
    state.portfolioData   = port;

    renderDashboard();
    updatePortfolio(port);
    checkModelStatus();
    hideLoader();
    toast('Analysis complete', 'success');

  } catch (err) {
    hideLoader();
    toast('Error: ' + (err.message || err.detail || 'Check server logs'), 'error');
    console.error(err);
  } finally {
    state.pipelineRunning = false;
  }
}

function pollUntilDone(timeoutMs = 600_000) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;
    const iv = setInterval(async () => {
      if (Date.now() > deadline) { clearInterval(iv); reject(new Error('Timed out')); return; }
      try {
        const s    = await api.pipelineStatus();
        const step = PIPELINE_STEPS.find(x => x.key === s.step);
        updateLoaderStep(s.step, s.message || step?.label || s.step, s.progress || step?.pct || 0);
        if (s.step === 'done')  { clearInterval(iv); resolve(); }
        if (s.step === 'error') { clearInterval(iv); reject(new Error(s.error || 'Pipeline error')); }
      } catch {}
    }, 2000);
  });
}

/* ── Loader ──────────────────────────────────────────────────────────────── */

function showLoader() {
  let el = document.getElementById('pipeline-overlay');
  if (!el) {
    el = document.createElement('div');
    el.id = 'pipeline-overlay';
    el.innerHTML = `
      <div class="loader-card">
        <div class="loader-logo">
          <svg width="44" height="44" viewBox="0 0 28 28" fill="none">
            <rect width="28" height="28" rx="7" fill="#00C896" fill-opacity="0.15"/>
            <path d="M6 20L11 13L15 16L20 8" stroke="#00C896" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
        <div class="loader-title">Running Analysis</div>
        <div class="loader-step-label" id="ldr-label">Starting…</div>
        <div class="loader-bar-wrap"><div class="loader-bar" id="ldr-bar" style="width:2%"></div></div>
        <div class="loader-steps">
          ${PIPELINE_STEPS.filter(s => s.key !== 'done').map(s =>
            `<div class="loader-step" id="ls-${s.key}"><span class="ls-dot"></span><span>${s.label}</span></div>`
          ).join('')}
        </div>
        <div style="font-size:11px;color:var(--txt-3);margin-top:14px;text-align:center">
          First run ~15 min (FinBERT + training). After that ~2 min.
        </div>
      </div>`;
    document.body.appendChild(el);
  }
  el.style.display = 'flex';
}

function updateLoaderStep(stepKey, message, pct) {
  const label = document.getElementById('ldr-label');
  const bar   = document.getElementById('ldr-bar');
  if (label) label.textContent = message;
  if (bar)   bar.style.width   = Math.max(2, pct || 0) + '%';

  const stepIdx = PIPELINE_STEPS.findIndex(x => x.key === stepKey);
  PIPELINE_STEPS.forEach((s, i) => {
    const el  = document.getElementById(`ls-${s.key}`);
    const dot = el?.querySelector('.ls-dot');
    if (!el) return;
    if (i < stepIdx)     { el.className='loader-step done';   if(dot){dot.textContent='✓';dot.style.cssText='background:var(--green);width:14px;height:14px;color:#000;font-size:9px';} }
    else if(i===stepIdx) { el.className='loader-step active'; if(dot){dot.textContent=''; dot.style.cssText='background:var(--green);width:7px;height:7px;animation:spin 1s linear infinite';} }
    else                 { el.className='loader-step';         if(dot){dot.textContent=''; dot.style.cssText='background:var(--txt-4);width:7px;height:7px';} }
  });
}

function hideLoader() {
  const el = document.getElementById('pipeline-overlay');
  if (el) el.style.display = 'none';
}

/* ── Model status ─────────────────────────────────────────────────────────── */

async function checkModelStatus() {
  try {
    const d  = await api.modelStatus();
    const ok = d.xgboost_trained || d.lstm_trained;
    document.getElementById('model-dot')?.classList.toggle('trained', ok);
    setText('model-label', ok
      ? `XGBoost ${d.xgboost_trained?'✓':'✗'}  LSTM ${d.lstm_trained?'✓':'✗'}`
      : 'Models not trained');
    state.modelsReady = ok;
  } catch {}
}

/* ── Toast ────────────────────────────────────────────────────────────────── */

function toast(msg, type = 'info') {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = msg;
  el.className   = `toast ${type}`;
  el.classList.remove('hidden');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add('hidden'), 4500);
}
