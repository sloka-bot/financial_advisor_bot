/*
 * Onboarding questionnaire — shown once per user, never again.
 * Stores risk profile, goal, horizon, and budget to the backend.
 * Subsequent logins go straight to the dashboard using the stored profile.
 */

const OB_STEPS = [
  {
    id: 'welcome',
    title: 'Welcome to FinAdvisor',
    subtitle: 'Your AI-powered portfolio advisor. We\'ll set up your profile in 4 quick steps.',
    type: 'welcome',
  },
  {
    id: 'goal',
    title: 'What\'s your investment goal?',
    subtitle: 'This shapes how we balance your portfolio.',
    type: 'cards',
    options: [
      { value: 'growth',       icon: '📈', label: 'Growth',       desc: 'Maximise long-term capital appreciation' },
      { value: 'income',       icon: '💰', label: 'Income',       desc: 'Generate steady dividend and interest income' },
      { value: 'preservation', icon: '🛡️', label: 'Preservation', desc: 'Protect capital with minimal volatility' },
      { value: 'balanced',     icon: '⚖️', label: 'Balanced',     desc: 'Equal mix of growth and stability' },
    ],
  },
  {
    id: 'risk',
    title: 'What\'s your risk tolerance?',
    subtitle: 'How would you react if your portfolio dropped 20% in a month?',
    type: 'cards',
    options: [
      { value: 'conservative', icon: '🌱', label: 'Conservative', desc: 'I\'d sell immediately — capital safety comes first' },
      { value: 'moderate',     icon: '🌿', label: 'Moderate',     desc: 'I\'d hold and wait for recovery' },
      { value: 'aggressive',   icon: '🌳', label: 'Aggressive',   desc: 'I\'d buy more — this is a buying opportunity' },
    ],
  },
  {
    id: 'horizon',
    title: 'What\'s your investment horizon?',
    subtitle: 'How long can you leave this money invested?',
    type: 'cards',
    options: [
      { value: '< 1 year',    icon: '⚡', label: 'Short term',    desc: 'Less than 1 year' },
      { value: '1-5 years',   icon: '📅', label: 'Medium term',   desc: '1 to 5 years' },
      { value: '5-10 years',  icon: '🗓️', label: 'Long term',    desc: '5 to 10 years' },
      { value: '10+ years',   icon: '🏦', label: 'Very long term',desc: 'More than 10 years' },
    ],
  },
  {
    id: 'budget',
    title: 'What\'s your starting budget?',
    subtitle: 'This is the amount you\'ll invest initially. You can change it later.',
    type: 'budget',
  },
];

let _obStep    = 0;
let _obAnswers = {};
let _obUserId  = '';

function initOnboarding(userId, isNew) {
  _obUserId = userId;
  if (!isNew) return;   /* returning users skip onboarding */

  _obStep    = 0;
  _obAnswers = {};
  showOnboarding();
}

function showOnboarding() {
  let overlay = document.getElementById('onboarding-overlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'onboarding-overlay';
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(6,10,18,0.92);display:flex;align-items:center;justify-content:center;z-index:4000;backdrop-filter:blur(6px)';
    document.body.appendChild(overlay);
  }
  overlay.style.display = 'flex';
  renderStep();
}

function renderStep() {
  const step    = OB_STEPS[_obStep];
  const overlay = document.getElementById('onboarding-overlay');
  const isLast  = _obStep === OB_STEPS.length - 1;
  const pct     = Math.round((_obStep / OB_STEPS.length) * 100);

  overlay.innerHTML = `
    <div style="background:var(--bg-card);border:1px solid var(--border-hi);border-radius:var(--r-xl);padding:36px 40px;width:520px;box-shadow:var(--shadow-lg)">

      <!-- progress -->
      <div style="margin-bottom:28px">
        <div style="height:3px;background:var(--bg-hover);border-radius:2px;margin-bottom:8px">
          <div style="width:${pct}%;height:100%;background:var(--green);border-radius:2px;transition:width 0.3s"></div>
        </div>
        <div style="font-size:11px;color:var(--txt-3)">Step ${_obStep + 1} of ${OB_STEPS.length}</div>
      </div>

      <!-- logo on welcome step -->
      ${step.type === 'welcome' ? `
        <div style="text-align:center;margin-bottom:20px">
          <svg width="48" height="48" viewBox="0 0 28 28" fill="none">
            <rect width="28" height="28" rx="7" fill="#00C896" fill-opacity="0.15"/>
            <path d="M6 20L11 13L15 16L20 8" stroke="#00C896" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>` : ''}

      <div style="font-size:19px;font-weight:700;margin-bottom:6px;text-align:${step.type==='welcome'?'center':'left'}">${step.title}</div>
      <div style="font-size:13.5px;color:var(--txt-2);margin-bottom:24px;text-align:${step.type==='welcome'?'center':'left'}">${step.subtitle}</div>

      <!-- step content -->
      ${step.type === 'cards'  ? buildCards(step)  : ''}
      ${step.type === 'budget' ? buildBudget()      : ''}
      ${step.type === 'welcome'? buildWelcomeActions() : ''}

      <!-- nav buttons (not on welcome) -->
      ${step.type !== 'welcome' ? `
        <div style="display:flex;justify-content:space-between;margin-top:24px">
          <button class="btn-secondary" onclick="obBack()">Back</button>
          <button class="btn-primary" id="ob-next" onclick="obNext()">${isLast ? 'Start investing' : 'Continue'}</button>
        </div>` : ''}
    </div>`;
}

function buildCards(step) {
  const selected = _obAnswers[step.id];
  return `<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
    ${step.options.map(o => `
      <div class="ob-card ${selected === o.value ? 'ob-card-selected' : ''}"
           onclick="selectCard('${step.id}', '${o.value}', this)"
           style="background:var(--bg-2);border:2px solid ${selected===o.value?'var(--green)':'var(--border)'};border-radius:var(--r-lg);padding:16px;cursor:pointer;transition:border-color 0.15s">
        <div style="font-size:22px;margin-bottom:8px">${o.icon}</div>
        <div style="font-weight:600;font-size:13.5px;margin-bottom:4px">${o.label}</div>
        <div style="font-size:12px;color:var(--txt-2)">${o.desc}</div>
      </div>`).join('')}
  </div>`;
}

function buildBudget() {
  return `
    <div style="display:flex;flex-direction:column;gap:16px">
      <div>
        <label class="field-label">Starting investment ($)</label>
        <input id="ob-budget" class="field-input" type="number" value="${_obAnswers.budget || 10000}" min="500" step="500"/>
      </div>
      <div>
        <label class="field-label">Monthly contribution ($) — optional</label>
        <input id="ob-monthly" class="field-input" type="number" value="${_obAnswers.monthly_contribution || 0}" min="0"/>
      </div>
    </div>`;
}

function buildWelcomeActions() {
  return `
    <div style="display:flex;justify-content:center;margin-top:8px">
      <button class="btn-primary" onclick="obNext()" style="padding:12px 32px">
        Get started
      </button>
    </div>`;
}

function selectCard(stepId, value, el) {
  _obAnswers[stepId] = value;
  /* update border on all cards */
  el.closest('.ob-card')?.parentElement?.querySelectorAll('.ob-card').forEach(c => {
    c.style.borderColor = 'var(--border)';
  });
  el.style.borderColor = 'var(--green)';
}

function obNext() {
  const step = OB_STEPS[_obStep];

  /* validate */
  if (step.type === 'cards' && !_obAnswers[step.id]) {
    toast('Please select an option', 'error');
    return;
  }
  if (step.type === 'budget') {
    _obAnswers.budget               = parseFloat(document.getElementById('ob-budget')?.value)  || 10000;
    _obAnswers.monthly_contribution = parseFloat(document.getElementById('ob-monthly')?.value) || 0;
  }

  if (_obStep >= OB_STEPS.length - 1) {
    finishOnboarding();
    return;
  }

  _obStep++;
  renderStep();
}

function obBack() {
  if (_obStep > 0) { _obStep--; renderStep(); }
}

async function finishOnboarding() {
  const btn = document.getElementById('ob-next');
  if (btn) { btn.textContent = 'Saving…'; btn.disabled = true; }

  try {
    await fetch(`http://localhost:8000/api/user/${_obUserId}`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        risk_profile:         _obAnswers.risk        || 'moderate',
        goal:                 _obAnswers.goal        || 'growth',
        investment_horizon:   _obAnswers.horizon     || '5-10 years',
        budget:               _obAnswers.budget      || 10000,
        monthly_contribution: _obAnswers.monthly_contribution || 0,
      }),
    });

    /* update sidebar with the stored profile */
    state.risk   = _obAnswers.risk   || 'moderate';
    state.budget = _obAnswers.budget || 10000;
    const riskEl   = document.getElementById('sb-risk');
    const budgetEl = document.getElementById('sb-budget');
    if (riskEl)   riskEl.value   = state.risk;
    if (budgetEl) budgetEl.value = state.budget;

    document.getElementById('onboarding-overlay').style.display = 'none';
    toast('Profile saved — run analysis to build your portfolio', 'success');

  } catch (e) {
    toast('Could not save profile — server running?', 'error');
    if (btn) { btn.textContent = 'Start investing'; btn.disabled = false; }
  }
}
