/*
 * Onboarding questionnaire - shown once per user, never again.
 * First asks: create a new portfolio or import an existing one.
 * Stores risk profile, goal, horizon, and budget to the backend.
 * Returning users skip onboarding and go straight to the dashboard.
 */

const ONBOARDING_STEPS = [
  {
    id: "portfolio_type",
    title: "Welcome to Financial Advisor Bot",
    subtitle:
      "Explore portfolio ideas and understand their risks. How would you like to start?",
    type: "choice",
    options: [
      {
        value: "create",
        icon: "✨",
        label: "Build new portfolio",
        desc: "Review model signals and a portfolio proposal constrained by your risk profile.",
      },
      {
        value: "import",
        icon: "📂",
        label: "Import existing portfolio",
        desc: "Enter your current holdings and we will analyse and optimise them",
      },
    ],
  },
  {
    id: "goal",
    title: "What's your investment goal?",
    subtitle:
      "Recorded as a preference. Allocation currently uses your selected risk tolerance.",
    type: "cards",
    options: [
      {
        value: "growth",
        icon: "📈",
        label: "Growth",
        desc: "Maximise long-term capital appreciation",
      },
      {
        value: "income",
        icon: "💰",
        label: "Income",
        desc: "Generate steady dividend and interest income",
      },
      {
        value: "preservation",
        icon: "🛡️",
        label: "Preservation",
        desc: "Protect capital with minimal volatility",
      },
      {
        value: "balanced",
        icon: "⚖️",
        label: "Balanced",
        desc: "Equal mix of growth and stability",
      },
    ],
  },
  {
    id: "risk",
    title: "What's your risk tolerance?",
    subtitle: "How would you react if your portfolio dropped 20% in a month?",
    type: "cards",
    options: [
      {
        value: "conservative",
        icon: "🌱",
        label: "Conservative",
        desc: "I'd sell immediately - capital safety comes first",
      },
      {
        value: "moderate",
        icon: "🌿",
        label: "Moderate",
        desc: "I'd hold and wait for recovery",
      },
      {
        value: "aggressive",
        icon: "🌳",
        label: "Aggressive",
        desc: "I'd buy more - this is a buying opportunity",
      },
    ],
  },
  {
    id: "horizon",
    title: "What's your investment horizon?",
    subtitle:
      "Recorded as a preference; model forecasts cover 21 trading sessions.",
    type: "cards",
    options: [
      {
        value: "< 1 year",
        icon: "⚡",
        label: "Short term",
        desc: "Less than 1 year",
      },
      {
        value: "1-5 years",
        icon: "📅",
        label: "Medium term",
        desc: "1 to 5 years",
      },
      {
        value: "5-10 years",
        icon: "🗓️",
        label: "Long term",
        desc: "5 to 10 years",
      },
      {
        value: "10+ years",
        icon: "🏦",
        label: "Very long term",
        desc: "More than 10 years",
      },
    ],
  },
  {
    id: "budget",
    title: "What's your starting budget?",
    subtitle:
      "This is the amount you'll invest initially. You can change it later.",
    type: "budget",
  },
];
function initOnboarding(newUserId, isNew) {
  userId = newUserId;
  if (!isNew) return;

  currentStep = 0;
  answers = {};
  importMode = false;
  importRows = [{ ticker: "", shares: "", price: "" }];
  showOnboarding();
}

// Open the onboarding popup directly in IMPORT mode (used by the "Import
// portfolio" buttons, which otherwise called initOnboarding(..., false) - a no-op).
function openImport(newUserId) {
  userId = newUserId;
  currentStep = 0;
  answers = { portfolio_type: "import" };
  importMode = true;
  importRows = [{ ticker: "", shares: "", price: "" }];
  showOnboarding();
}

function showOnboarding() {
  let overlay = document.getElementById("onboarding-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "onboarding-overlay";
    overlay.style.cssText =
      "position:fixed;inset:0;background:rgba(6,10,18,0.92);display:flex;align-items:center;justify-content:center;z-index:4000;backdrop-filter:blur(6px)";
    document.body.appendChild(overlay);
  }
  overlay.style.display = "flex";
  paintStep();
}

// Render the current onboarding question into the overlay panel
function paintStep() {
  const step = ONBOARDING_STEPS[currentStep];
  const overlay = document.getElementById("onboarding-overlay");
  const isLast = currentStep === ONBOARDING_STEPS.length - 1;
  const pct = Math.round((currentStep / ONBOARDING_STEPS.length) * 100);

  overlay.innerHTML = `
    <div style="background:var(--bg-card);border:1px solid var(--border-hi);border-radius:var(--r-xl);padding:36px 40px;width:560px;max-height:90vh;overflow-y:auto;box-shadow:var(--shadow-lg)">

      <!-- progress bar -->
      <div style="margin-bottom:28px">
        <div style="height:3px;background:var(--bg-hover);border-radius:2px;margin-bottom:8px">
          <div style="width:${pct}%;height:100%;background:var(--green);border-radius:2px;transition:width 0.3s"></div>
        </div>
        <div style="font-size:11px;color:var(--txt-3)">Step ${currentStep + 1} of ${ONBOARDING_STEPS.length}</div>
      </div>

      <!-- logo on first step -->
      ${
        currentStep === 0
          ? `
        <div style="text-align:center;margin-bottom:20px">
          <svg width="48" height="48" viewBox="0 0 28 28" fill="none">
            <rect width="28" height="28" rx="7" fill="#00C896" fill-opacity="0.15"/>
            <path d="M6 20L11 13L15 16L20 8" stroke="#00C896" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>`
          : ""
      }

      <div style="font-size:19px;font-weight:700;margin-bottom:6px">${step.title}</div>
      <div style="font-size:13.5px;color:var(--txt-2);margin-bottom:24px">${step.subtitle}</div>

      <!-- step content -->
      ${step.type === "choice" ? buildChoiceCards(step) : ""}
      ${step.type === "cards" ? buildOptionCards(step) : ""}
      ${step.type === "budget" ? buildBudgetForm() : ""}

      <!-- import form shown inline after choosing "import" -->
      ${step.type === "choice" && importMode ? buildImportForm() : ""}

      <!-- navigation -->
      <div style="display:flex;justify-content:space-between;margin-top:24px">
        ${currentStep > 0 ? `<button class="btn-secondary" onclick="stepBack()">Back</button>` : "<span></span>"}
        <button class="btn-primary" id="ob-next" onclick="stepForward()">
          ${isLast ? "Start investing" : "Continue"}
        </button>
      </div>
    </div>`;
}

// Build the large selectable cards used for portfolio type and training scope
function buildChoiceCards(step) {
  return `<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
    ${step.options
      .map(
        (o) => `
      <div class="ob-card ${answers[step.id] === o.value ? "ob-card-selected" : ""}"
           onclick="selectCard('${step.id}', '${o.value}', this)"
           style="background:var(--bg-2);border:2px solid ${answers[step.id] === o.value ? "var(--green)" : "var(--border)"};border-radius:var(--r-lg);padding:16px;cursor:pointer;transition:border-color 0.15s">
        <div style="font-size:24px;margin-bottom:8px">${o.icon}</div>
        <div style="font-weight:600;font-size:13.5px;margin-bottom:4px">${o.label}</div>
        <div style="font-size:12px;color:var(--txt-2)">${o.desc}</div>
      </div>`,
      )
      .join("")}
  </div>`;
}

// Build the smaller option cards used for risk, goal, and horizon
function buildOptionCards(step) {
  return `<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
    ${step.options
      .map(
        (o) => `
      <div class="ob-card ${answers[step.id] === o.value ? "ob-card-selected" : ""}"
           onclick="selectCard('${step.id}', '${o.value}', this)"
           style="background:var(--bg-2);border:2px solid ${answers[step.id] === o.value ? "var(--green)" : "var(--border)"};border-radius:var(--r-lg);padding:16px;cursor:pointer;transition:border-color 0.15s">
        <div style="font-size:22px;margin-bottom:8px">${o.icon}</div>
        <div style="font-weight:600;font-size:13.5px;margin-bottom:4px">${o.label}</div>
        <div style="font-size:12px;color:var(--txt-2)">${o.desc}</div>
      </div>`,
      )
      .join("")}
  </div>`;
}

// Render the budget and monthly contribution input fields
function buildBudgetForm() {
  return `
    <div style="display:flex;flex-direction:column;gap:16px">
      <div>
        <label class="field-label">Starting investment ($)</label>
        <input id="ob-budget" class="field-input" type="number" value="${answers.budget || 10000}" min="500" step="500"/>
      </div>
      <div>
        <label class="field-label">Monthly contribution ($) - optional</label>
        <input id="ob-monthly" class="field-input" type="number" value="${answers.monthly_contribution || 0}" min="0"/>
      </div>
    </div>`;
}

function buildImportForm() {
  const rows = importRows
    .map(
      (r, i) => `
    <tr>
      <td><input class="field-input import-ticker" data-row="${i}" type="text" placeholder="AAPL" value="${escapeHtml(r.ticker ?? "")}" style="max-width:80px;text-transform:uppercase"/></td>
      <td><input class="field-input import-shares" data-row="${i}" type="number" placeholder="10" value="${escapeHtml(r.shares ?? "")}" min="0" step="any" style="max-width:80px"/></td>
      <td><input class="field-input import-price"  data-row="${i}" type="number" placeholder="150.00" value="${escapeHtml(r.price ?? "")}" min="0" step="any" style="max-width:100px"/></td>
      <td>${i > 0 ? `<button onclick="removeImportRow(${i})" style="background:none;border:none;color:var(--red);cursor:pointer;font-size:16px">x</button>` : ""}</td>
    </tr>`,
    )
    .join("");

  return `
    <div style="margin-top:20px;border-top:1px solid var(--border);padding-top:18px">
      <div style="font-weight:600;font-size:13px;margin-bottom:10px">Your current holdings</div>
      <label>Load CSV or Excel (.xlsx): <input type="file" accept=".csv,.xlsx" onchange="previewHoldingsFile(this.files[0])"/></label>
      <p style="font-size:12px">Columns: ticker, shares, price (average purchase price). Review the rows before saving.</p>
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead>
          <tr style="color:var(--txt-3)">
            <th style="text-align:left;padding:4px 8px 8px 0">Ticker</th>
            <th style="text-align:left;padding:4px 8px 8px 0">Shares</th>
            <th style="text-align:left;padding:4px 8px 8px 0">Avg. price ($)</th>
            <th></th>
          </tr>
        </thead>
        <tbody id="import-rows">${rows}</tbody>
      </table>
      <button class="btn-secondary" style="margin-top:10px;font-size:11px" onclick="addImportRow()">
        + Add holding
      </button>
    </div>`;
}

function addImportRow() {
  captureImportRows();
  importRows.push({ ticker: "", shares: "", price: "" });
  paintStep();
}

function removeImportRow(index) {
  captureImportRows();
  importRows.splice(index, 1);
  paintStep();
}

function captureImportRows() {
  document.querySelectorAll(".import-ticker").forEach((el, i) => {
    if (!importRows[i]) importRows[i] = { ticker: "", shares: "", price: "" };
    importRows[i].ticker = el.value.trim().toUpperCase();
  });
  document.querySelectorAll(".import-shares").forEach((el, i) => {
    if (importRows[i]) importRows[i].shares = el.value;
  });
  document.querySelectorAll(".import-price").forEach((el, i) => {
    if (importRows[i]) importRows[i].price = el.value;
  });
}

function selectCard(stepId, value, el) {
  answers[stepId] = value;

  // Reflect the choice in import mode flag
  if (stepId === "portfolio_type") {
    importMode = value === "import";
    paintStep(); // repaint to show/hide import form
    return;
  }

  // Update card borders without a full repaint
  el.closest(".ob-card")
    ?.parentElement?.querySelectorAll(".ob-card")
    .forEach((c) => {
      c.style.borderColor = "var(--border)";
    });
  el.style.borderColor = "var(--green)";
}

// Validate the current step answer then advance to the next question
function stepForward() {
  const step = ONBOARDING_STEPS[currentStep];

  if ((step.type === "cards" || step.type === "choice") && !answers[step.id]) {
    toast("Please select an option", "error");
    return;
  }

  if (step.type === "choice" && importMode) {
    captureImportRows();
    const valid = importRows.filter(
      (r) => r.ticker && parseFloat(r.shares) > 0 && parseFloat(r.price) > 0,
    );
    if (!valid.length) {
      toast("Add at least one holding to import", "error");
      return;
    }
  }

  if (step.type === "budget") {
    answers.budget =
      parseFloat(document.getElementById("ob-budget")?.value) || 10000;
    answers.monthly_contribution =
      parseFloat(document.getElementById("ob-monthly")?.value) || 0;
  }

  if (currentStep >= ONBOARDING_STEPS.length - 1) {
    submitOnboarding();
    return;
  }

  currentStep++;
  paintStep();
}

// Go back to the previous step without clearing the current answer
function stepBack() {
  if (currentStep > 0) {
    currentStep--;
    paintStep();
  }
}

// Save the completed questionnaire to the backend and sync the sidebar
async function submitOnboarding() {
  const btn = document.getElementById("ob-next");
  if (btn) {
    btn.textContent = "Saving...";
    btn.disabled = true;
  }

  try {
    // 1. Save profile
    await api.createUser(userId, {
      risk_profile: answers.risk || "moderate",
      goal: answers.goal || "growth",
      investment_horizon: answers.horizon || "5-10 years",
      budget: answers.budget || 10000,
      monthly_contribution: answers.monthly_contribution || 0,
      market: "United States",
      index: "S&P 500",
    });

    // 2. If importing, send holdings to backend
    if (importMode) {
      const holdings = importRows
        .filter(
          (r) =>
            r.ticker && parseFloat(r.shares) > 0 && parseFloat(r.price) > 0,
        )
        .map((r) => ({
          ticker: r.ticker,
          shares: parseFloat(r.shares),
          price: parseFloat(r.price),
        }));

      if (holdings.length) {
        await api.importPortfolio(userId, holdings);
      }
    }

    // 3. Update sidebar state from onboarding answers
    state.risk = answers.risk || "moderate";
    state.budget = answers.budget || 10000;
    state.market = "United States";
    state.index = "S&P 500";

    const riskEl = document.getElementById("sb-risk");
    const budgetEl = document.getElementById("sb-budget");
    if (riskEl) riskEl.value = state.risk;
    if (budgetEl) budgetEl.value = state.budget;

    document.getElementById("onboarding-overlay").style.display = "none";

    const nextStep = importMode
      ? "Profile and holdings saved. Review recommendations in Portfolio."
      : "Profile saved. Review recommendations in Portfolio.";
    await reloadLastPortfolio(userId);
    navigateTo("portfolio");
    toast(nextStep, "success");
  } catch (e) {
    toast("Could not save profile - is the server running?", "error");
    if (btn) {
      btn.textContent = "Start investing";
      btn.disabled = false;
    }
  }
}

async function previewHoldingsFile(file) {
  if (!file) return;
  try {
    const response = await fetch(
      `${BASE}/api/import-preview?filename=${encodeURIComponent(file.name)}`,
      {
        method: "POST",
        body: file,
        headers: { "Content-Type": "application/octet-stream" },
      },
    );
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "Invalid file");
    importRows = result.holdings;
    paintStep();
    toast("Holdings loaded for review. They are not saved yet.", "info");
  } catch (error) {
    toast(error.message, "error");
  }
}
