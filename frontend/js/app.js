/* Navigation and saved-account state for the five-page workspace. */
const PAGES = {
  dashboard: ["Dashboard", () => renderDashboard()],
  portfolio: ["Portfolio", () => renderPortfolio()],
  predictor: ["Stock Predictor", () => renderPredictor()],
  chatbot: ["Chatbot", () => renderChatbot()],
  tools: ["Tools", () => renderTools()],
};

document.addEventListener("DOMContentLoaded", async () => {
  document
    .querySelectorAll(".nav-item[data-page]")
    .forEach((button) =>
      button.addEventListener("click", () => navigateTo(button.dataset.page)),
    );
  await loadUserSession("dev-user");
});

async function loadUserSession(userId) {
  state.userId = userId;
  navigateTo("dashboard");
  refreshModelStatus();
  try {
    const profile = await api.getUser(userId);
    if (profile.is_new) {
      initOnboarding(userId, true);
      return;
    }
    state.risk = profile.risk_profile || "moderate";
    state.budget = profile.budget || 10000;
    document.getElementById("sb-risk").value = state.risk;
    document.getElementById("sb-budget").value = state.budget;
    await reloadLastPortfolio(userId);
    renderDashboard();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function reloadLastPortfolio(userId) {
  try {
    const saved = await api.getUserPortfolio(userId);
    state.portfolioData = { portfolio: saved };
  } catch (error) {
    toast(error.message, "error");
  }
}

function navigateTo(page) {
  if (!PAGES[page]) page = "dashboard";
  if (state.currentPage === "chatbot" && page !== "chatbot")
    stopAvatarSession();
  state.currentPage = page;
  document
    .querySelectorAll(".view")
    .forEach((view) =>
      view.classList.toggle("active", view.id === `view-${page}`),
    );
  document.querySelectorAll(".nav-item[data-page]").forEach((button) => {
    button.classList.toggle("active", button.dataset.page === page);
    if (button.dataset.page === page)
      button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  setText("page-title", PAGES[page][0]);
  PAGES[page][1]();
}

async function savePreferences() {
  const budget = Number(document.getElementById("sb-budget").value);
  if (!Number.isFinite(budget) || budget < 100) {
    toast("Enter a budget of at least $100.", "error");
    return;
  }
  try {
    const risk = document.getElementById("sb-risk").value;
    await api.updateUser(state.userId, { risk_profile: risk, budget });
    state.risk = risk;
    state.budget = budget;
    await reloadLastPortfolio(state.userId);
    PAGES[state.currentPage][1]();
    toast("Preferences saved. Holdings change only after approval.", "success");
  } catch (error) {
    toast(error.message, "error");
  }
}

async function refreshAdvice() {
  navigateTo("portfolio");
  await generateNewRecs();
}

async function refreshModelStatus() {
  try {
    const status = await api.modelStatus();
    state.modelsReady = status.xgboost_trained && status.lstm_trained;
    document
      .getElementById("model-dot")
      ?.classList.toggle("trained", state.modelsReady);
    setText(
      "model-label",
      state.modelsReady
        ? "Forecast models ready"
        : "Some forecasts unavailable",
    );
  } catch {
    setText("model-label", "Connection unavailable");
  }
}

function toast(message, type = "info") {
  const element = document.getElementById("toast");
  if (!element) return;
  element.textContent = message;
  element.className = `toast ${type}`;
  clearTimeout(element._timer);
  element._timer = setTimeout(() => element.classList.add("hidden"), 6000);
}
