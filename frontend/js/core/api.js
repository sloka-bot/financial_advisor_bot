/* HTTP client for the FastAPI backend. */

const BASE =
  location.protocol === "file:" ? "http://localhost:8000" : location.origin;

async function apiRequest(path, method = "GET", body) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      ...(body === undefined
        ? {}
        : {
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }),
    });
  } catch {
    throw new Error(
      "The advisor is offline. Please start the app and try again.",
    );
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    let message =
      typeof data.detail === "string"
        ? data.detail
        : "Please check your entries and try again.";
    if (Array.isArray(data.detail))
      message =
        "Please check: " +
        data.detail.map((item) => item.loc.slice(1).join(" ")).join(", ") +
        ".";
    if (response.status >= 500)
      message =
        "The advisor could not finish this request. Please try again shortly.";
    if (
      /models? not trained|no predictions|no processed tickers/i.test(message)
    ) {
      message = "Advice is not ready yet. Your saved holdings are unchanged.";
    }
    const error = new Error(message);
    error.detail = message;
    error.status = response.status;
    throw error;
  }
  return data;
}

const get = (path) => apiRequest(path);
const post = (path, body) => apiRequest(path, "POST", body);
const put = (path, body) => apiRequest(path, "PUT", body);

const api = {
  markets: () => get("/api/markets"),
  modelStatus: () => get("/api/model-status"),
  pipelineStatus: () => get("/api/pipeline-status"),
  ollamaStatus: () => get("/api/ollama-status"),
  correlation: (userId) => get(`/api/user/${userId}/correlation`),
  regime: () => get("/api/regime"),

  stockHistory: (ticker, days = 120) =>
    get(`/api/stock/${ticker}/history?days=${days}`),
  analysis: (ticker, risk = "moderate", horizon = 21) =>
    get(`/api/analysis/${ticker}?risk_profile=${encodeURIComponent(risk)}&horizon=${horizon}`),

  /* User profile */
  getUser: (userId) => get(`/api/user/${userId}`),
  createUser: (userId, profile) => post(`/api/user/${userId}`, profile),
  updateUser: (userId, profile) => put(`/api/user/${userId}`, profile),
  getUserPortfolio: (userId) => get(`/api/user/${userId}/portfolio`),
  getUserRecs: (userId) => get(`/api/user/${userId}/recommendations`),
  importPortfolio: (userId, holdings) =>
    post(`/api/user/${userId}/import-portfolio`, { user_id: userId, holdings }),
  sellHolding: (userId, ticker) =>
    post(`/api/user/${userId}/sell`, { ticker }),
  buyMore: (userId, ticker, shares) => post(`/api/user/${userId}/buy`, { ticker, shares }),
  buildPortfolio: (userId) => post(`/api/user/${userId}/build`, {}),

  /* Recommendation approval */
  approveRec: (userId, recId) =>
    post("/api/recommendations/approve", { user_id: userId, rec_id: recId }),
  rejectRec: (userId, recId) =>
    post("/api/recommendations/reject", { user_id: userId, rec_id: recId }),

  /* Pipeline */
  runPipeline: (market, index, risk, budget, scope = "sample", uid = "") =>
    post("/api/run", {
      market,
      index,
      risk_profile: risk,
      budget,
      scope,
      user_id: uid,
    }),
  advisorRun: (userId) => post(`/api/advisor/run/${userId}`, {}),

  /* Analysis */
  recommend: (market, index, risk, n = 10) =>
    post("/api/recommend", { market, index, risk_profile: risk, top_n: n }),
  portfolio: (market, index, risk, budget, n = 10) =>
    post("/api/portfolio", {
      market,
      index,
      risk_profile: risk,
      budget,
      top_n: n,
      user_id: state.userId || "",
    }),
  backtest: (tickers, capital = 10000, mode = false) =>
    post("/api/backtest", { tickers, capital, portfolio_mode: mode }),

  /* Chat; mode is 'normal' or 'beginner' */
  chat: (message, portfolio, context, mode = "normal", userId = "dev-user") =>
    post("/api/chat", { message, portfolio, context, mode, user_id: userId }),

  runEvaluation: () => post("/api/evaluate-all", {}),

  avatarConfig: () => get("/api/avatar/config"),
  avatarSession: () => post("/api/avatar/session", {}),

  backtestPredict: (ticker, start_date, window) =>
    post("/api/backtest-predict", { ticker, start_date, window }),
  generateRecs: (userId) => post(`/api/recommendations/generate/${userId}`, {}),

  /* Date-range backtest and saved experiment results */
  backtestRange: (
    tickers,
    start_date,
    end_date,
    horizon = 21,
    feature_set = "both",
  ) =>
    post("/api/backtest-range", {
      tickers,
      start_date,
      end_date,
      horizon,
      feature_set,
    }),
  experimentResults: () => get("/api/experiment-results"),
  portfolioComparison: () => get("/api/portfolio-comparison"),
  riskPreview: (risk_profile, budget) =>
    post("/api/risk-preview", { risk_profile, budget }),

  /* Raw HTTP helpers */
  get: (path) => get(path),
  post: (path, body) => post(path, body),
};
