const BASE = 'http://localhost:8000';

function get(path) {
  return fetch(`${BASE}${path}`)
    .then(r => { if (!r.ok) return r.json().then(e => Promise.reject(e)); return r.json(); });
}
function post(path, body) {
  return fetch(`${BASE}${path}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }).then(r => { if (!r.ok) return r.json().then(e => Promise.reject(e)); return r.json(); });
}
function put(path, body) {
  return fetch(`${BASE}${path}`, {
    method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  }).then(r => { if (!r.ok) return r.json().then(e => Promise.reject(e)); return r.json(); });
}

const api = {
  config:            ()                        => get('/api/config'),
  markets:           ()                        => get('/api/markets'),
  modelStatus:       ()                        => get('/api/model-status'),
  pipelineStatus:    ()                        => get('/api/pipeline-status'),
  ollamaStatus:      ()                        => get('/api/ollama-status'),
  validationResults: ()                        => get('/api/validation-results'),
  regime:            ()                        => get('/api/regime'),

  stockHistory:  (ticker, days = 120)          => get(`/api/stock/${ticker}/history?days=${days}`),
  analysis:      ticker                        => get(`/api/analysis/${ticker}`),
  evaluate:      ticker                        => get(`/api/evaluate/${ticker}`),

  /* user profile */
  getUser:       userId                        => get(`/api/user/${userId}`),
  createUser:    (userId, profile)             => post(`/api/user/${userId}`, profile),
  updateUser:    (userId, profile)             => put(`/api/user/${userId}`, profile),
  getUserPortfolio: userId                     => get(`/api/user/${userId}/portfolio`),
  getUserRecs:   userId                        => get(`/api/user/${userId}/recommendations`),

  /* recommendations approval */
  approveRec:    (userId, recId)               => post('/api/recommendations/approve', { user_id: userId, rec_id: recId }),
  rejectRec:     (userId, recId)               => post('/api/recommendations/reject',  { user_id: userId, rec_id: recId }),

  /* pipeline */
  runPipeline:   (market, index, risk, budget, scope = 'sample', userId = '') =>
    post('/api/run', { market, index, risk_profile: risk, budget, scope, user_id: userId }),
  advisorRun:    userId                        => post(`/api/advisor/run/${userId}`, {}),

  /* analysis */
  recommend:     (market, index, risk, n = 10) =>
    post('/api/recommend', { market, index, risk_profile: risk, top_n: n }),
  portfolio:     (market, index, risk, budget, n = 10) =>
    post('/api/portfolio', { market, index, risk_profile: risk, budget, top_n: n }),
  backtest:      (tickers, capital = 10000, mode = false) =>
    post('/api/backtest', { tickers, capital, portfolio_mode: mode }),

  chat:          (message, portfolio, context) =>
    post('/api/chat', { message, portfolio, context }),

  runEvaluation: ()                            => post('/api/evaluate-all', {}),
};
