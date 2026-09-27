/*
 * state.js
 *
 * Global application state shared across all tabs.
 * Any variable that needs to survive a tab switch lives here.
 */

const state = {
  market: "United States",
  index: "S&P 500",
  risk: "moderate",
  budget: 10000,
  userId: "dev-user" /* single local user (no auth) */,
  modelsReady: false,
  portfolioData: null,
  currentPage: "dashboard",
};
