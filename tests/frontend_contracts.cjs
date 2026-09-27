const vm = require("node:vm");
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");
const elements = {};
const context = {
  state: {
    risk: "moderate",
    portfolioData: {
      portfolio: {
        holdings: [],
        cash_remaining: 1000,
        total_invested: 0,
        total_value: 1000,
      },
    },
  },
  document: {
    getElementById: (id) =>
      (elements[id] ??= { style: {}, innerHTML: "", textContent: "" }),
  },
  setText: (id, text) => ((elements[id] ??= {}).textContent = text),
  safeChart: () => null,
  allocationChart: () => null,
};
vm.createContext(context);
for (const file of [
  "core/ui",
  "pages/portfolio",
  "pages/dashboard",
  "pages/predictor",
  "components/onboarding",
]) {
  vm.runInContext(fs.readFileSync(`frontend/js/${file}.js`, "utf8"), context);
}
assert.equal(
  vm.runInContext("predictedReturn({predicted_return:2})", context),
  2,
);
assert.equal(
  vm.runInContext("predictedReturn({predicted_return:null})", context),
  null,
);
assert.equal(vm.runInContext("forecastPrice(100,2)", context), 102);
assert.equal(vm.runInContext("forecastPrice(100,null)", context), null);
assert.equal(vm.runInContext("forecastPrice(0,2)", context), null);
assert.equal(vm.runInContext("money(null)", context), "Unavailable");
assert.equal(vm.runInContext("money(0)", context), "$0.00");
vm.runInContext("renderDashboard()", context);
assert(elements["view-dashboard"].innerHTML.includes("$1,000.00"));
assert(elements["view-dashboard"].innerHTML.includes("No saved positions"));
assert(!elements["view-dashboard"].innerHTML.includes("Projected over"));
vm.runInContext("renderHoldings([])", context);
assert(elements["holdings-wrap"].innerHTML.includes("No saved holdings"));
vm.runInContext(
  `importRows=[{ticker:'" onfocus="alert(1)<',shares:'1" autofocus',price:'2" onfocus="alert(1)'}]`,
  context,
);
const imported = vm.runInContext("buildImportForm()", context);
assert(imported.includes("&quot; onfocus=&quot;alert(1)&lt;"));
assert(!imported.includes('value="" onfocus='));
assert(!imported.includes('value="1" autofocus'));
vm.runInContext(
  "drawComparison({ticker:'TEST',trained_until:'2019-12-01',rows:[{date:'2020-01-02',target_date:'2020-02-03',known:true,predicted_price:102,actual_price:103}]})",
  context,
);
assert(elements["comparison-result"].innerHTML.includes("2020-02-03"));
assert(elements["comparison-result"].innerHTML.includes("$1.00"));
const html = fs.readFileSync("frontend/index.html", "utf8");
assert.deepEqual(
  [...html.matchAll(/data-page="([^"]+)"/g)].map((m) => m[1]),
  ["dashboard", "portfolio", "predictor", "chatbot", "tools"],
);
for (const match of html.matchAll(
  /(?:src|href)="((?:js|css)\/[^"?]+)(?:\?[^"]*)?"/g,
))
  assert(
    fs.existsSync(path.join("frontend", match[1])),
    `Missing asset: ${match[1]}`,
  );
const app = fs.readFileSync("frontend/js/app.js", "utf8");
assert(/state\.portfolioData\s*=\s*\{\s*portfolio:\s*saved\s*\}/.test(app));
assert(!app.includes("state.portfolioData = profile.last_recommendation"));
console.log(
  "Frontend contracts passed: five pages, local assets, saved holdings, forecast units, missing values, target dates, import escaping.",
);
vm.runInContext(fs.readFileSync("frontend/js/pages/tools.js", "utf8"), context);
elements["ci-rate"] = { value: "0" };
assert.equal(vm.runInContext("toolNumber('ci-rate',10)", context), 0);
assert(Number.isFinite(vm.runInContext("randn()", context)));
console.log(
  "Calculator contracts passed: valid zero inputs and finite simulation draws.",
);
