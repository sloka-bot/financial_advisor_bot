/* Shared Chart.js configuration and workspace charts. */

if (typeof Chart !== "undefined") {
  Chart.defaults.color = "#b2c3d9";
  Chart.defaults.borderColor = "rgba(164,187,216,0.14)";
  Chart.defaults.font.family =
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  Chart.defaults.font.size = 11;
}

/* Colour palette matching the CSS design tokens */
const C = {
  green: "#73dcc8",
  red: "#F43F5E",
  purple: "#7C3AED",
  blue: "#3B82F6",
  amber: "#F59E0B",
  teal: "#06B6D4",
  greenFill: "rgba(0,200,150,0.10)",
  purpleFill: "rgba(124,58,237,0.12)",
  blueFill: "rgba(59,130,246,0.10)",
  grid: "rgba(164,187,216,0.14)",
  tooltip: "#0B1220",
};

/* Replace any existing chart on the canvas. */
function drawChart(id, config) {
  if (typeof Chart === "undefined") return null;
  const ex = Chart.getChart(id);
  if (ex) ex.destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return null;
  return new Chart(ctx, config);
}

/* Shared tooltip style */
const TOOLTIP = {
  backgroundColor: C.tooltip,
  borderColor: "rgba(255,255,255,0.10)",
  borderWidth: 1,
  padding: 10,
  cornerRadius: 8,
};

/* Shared dark-theme axis style */
function darkAxis(overrides = {}) {
  return {
    grid: { color: C.grid },
    ticks: { color: "#90a7c2", font: { size: 11 } },
    ...overrides,
  };
}

/* Portfolio growth line chart */
function allocationChart(id, labels, values) {
  const palette = [
    C.green,
    C.purple,
    C.blue,
    C.amber,
    C.red,
    C.teal,
    "#EC4899",
    "#84CC16",
    "#F97316",
    "#8B5CF6",
  ];
  return drawChart(id, {
    type: "doughnut",
    data: {
      labels,
      datasets: [
        {
          data: values,
          backgroundColor: palette.slice(0, values.length),
          borderColor: "#0B1220",
          borderWidth: 3,
          hoverOffset: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "70%",
      plugins: {
        legend: {
          position: "right",
          labels: { boxWidth: 10, padding: 14, usePointStyle: true },
        },
        tooltip: {
          ...TOOLTIP,
          callbacks: {
            label: (ctx) => ` ${ctx.label}: $${ctx.parsed.toLocaleString()}`,
          },
        },
      },
    },
  });
}

/* Top performers bar chart */
function compoundChart(id, labels, flat, compounded) {
  return drawChart(id, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Compounded",
          data: compounded,
          borderColor: C.green,
          backgroundColor: C.greenFill,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: "No return",
          data: flat,
          borderColor: C.purple,
          backgroundColor: "transparent",
          borderDash: [5, 4],
          tension: 0,
          pointRadius: 0,
          borderWidth: 1.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
          align: "end",
          labels: { boxWidth: 10, usePointStyle: true },
        },
        tooltip: {
          ...TOOLTIP,
          callbacks: {
            label: (ctx) =>
              ` $${ctx.parsed.y.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
          },
        },
      },
      scales: {
        x: darkAxis({ ticks: { maxTicksLimit: 10, color: "#90a7c2" } }),
        y: {
          ...darkAxis(),
          ticks: {
            ...darkAxis().ticks,
            callback: (v) =>
              "$" + (v >= 1000 ? (v / 1000).toFixed(0) + "k" : v),
          },
        },
      },
    },
  });
}

/* Monte Carlo fan chart */
function monteCarloChart(id, labels, pcts) {
  return drawChart(id, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "90th percentile",
          data: pcts.p90,
          borderColor: C.green,
          backgroundColor: "transparent",
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 1.5,
        },
        {
          label: "Median (50th)",
          data: pcts.p50,
          borderColor: C.blue,
          backgroundColor: C.blueFill,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: "10th percentile",
          data: pcts.p10,
          borderColor: C.red,
          backgroundColor: "transparent",
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 1.5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          align: "end",
          labels: { boxWidth: 10, usePointStyle: true },
        },
        tooltip: {
          ...TOOLTIP,
          callbacks: {
            label: (ctx) =>
              ` $${ctx.parsed.y.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
          },
        },
      },
      scales: {
        x: darkAxis({ ticks: { maxTicksLimit: 8, color: "#90a7c2" } }),
        y: {
          ...darkAxis(),
          ticks: {
            ...darkAxis().ticks,
            callback: (v) =>
              "$" + (v >= 1000 ? (v / 1000).toFixed(0) + "k" : v),
          },
        },
      },
    },
  });
}

/* Monte Carlo outcome histogram */
function histogramChart(id, buckets, counts) {
  return drawChart(id, {
    type: "bar",
    data: {
      labels: buckets.map((b) => "$" + (b / 1000).toFixed(0) + "k"),
      datasets: [
        {
          data: counts,
          backgroundColor: C.blue + "88",
          borderColor: C.blue,
          borderWidth: 1,
          borderRadius: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          ...TOOLTIP,
          callbacks: {
            title: ([ctx]) =>
              "Final value: " + buckets[ctx.dataIndex].toLocaleString(),
            label: (ctx) => ` ${ctx.parsed.y} simulations`,
          },
        },
      },
      scales: {
        x: darkAxis({ ticks: { color: "#90a7c2", maxTicksLimit: 10 } }),
        y: darkAxis({ ticks: { color: "#90a7c2" } }),
      },
    },
  });
}

/* Price chart with SMA overlay */
function efficientFrontierChart(id, points, optimal) {
  return drawChart(id, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Random portfolios",
          data: points,
          backgroundColor: points.map((p) => {
            const s = (p.sharpe - 0) / 2;
            const r = Math.round(255 * (1 - s));
            const g = Math.round(200 * s);
            return `rgba(${r},${g},150,0.4)`;
          }),
          pointRadius: 3,
          pointHoverRadius: 5,
        },
        {
          label: "Max Sharpe",
          data: [optimal],
          backgroundColor: C.green,
          pointRadius: 8,
          pointStyle: "star",
          borderColor: "#000",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "top",
          labels: { boxWidth: 10, usePointStyle: true },
        },
        tooltip: {
          ...TOOLTIP,
          callbacks: {
            label: (ctx) => [
              `Vol: ${(ctx.parsed.x * 100).toFixed(1)}%`,
              `Ret: ${(ctx.parsed.y * 100).toFixed(1)}%`,
            ],
          },
        },
      },
      scales: {
        x: {
          ...darkAxis(),
          title: {
            display: true,
            text: "Annualised Volatility",
            color: "#90a7c2",
            font: { size: 11 },
          },
          ticks: {
            ...darkAxis().ticks,
            callback: (v) => (v * 100).toFixed(0) + "%",
          },
        },
        y: {
          ...darkAxis(),
          title: {
            display: true,
            text: "Annualised Return",
            color: "#90a7c2",
            font: { size: 11 },
          },
          ticks: {
            ...darkAxis().ticks,
            callback: (v) => (v * 100).toFixed(0) + "%",
          },
        },
      },
    },
  });
}

/* Backtest equity curves */
/* Technical-indicator charts */
function priceChart(id, labels, closes, sma20, sma50, bbUpper, bbLower) {
  const datasets = [
    { label: "Close", data: closes, borderColor: "#F1F5F9", borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false, order: 1 },
  ];
  if (sma20.some((v) => v != null)) datasets.push({ label: "SMA 20", data: sma20, borderColor: C.amber, borderWidth: 1.2, borderDash: [3, 2], pointRadius: 0, tension: 0.2, fill: false, order: 2 });
  if (sma50.some((v) => v != null)) datasets.push({ label: "SMA 50", data: sma50, borderColor: C.blue, borderWidth: 1.2, borderDash: [5, 3], pointRadius: 0, tension: 0.2, fill: false, order: 3 });
  if (bbUpper && bbUpper.some((v) => v != null)) {
    datasets.push({ label: "BB Upper", data: bbUpper, borderColor: "rgba(255,255,255,0.18)", borderWidth: 0.8, borderDash: [2, 3], pointRadius: 0, fill: false, tension: 0.2, order: 4 });
    datasets.push({ label: "BB Lower", data: bbLower, borderColor: "rgba(255,255,255,0.18)", borderWidth: 0.8, borderDash: [2, 3], pointRadius: 0, fill: { target: "-1", above: "rgba(255,255,255,0.02)" }, tension: 0.2, order: 5 });
  }
  return drawChart(id, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "top", align: "end", labels: { boxWidth: 10, usePointStyle: true, padding: 14 } },
        tooltip: { ...TOOLTIP, callbacks: { label: (ctx) => ` ${ctx.dataset.label}: $${ctx.parsed.y?.toFixed(2) ?? "--"}` } },
      },
      scales: {
        x: darkAxis({ ticks: { maxTicksLimit: 8, color: "#90a7c2" } }),
        y: { ...darkAxis(), ticks: { ...darkAxis().ticks, callback: (v) => "$" + v.toFixed(0) } },
      },
    },
  });
}

function rsiChart(id, labels, rsiValues) {
  const overbought = Array(labels.length).fill(70);
  const oversold = Array(labels.length).fill(30);
  return drawChart(id, {
    type: "line",
    data: { labels, datasets: [
      { label: "RSI 14", data: rsiValues, borderColor: C.teal, borderWidth: 1.5, pointRadius: 0, tension: 0.2, fill: false },
      { label: "Overbought (70)", data: overbought, borderColor: "rgba(244,63,94,0.4)", borderWidth: 1, borderDash: [4, 3], pointRadius: 0, fill: false },
      { label: "Oversold (30)", data: oversold, borderColor: "rgba(0,200,150,0.4)", borderWidth: 1, borderDash: [4, 3], pointRadius: 0, fill: false },
    ] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { ...TOOLTIP, callbacks: { label: (ctx) => (ctx.dataset.label === "RSI 14" ? ` RSI: ${ctx.parsed.y?.toFixed(1)}` : null) } } },
      scales: { x: darkAxis({ ticks: { maxTicksLimit: 8, color: "#90a7c2" } }), y: { ...darkAxis(), min: 0, max: 100, ticks: { ...darkAxis().ticks, stepSize: 25 } } },
    },
  });
}

function macdChart(id, labels, macd, signal, hist) {
  return drawChart(id, {
    type: "bar",
    data: { labels, datasets: [
      { type: "bar", label: "Histogram", data: hist, backgroundColor: hist.map((v) => (v >= 0 ? "rgba(0,200,150,0.5)" : "rgba(244,63,94,0.5)")), borderColor: hist.map((v) => (v >= 0 ? C.green : C.red)), borderWidth: 1, order: 3 },
      { type: "line", label: "MACD", data: macd, borderColor: C.blue, borderWidth: 1.5, pointRadius: 0, tension: 0.2, fill: false, order: 1 },
      { type: "line", label: "Signal", data: signal, borderColor: C.amber, borderWidth: 1.2, borderDash: [3, 2], pointRadius: 0, tension: 0.2, fill: false, order: 2 },
    ] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "top", align: "end", labels: { boxWidth: 10, usePointStyle: true } }, tooltip: { ...TOOLTIP } },
      scales: { x: darkAxis({ ticks: { maxTicksLimit: 8, color: "#90a7c2" } }), y: darkAxis() },
    },
  });
}

function volumeChart(id, labels, volumes, volRatios) {
  const colours = volRatios ? volRatios.map((r) => (r > 1.5 ? "rgba(0,200,150,0.5)" : "rgba(71,85,105,0.5)")) : Array(volumes.length).fill("rgba(71,85,105,0.5)");
  return drawChart(id, {
    type: "bar",
    data: { labels, datasets: [{ label: "Volume", data: volumes, backgroundColor: colours, borderRadius: 2, borderWidth: 0 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { ...TOOLTIP, callbacks: { label: (ctx) => ` Vol: ${(ctx.parsed.y / 1e6).toFixed(2)}M` } } },
      scales: { x: darkAxis({ ticks: { maxTicksLimit: 8, color: "#90a7c2" } }), y: { ...darkAxis(), ticks: { ...darkAxis().ticks, callback: (v) => (v / 1e6).toFixed(0) + "M" } } },
    },
  });
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
