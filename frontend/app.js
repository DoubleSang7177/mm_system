const statusEl = document.getElementById("status");
const pnlEl = document.getElementById("pnl");
const icEl = document.getElementById("ic");
const accEl = document.getElementById("accuracy");
const tradesEl = document.getElementById("trades");
const invEl = document.getElementById("inventory");
const feesEl = document.getElementById("fees");

const commonOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  plugins: {
    legend: { display: false },
  },
  scales: {
    x: {
      grid: { color: "#1f2a3a" },
      ticks: { color: "#8b9bb0", maxTicksLimit: 8 },
    },
    y: {
      grid: { color: "#1f2a3a" },
      ticks: { color: "#8b9bb0" },
    },
  },
};

const priceChart = new Chart(document.getElementById("priceChart").getContext("2d"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "Price",
        data: [],
        borderColor: "#3b82f6",
        backgroundColor: "rgba(59,130,246,0.2)",
        borderWidth: 2,
        pointRadius: 0,
      },
    ],
  },
  options: commonOptions,
});

const pnlChart = new Chart(document.getElementById("pnlChart").getContext("2d"), {
  type: "line",
  data: {
    labels: [],
    datasets: [
      {
        label: "PnL",
        data: [],
        borderColor: "#2ecc71",
        backgroundColor: "rgba(46,204,113,0.2)",
        borderWidth: 2,
        pointRadius: 0,
      },
    ],
  },
  options: commonOptions,
});

function toTimeLabel(ms) {
  const d = new Date(ms);
  return d.toLocaleTimeString();
}

function setPnLColor(value) {
  pnlEl.classList.remove("positive", "negative", "neutral");
  if (value > 0) pnlEl.classList.add("positive");
  else if (value < 0) pnlEl.classList.add("negative");
  else pnlEl.classList.add("neutral");
}

function setInventoryColor(value) {
  invEl.classList.remove("positive", "negative", "neutral");
  if (value > 0) invEl.classList.add("positive");
  else if (value < 0) invEl.classList.add("negative");
  else invEl.classList.add("neutral");
}

async function refreshMetrics() {
  try {
    const resp = await fetch("/metrics", { cache: "no-store" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    const pnl = Number(data.pnl || 0);
    const ic = Number(data.ic || 0);
    const acc = Number(data.accuracy || 0);
    const trades = Number(data.trades || 0);
    const inv = Number(data.inventory || 0);
    const fees = Number(data.fees || 0);

    pnlEl.textContent = pnl.toFixed(2);
    icEl.textContent = ic.toFixed(4);
    accEl.textContent = `${(acc * 100).toFixed(2)}%`;
    tradesEl.textContent = String(trades);
    invEl.textContent = inv.toFixed(4);
    feesEl.textContent = fees.toFixed(2);

    setPnLColor(pnl);
    setInventoryColor(inv);
    feesEl.classList.remove("positive", "neutral");
    feesEl.classList.add("negative");

    const priceSeries = Array.isArray(data.price_series) ? data.price_series : [];
    const pnlSeries = Array.isArray(data.pnl_series) ? data.pnl_series : [];

    priceChart.data.labels = priceSeries.map((p) => toTimeLabel(p.t));
    priceChart.data.datasets[0].data = priceSeries.map((p) => Number(p.v));
    priceChart.update();

    pnlChart.data.labels = pnlSeries.map((p) => toTimeLabel(p.t));
    pnlChart.data.datasets[0].data = pnlSeries.map((p) => Number(p.v));
    pnlChart.data.datasets[0].borderColor = pnl >= 0 ? "#2ecc71" : "#ff4d4f";
    pnlChart.data.datasets[0].backgroundColor =
      pnl >= 0 ? "rgba(46,204,113,0.2)" : "rgba(255,77,79,0.2)";
    pnlChart.update();

    statusEl.textContent = `Live · ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    statusEl.textContent = `Disconnected · ${new Date().toLocaleTimeString()}`;
    console.error(err);
  }
}

refreshMetrics();
setInterval(refreshMetrics, 1000);
