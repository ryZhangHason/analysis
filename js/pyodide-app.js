let pyodide;
let isPythonReady = false;
let currentRun = null;
let priceChart;
let compositeChart;
let strategyChart;

const STAGES = [
    "Fetch and validate data",
    "Build feature store",
    "Train model stack",
    "Run walk-forward validation",
    "Simulate execution",
    "Render research views",
];

const predictBtn = document.getElementById("predictBtn");
const stockSymbolInput = document.getElementById("stockSymbol");
const timePeriodSelect = document.getElementById("timePeriod");
const optimizeStrategyCheckbox = document.getElementById("optimizeStrategy");
const loadingSection = document.getElementById("loadingSection");
const loadingMessage = document.getElementById("loadingMessage");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");
const cacheStatus = document.getElementById("cacheStatus");
const stageList = document.getElementById("stageList");
const errorSection = document.getElementById("errorSection");
const errorMessage = document.getElementById("errorMessage");
const resultSection = document.getElementById("resultSection");

function renderStages(activeIndex = -1, complete = false) {
    stageList.innerHTML = "";
    STAGES.forEach((label, index) => {
        const item = document.createElement("div");
        item.className = "stage-item";
        if (complete || index < activeIndex) item.classList.add("is-done");
        if (!complete && index === activeIndex) item.classList.add("is-active");

        const status = complete || index < activeIndex ? "Done" : index === activeIndex ? "Running" : "Queued";
        item.innerHTML = `<span>${label}</span><strong>${status}</strong>`;
        stageList.appendChild(item);
    });
}

function updateProgress(percent, message, activeStage = -1) {
    progressBar.style.width = `${percent}%`;
    progressText.textContent = `${percent}%`;
    loadingMessage.textContent = message;
    renderStages(activeStage, percent >= 100);
}

function showError(message) {
    errorSection.classList.remove("hidden");
    errorMessage.textContent = message;
}

function hideError() {
    errorSection.classList.add("hidden");
    errorMessage.textContent = "";
}

function hideResults() {
    resultSection.classList.add("hidden");
}

function showResults() {
    resultSection.classList.remove("hidden");
}

function setCacheStatus(message) {
    cacheStatus.textContent = message;
}

function getCacheKey(symbol, period) {
    return `quant-workstation:${symbol}:${period}`;
}

function saveRunToCache(symbol, period, data) {
    try {
        localStorage.setItem(getCacheKey(symbol, period), JSON.stringify({
            savedAt: new Date().toISOString(),
            data,
        }));
        setCacheStatus("Cached in browser storage");
    } catch (error) {
        setCacheStatus("Cache unavailable");
    }
}

function readRunFromCache(symbol, period) {
    try {
        const raw = localStorage.getItem(getCacheKey(symbol, period));
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        setCacheStatus(`Loaded cached run from ${new Date(parsed.savedAt).toLocaleString()}`);
        return parsed.data;
    } catch (error) {
        setCacheStatus("Cache read failed");
        return null;
    }
}

function escapeForPython(value) {
    return String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

async function initPython() {
    try {
        predictBtn.disabled = true;
        predictBtn.textContent = "Loading Pyodide runtime...";
        loadingSection.classList.remove("hidden");
        updateProgress(5, "Loading Pyodide base runtime...", 0);

        pyodide = await loadPyodide({
            indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/",
        });

        updateProgress(20, "Loading scientific Python packages...", 0);
        await pyodide.loadPackage(["numpy", "pandas", "micropip", "scikit-learn"]);

        updateProgress(35, "Loading browser research modules...", 1);
        const moduleFiles = [
            "python/data_fetcher.py",
            "python/feature_engineering.py",
            "python/model.py",
            "python/strategy_optimizer.py",
            "python/research_workstation.py",
        ];

        for (const file of moduleFiles) {
            const code = await fetch(file).then((response) => {
                if (!response.ok) throw new Error(`Failed to load ${file}`);
                return response.text();
            });
            await pyodide.runPythonAsync(code);
        }

        isPythonReady = true;
        predictBtn.disabled = false;
        predictBtn.textContent = "Run Quant Research";
        updateProgress(100, "Pyodide research environment ready.", 5);
        setCacheStatus("Ready for first run");
    } catch (error) {
        predictBtn.textContent = "Runtime failed to load";
        showError(`Failed to initialize Pyodide research environment: ${error.message}`);
    }
}

async function runResearch(symbol, period, optimizeStrategy) {
    const pythonCode = `
result_json = await run_quant_research_async(
    symbol="${escapeForPython(symbol)}",
    period="${escapeForPython(period)}",
    optimize_strategy=${optimizeStrategy ? "True" : "False"}
)
result_json
`;
    const output = await pyodide.runPythonAsync(pythonCode);
    return JSON.parse(output);
}

function formatPct(value, digits = 2) {
    return `${Number(value).toFixed(digits)}%`;
}

function formatMaybePct(value, digits = 2) {
    return value === undefined || value === null ? "-" : formatPct(value, digits);
}

function renderKvGrid(targetId, rows) {
    const target = document.getElementById(targetId);
    target.innerHTML = rows.map((row) => `
        <div class="kv-item">
            <span>${row.label}</span>
            <strong>${row.value}</strong>
        </div>
    `).join("");
}

function renderTable(targetId, columns, rows) {
    const target = document.getElementById(targetId);
    if (!rows || !rows.length) {
        target.innerHTML = "<p class='muted'>No data available for this panel.</p>";
        return;
    }

    const head = columns.map((column) => `<th>${column.label}</th>`).join("");
    const body = rows.map((row) => `<tr>${columns.map((column) => `<td>${row[column.key] ?? "-"}</td>`).join("")}</tr>`).join("");
    target.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderNarrative(targetId, paragraphs) {
    const target = document.getElementById(targetId);
    target.innerHTML = paragraphs.map((text) => `<p>${text}</p>`).join("");
}

function destroyCharts() {
    [priceChart, compositeChart, strategyChart].forEach((chart) => {
        if (chart) chart.destroy();
    });
}

function renderPriceChart(charts) {
    const ctx = document.getElementById("priceChart").getContext("2d");
    priceChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: charts.price.dates,
            datasets: [
                {
                    label: "Close",
                    data: charts.price.close,
                    borderColor: "#39d0b6",
                    backgroundColor: "rgba(57, 208, 182, 0.1)",
                    borderWidth: 2.5,
                    tension: 0.25,
                },
                {
                    label: "MA20",
                    data: charts.price.ma20,
                    borderColor: "#f59e0b",
                    borderDash: [6, 4],
                    pointRadius: 0,
                    borderWidth: 1.5,
                    tension: 0.25,
                },
                {
                    label: "MA50",
                    data: charts.price.ma50,
                    borderColor: "#8b5cf6",
                    borderDash: [4, 4],
                    pointRadius: 0,
                    borderWidth: 1.5,
                    tension: 0.25,
                },
            ],
        },
        options: chartOptions("Price"),
    });
}

function renderCompositeChart(charts) {
    const ctx = document.getElementById("compositeChart").getContext("2d");
    compositeChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: charts.composite.dates,
            datasets: [
                {
                    label: "Composite Index",
                    data: charts.composite.values,
                    borderColor: "#1d7cd6",
                    backgroundColor: "rgba(29, 124, 214, 0.12)",
                    borderWidth: 2.5,
                    tension: 0.25,
                    fill: true,
                },
                {
                    label: "Long Entry",
                    data: charts.composite.long_entry,
                    borderColor: "#22c55e",
                    borderDash: [7, 4],
                    pointRadius: 0,
                    borderWidth: 1.5,
                },
                {
                    label: "Long Exit",
                    data: charts.composite.long_exit,
                    borderColor: "#84cc16",
                    borderDash: [5, 5],
                    pointRadius: 0,
                    borderWidth: 1.2,
                },
                {
                    label: "Short Entry",
                    data: charts.composite.short_entry,
                    borderColor: "#f97316",
                    borderDash: [7, 4],
                    pointRadius: 0,
                    borderWidth: 1.5,
                },
                {
                    label: "Short Exit",
                    data: charts.composite.short_exit,
                    borderColor: "#fb7185",
                    borderDash: [5, 5],
                    pointRadius: 0,
                    borderWidth: 1.2,
                },
            ],
        },
        options: chartOptions("Composite"),
    });
}

function renderStrategyChart(charts) {
    const ctx = document.getElementById("strategyChart").getContext("2d");
    strategyChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: charts.strategy.dates,
            datasets: [
                {
                    label: "Strategy",
                    data: charts.strategy.strategy,
                    borderColor: "#22c55e",
                    backgroundColor: "rgba(34, 197, 94, 0.12)",
                    borderWidth: 2.5,
                    fill: true,
                    tension: 0.25,
                },
                {
                    label: "Buy & Hold",
                    data: charts.strategy.buyhold,
                    borderColor: "#94a3b8",
                    borderDash: [6, 4],
                    pointRadius: 0,
                    borderWidth: 1.5,
                    tension: 0.25,
                },
                {
                    label: "Exposure Score",
                    data: charts.strategy.exposure.map((value) => value * 100),
                    borderColor: "#f59e0b",
                    borderDash: [3, 3],
                    pointRadius: 0,
                    borderWidth: 1.2,
                    tension: 0.15,
                },
            ],
        },
        options: chartOptions("Strategy"),
    });
}

function chartOptions(title) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: "#cfe0fb" },
            },
            title: {
                display: true,
                text: title,
                color: "#e5eefc",
            },
        },
        scales: {
            x: {
                ticks: { color: "#92a8c7" },
                grid: { color: "rgba(148,163,184,0.08)" },
            },
            y: {
                ticks: { color: "#92a8c7" },
                grid: { color: "rgba(148,163,184,0.08)" },
            },
        },
    };
}

function renderHeadlines(data) {
    document.getElementById("headlineRecommendation").textContent = data.summary.recommendation;
    document.getElementById("headlineMeta").textContent = `${data.summary.symbol} | Last close ${data.summary.last_close} | Daily change ${formatPct(data.summary.day_change_pct)}`;
    document.getElementById("headlineConviction").textContent = `${data.summary.composite_index}/100`;
    document.getElementById("headlineState").textContent = `${data.summary.target_state.toUpperCase()} bias | regime ${data.models.regime_classifier.current_regime}`;
    document.getElementById("headlineValidation").textContent = formatPct(data.validation.walk_forward.rolling_accuracy * 100);
    document.getElementById("headlineQuality").textContent = `Data quality ${data.data_quality.score}/100`;
    document.getElementById("headlineAlpha").textContent = formatPct(data.strategy.performance.alpha_vs_buy_hold);
    document.getElementById("headlineRisk").textContent = `${data.signals.risk_state} risk | max DD ${formatPct(data.strategy.performance.max_drawdown)}`;
}

function renderPanels(data) {
    renderKvGrid("overviewSnapshot", [
        { label: "Research Score", value: data.summary.research_score },
        { label: "Recommendation", value: data.summary.recommendation },
        { label: "Target State", value: data.strategy.latest_state.target_state.toUpperCase() },
        { label: "Conviction", value: formatPct(data.strategy.latest_state.conviction * 100) },
        { label: "Walk-Forward Accuracy", value: formatPct(data.validation.walk_forward.rolling_accuracy * 100) },
        { label: "Data Quality", value: `${data.data_quality.score}/100 (${data.data_quality.quality_state})` },
    ]);

    renderNarrative("overviewNotes", [
        data.explanations.overview,
        data.explanations.why_now,
        `Data issues: ${data.data_quality.issues.join("; ")}`,
    ]);

    renderKvGrid("regimeProbabilities", Object.entries(data.models.regime_classifier.probabilities).map(([key, value]) => ({
        label: key.replace(/_/g, " "),
        value: formatPct(value * 100),
    })));

    renderTable("regimeSlices", [
        { key: "regime", label: "Regime" },
        { key: "accuracy", label: "Accuracy" },
        { key: "sample_size", label: "Samples" },
    ], data.validation.regime_slices.map((row) => ({
        regime: row.regime,
        accuracy: formatPct(row.accuracy * 100),
        sample_size: row.sample_size,
    })));

    renderTable("alphaSleeves", [
        { key: "name", label: "Sleeve" },
        { key: "direction", label: "Direction" },
        { key: "score", label: "Score" },
        { key: "strength", label: "Strength" },
        { key: "decay", label: "Decay" },
    ], data.signals.alpha_sleeves);

    const bullish = data.signals.alpha_sleeves.filter((item) => item.direction === "bullish").map((item) => item.name);
    const bearish = data.signals.alpha_sleeves.filter((item) => item.direction === "bearish").map((item) => item.name);
    renderNarrative("alphaConflict", [
        bullish.length ? `Bullish sleeves: ${bullish.join(", ")}.` : "No dominant bullish sleeves.",
        bearish.length ? `Bearish sleeves: ${bearish.join(", ")}.` : "No dominant bearish sleeves.",
        "The conflict map is derived from sleeve direction, strength, and current regime fit.",
    ]);

    renderTable("modelVotes", [
        { key: "name", label: "Model" },
        { key: "vote", label: "Vote" },
        { key: "prob", label: "Prob. Up" },
        { key: "acc", label: "Test Accuracy" },
    ], Object.entries(data.models.base_models).map(([name, model]) => ({
        name,
        vote: model.latest_vote,
        prob: formatPct(model.latest_probability_up * 100),
        acc: formatPct(model.test_accuracy * 100),
    })));

    renderTable("featureImportance", [
        { key: "feature", label: "Feature" },
        { key: "importance", label: "Importance" },
    ], data.models.feature_importance);

    renderKvGrid("strategySummary", [
        { label: "Strategy Return", value: formatPct(data.strategy.performance.strategy_total_return) },
        { label: "Buy & Hold", value: formatPct(data.strategy.performance.buy_hold_total_return) },
        { label: "Alpha vs Buy & Hold", value: formatPct(data.strategy.performance.alpha_vs_buy_hold) },
        { label: "Max Drawdown", value: formatPct(data.strategy.performance.max_drawdown) },
        { label: "Turnover", value: data.strategy.performance.turnover },
        { label: "Trade Count", value: data.strategy.performance.trade_count },
        { label: "Win Rate", value: formatPct(data.strategy.performance.win_rate) },
        { label: "Payoff Ratio", value: data.strategy.performance.payoff_ratio },
    ]);

    renderTable("tradeLog", [
        { key: "entry_date", label: "Entry" },
        { key: "exit_date", label: "Exit" },
        { key: "side", label: "Side" },
        { key: "return_pct", label: "Return %" },
        { key: "holding_days", label: "Days" },
    ], data.strategy.trade_log);

    renderKvGrid("riskSummary", [
        { label: "Tail Loss (5%)", value: formatPct(data.strategy.performance.tail_loss_5pct) },
        { label: "Gross Exposure", value: data.strategy.performance.gross_exposure },
        { label: "Expected Horizon", value: `${data.strategy.latest_state.expected_horizon_days} days` },
        { label: "Dominant Model", value: data.strategy.diagnostics.dominant_model },
        { label: "Execution Mode", value: data.strategy.execution_config.mode },
        { label: "Risk Flags", value: data.strategy.latest_state.risk_flags.join(" | ") },
    ]);

    renderTable("calibrationTable", [
        { key: "bucket", label: "Bucket" },
        { key: "avg_probability_up", label: "Avg Prob Up" },
        { key: "actual_up_rate", label: "Actual Up Rate" },
        { key: "count", label: "Count" },
    ], data.validation.calibration);

    renderNarrative("explainWhy", [
        data.explanations.overview,
        data.explanations.why_now,
        data.explanations.risk,
    ]);

    renderNarrative("explainExecution", [
        data.explanations.execution,
        `Current entry / exit rationale: ${data.strategy.latest_state.entry_exit_rationale}.`,
        `Confidence gating bands are long entry ${formatPct(data.signals.confidence_bands.long_entry * 100, 0)} and short entry ${formatPct(data.signals.confidence_bands.short_entry * 100, 0)}.`,
    ]);
}

function displayResearch(data) {
    currentRun = data;
    renderHeadlines(data);
    renderPanels(data);
    destroyCharts();
    renderPriceChart(data.charts);
    renderCompositeChart(data.charts);
    renderStrategyChart(data.charts);
    showResults();
}

async function handlePredict() {
    const symbol = stockSymbolInput.value.trim().toUpperCase();
    const period = timePeriodSelect.value;
    const optimizeStrategy = optimizeStrategyCheckbox.checked;

    if (!symbol) {
        showError("Please enter a stock symbol.");
        return;
    }

    if (!isPythonReady) {
        showError("Pyodide is still initializing. Please wait a moment and try again.");
        return;
    }

    hideError();
    hideResults();
    loadingSection.classList.remove("hidden");
    predictBtn.disabled = true;

    const cached = readRunFromCache(symbol, period);
    if (cached) {
        displayResearch(cached);
    }

    try {
        updateProgress(8, `Stage 1/6: validating ${symbol} market data...`, 0);
        updateProgress(20, "Stage 2/6: building browser feature store...", 1);
        updateProgress(38, "Stage 3/6: training the model stack...", 2);

        const data = await runResearch(symbol, period, optimizeStrategy);

        updateProgress(62, "Stage 4/6: running walk-forward validation...", 3);
        updateProgress(82, "Stage 5/6: simulating execution layer...", 4);
        updateProgress(100, "Stage 6/6: rendering research workstation...", 5);

        saveRunToCache(symbol, period, data);
        displayResearch(data);
    } catch (error) {
        showError(error.message || "Research pipeline failed.");
    } finally {
        predictBtn.disabled = false;
    }
}

function setupTabs() {
    const buttons = Array.from(document.querySelectorAll(".tab-btn"));
    const panels = Array.from(document.querySelectorAll(".tab-panel"));

    buttons.forEach((button) => {
        button.addEventListener("click", () => {
            const tab = button.dataset.tab;
            buttons.forEach((btn) => btn.classList.toggle("active", btn === button));
            panels.forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === tab));
        });
    });
}

predictBtn.addEventListener("click", handlePredict);
stockSymbolInput.addEventListener("keypress", (event) => {
    if (event.key === "Enter") {
        handlePredict();
    }
});

setupTabs();
renderStages(-1, false);
initPython();
