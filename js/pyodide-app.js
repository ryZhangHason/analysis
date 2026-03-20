let pyodide;
let isPythonReady = false;
let currentRun = null;
let currentLanguage = localStorage.getItem("quant-workstation-language") || "en";
let lastRunSummary = null;
let priceChart;
let compositeChart;
let strategyChart;

const LOCALES = {
    en: {
        hero_eyebrow: "Browser-First Pyodide Quant Lab",
        hero_title: "Quant Research Workstation",
        hero_copy: "Single-symbol institutional-style research built on the current Yahoo-derived fetch workflow. No server required.",
        language_label: "Language",
        hero_stat_pipeline: "Pipeline",
        hero_stat_pipeline_value: "6 Stages",
        hero_stat_scope: "Data Scope",
        hero_stat_scope_value: "OHLCV + events",
        hero_stat_execution: "Execution",
        hero_stat_execution_value: "Pyodide In-Browser",
        research_run: "Research Run",
        research_run_copy: "Fetch once, build a feature store, rank candidate strategies, and promote one best current policy.",
        field_symbol: "Symbol",
        field_symbol_placeholder: "e.g. AAPL, MSFT, NVDA",
        field_period: "Period",
        field_execution: "Execution Layer",
        field_execution_toggle: "Enable meta-learner strategy search",
        field_mode: "Run Mode",
        mode_quick: "Quick research",
        mode_full: "Full ranking",
        btn_initialize: "Initialize Research Pipeline",
        btn_run: "Run Quant Research",
        btn_loading: "Loading Pyodide runtime...",
        btn_failed: "Runtime failed to load",
        loading_title: "Running staged research pipeline",
        loading_prepare: "Preparing Pyodide environment...",
        debug_title: "Startup Debug",
        error_title: "Research Pipeline Error",
        headline_recommendation: "Latest Recommendation",
        headline_composite: "Composite Conviction",
        headline_walkforward: "Walk-Forward Accuracy",
        headline_alpha: "Strategy Alpha",
        tab_overview: "Overview",
        tab_regime: "Regime Lab",
        tab_alpha: "Alpha Lab",
        tab_models: "Model Lab",
        tab_strategy: "Strategy Lab",
        tab_risk: "Risk Lab",
        tab_explain: "Explainability",
        status_done: "done",
        status_running: "running",
        status_queued: "queued",
        debug_waiting: "waiting",
        debug_booting: "booting",
        debug_ready: "ready",
        debug_failed: "failed",
        debug_running_research: "running research",
        debug_research_ready: "research ready",
        debug_research_failed: "research failed",
        cache_none: "No cached run",
        cache_ready: "Ready for first run",
        cache_cached: "Cached in browser storage",
        cache_failed: "Cache read failed",
        cache_loaded: "Loaded cached stage snapshot",
        error_enter_symbol: "Please enter a stock symbol.",
        error_wait_python: "Pyodide is still initializing. Please wait a moment and try again.",
    },
    zh: {
        hero_eyebrow: "浏览器优先的 Pyodide 量化实验室",
        hero_title: "量化研究工作站",
        hero_copy: "基于当前 Yahoo 行情抓取流程的单标的机构级研究界面，无需后端服务器。",
        language_label: "语言",
        hero_stat_pipeline: "流程",
        hero_stat_pipeline_value: "6 个阶段",
        hero_stat_scope: "数据范围",
        hero_stat_scope_value: "OHLCV + 事件",
        hero_stat_execution: "执行方式",
        hero_stat_execution_value: "浏览器内 Pyodide",
        research_run: "研究运行",
        research_run_copy: "一次抓取，完成特征构建、候选策略排名，并提升一个最佳当前策略。",
        field_symbol: "代码",
        field_symbol_placeholder: "例如 AAPL、MSFT、NVDA",
        field_period: "周期",
        field_execution: "执行层",
        field_execution_toggle: "启用元学习策略搜索",
        field_mode: "运行模式",
        mode_quick: "快速研究",
        mode_full: "完整排名",
        btn_initialize: "初始化研究流程",
        btn_run: "运行量化研究",
        btn_loading: "正在加载 Pyodide 运行时...",
        btn_failed: "运行时加载失败",
        loading_title: "正在运行分阶段研究流程",
        loading_prepare: "正在准备 Pyodide 环境...",
        debug_title: "启动调试",
        error_title: "研究流程错误",
        headline_recommendation: "最新建议",
        headline_composite: "综合信号强度",
        headline_walkforward: "滚动前瞻准确率",
        headline_alpha: "策略 Alpha",
        tab_overview: "总览",
        tab_regime: "市场状态实验室",
        tab_alpha: "Alpha 实验室",
        tab_models: "模型实验室",
        tab_strategy: "策略实验室",
        tab_risk: "风险实验室",
        tab_explain: "可解释性",
        status_done: "完成",
        status_running: "运行中",
        status_queued: "排队中",
        debug_waiting: "等待中",
        debug_booting: "启动中",
        debug_ready: "就绪",
        debug_failed: "失败",
        debug_running_research: "研究运行中",
        debug_research_ready: "研究已就绪",
        debug_research_failed: "研究失败",
        cache_none: "没有缓存结果",
        cache_ready: "已准备好首次运行",
        cache_cached: "已缓存到浏览器存储",
        cache_failed: "读取缓存失败",
        cache_loaded: "已加载缓存阶段结果",
        error_enter_symbol: "请输入股票代码。",
        error_wait_python: "Pyodide 仍在初始化，请稍后再试。",
    },
};

const STAGES = {
    en: [
        "Startup and package load",
        "Fetch and validate raw data",
        "Build feature store and model matrix",
        "Fit model stack and walk-forward validation",
        "Rank strategies and promote one policy",
        "Render charts and diagnostics",
    ],
    zh: [
        "启动与依赖加载",
        "抓取并校验原始数据",
        "构建特征库与模型矩阵",
        "拟合模型栈并执行滚动验证",
        "候选策略排名并提升一个策略",
        "渲染图表与诊断信息",
    ],
};

const INSTALL_STEPS = {
    en: [
        "Load Pyodide runtime",
        "Load numpy",
        "Load pandas",
        "Load scikit-learn",
        "Load data_fetcher.py",
        "Load feature_engineering.py",
        "Load strategy_optimizer.py",
        "Load research_workstation.py",
    ],
    zh: [
        "加载 Pyodide 运行时",
        "加载 numpy",
        "加载 pandas",
        "加载 scikit-learn",
        "加载 data_fetcher.py",
        "加载 feature_engineering.py",
        "加载 strategy_optimizer.py",
        "加载 research_workstation.py",
    ],
};

const predictBtn = document.getElementById("predictBtn");
const stockSymbolInput = document.getElementById("stockSymbol");
const timePeriodSelect = document.getElementById("timePeriod");
const executionModeSelect = document.getElementById("executionMode");
const optimizeStrategyCheckbox = document.getElementById("optimizeStrategy");
const loadingSection = document.getElementById("loadingSection");
const loadingMessage = document.getElementById("loadingMessage");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");
const cacheStatus = document.getElementById("cacheStatus");
const stageList = document.getElementById("stageList");
const installList = document.getElementById("installList");
const debugLog = document.getElementById("debugLog");
const debugStatus = document.getElementById("debugStatus");
const errorSection = document.getElementById("errorSection");
const errorMessage = document.getElementById("errorMessage");
const resultSection = document.getElementById("resultSection");
const languageSelect = document.getElementById("languageSelect");

function t(key) {
    return (LOCALES[currentLanguage] || LOCALES.en)[key] || key;
}

function stageLabels() {
    return STAGES[currentLanguage] || STAGES.en;
}

function installLabels() {
    return INSTALL_STEPS[currentLanguage] || INSTALL_STEPS.en;
}

function applyStaticTranslations() {
    document.documentElement.lang = currentLanguage === "zh" ? "zh-CN" : "en";
    document.querySelectorAll("[data-i18n]").forEach((node) => {
        node.textContent = t(node.dataset.i18n);
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
        node.placeholder = t(node.dataset.i18nPlaceholder);
    });
    document.title = t("hero_title");
    languageSelect.value = currentLanguage;
    if (!isPythonReady) {
        predictBtn.textContent = t("btn_loading");
        cacheStatus.textContent = t("cache_none");
    } else {
        predictBtn.textContent = t("btn_run");
    }
    renderStages(-1, false);
    renderInstallSteps(-1, -1);
}

function renderStages(activeIndex = -1, complete = false) {
    stageList.innerHTML = "";
    stageLabels().forEach((label, index) => {
        const status = complete || index < activeIndex ? t("status_done") : index === activeIndex ? t("status_running") : t("status_queued");
        const item = document.createElement("div");
        item.className = "stage-item";
        if (complete || index < activeIndex) item.classList.add("is-done");
        if (!complete && index === activeIndex) item.classList.add("is-active");
        item.innerHTML = `<span>${label}</span><strong>${status}</strong>`;
        stageList.appendChild(item);
    });
}

function renderInstallSteps(activeIndex = -1, doneIndex = -1) {
    installList.innerHTML = "";
    installLabels().forEach((label, index) => {
        const status = index <= doneIndex ? t("status_done") : index === activeIndex ? t("status_running") : t("status_queued");
        const item = document.createElement("div");
        item.className = "install-item";
        if (index <= doneIndex) item.classList.add("done");
        else if (index === activeIndex) item.classList.add("active");
        item.innerHTML = `<div class="install-item-row"><span>${label}</span><strong>${status}</strong></div><div class="install-item-bar"><span></span></div>`;
        installList.appendChild(item);
    });
}

function updateProgress(percent, message, activeStage = -1) {
    progressBar.style.width = `${percent}%`;
    progressText.textContent = `${percent}%`;
    loadingMessage.textContent = message;
    renderStages(activeStage, percent >= 100);
}

function appendDebug(message) {
    const stamp = new Date().toLocaleTimeString();
    debugLog.textContent += `[${stamp}] ${message}\n`;
    debugLog.scrollTop = debugLog.scrollHeight;
}

function markDebugStatus(key) {
    debugStatus.textContent = t(key);
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

function getCacheKey(symbol, period, mode) {
    return `quant-workstation:${symbol}:${period}:${mode}:${currentLanguage}`;
}

function saveRunToCache(symbol, period, mode, data) {
    try {
        localStorage.setItem(getCacheKey(symbol, period, mode), JSON.stringify({ savedAt: new Date().toISOString(), data }));
        cacheStatus.textContent = t("cache_cached");
    } catch (error) {
        cacheStatus.textContent = t("cache_failed");
    }
}

function readRunFromCache(symbol, period, mode) {
    try {
        const raw = localStorage.getItem(getCacheKey(symbol, period, mode));
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        cacheStatus.textContent = `${t("cache_loaded")} ${new Date(parsed.savedAt).toLocaleString()}`;
        return parsed.data;
    } catch (error) {
        cacheStatus.textContent = t("cache_failed");
        return null;
    }
}

function escapeForPython(value) {
    return String(value).replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

async function initPython() {
    try {
        predictBtn.disabled = true;
        predictBtn.textContent = t("btn_loading");
        loadingSection.classList.remove("hidden");
        debugLog.textContent = "";
        markDebugStatus("debug_booting");
        updateProgress(5, t("loading_prepare"), 0);
        renderInstallSteps(0, -1);

        pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/" });
        appendDebug("Pyodide runtime loaded");
        renderInstallSteps(1, 0);

        await pyodide.loadPackage("numpy");
        appendDebug("numpy ready");
        renderInstallSteps(2, 1);

        await pyodide.loadPackage("pandas");
        appendDebug("pandas ready");
        renderInstallSteps(3, 2);

        await pyodide.loadPackage("scikit-learn");
        appendDebug("scikit-learn ready");
        renderInstallSteps(4, 3);

        const moduleFiles = [
            "python/data_fetcher.py",
            "python/feature_engineering.py",
            "python/strategy_optimizer.py",
            "python/research_workstation.py",
        ];
        for (let index = 0; index < moduleFiles.length; index += 1) {
            const file = moduleFiles[index];
            renderInstallSteps(4 + index, 3 + index);
            const code = await fetch(file).then((response) => response.text());
            await pyodide.runPythonAsync(code);
            appendDebug(`${file} loaded`);
        }

        isPythonReady = true;
        predictBtn.disabled = false;
        predictBtn.textContent = t("btn_run");
        markDebugStatus("debug_ready");
        cacheStatus.textContent = t("cache_ready");
        updateProgress(100, t("btn_run"), 5);
    } catch (error) {
        predictBtn.textContent = t("btn_failed");
        markDebugStatus("debug_failed");
        showError(`Pyodide init failed: ${error.message}`);
    }
}

async function runResearch(symbol, period, optimizeStrategy, mode) {
    const pythonCode = `
result_json = await run_quant_research_async(
    symbol="${escapeForPython(symbol)}",
    period="${escapeForPython(period)}",
    optimize_strategy=${optimizeStrategy ? "True" : "False"},
    execution_mode="${escapeForPython(mode)}"
)
result_json
`;
    const output = await pyodide.runPythonAsync(pythonCode);
    return JSON.parse(output);
}

function formatPct(value, digits = 2) {
    return `${Number(value).toFixed(digits)}%`;
}

function renderKvGrid(targetId, rows) {
    const target = document.getElementById(targetId);
    target.innerHTML = rows.map((row) => `<div class="kv-item"><span>${row.label}</span><strong>${row.value}</strong></div>`).join("");
}

function renderTable(targetId, columns, rows) {
    const target = document.getElementById(targetId);
    if (!rows || !rows.length) {
        target.innerHTML = `<p class="muted">${currentLanguage === "zh" ? "暂无数据" : "No data available"}</p>`;
        return;
    }
    target.innerHTML = `<table><thead><tr>${columns.map((c) => `<th>${c.label}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${columns.map((c) => `<td>${row[c.key] ?? "-"}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

function renderNarrative(targetId, paragraphs) {
    document.getElementById(targetId).innerHTML = paragraphs.map((text) => `<p>${text}</p>`).join("");
}

function destroyCharts() {
    [priceChart, compositeChart, strategyChart].forEach((chart) => chart && chart.destroy());
}

function chartOptions(title) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#cfe0fb" } }, title: { display: true, text: title, color: "#e5eefc" } },
        scales: {
            x: { ticks: { color: "#92a8c7" }, grid: { color: "rgba(148,163,184,0.08)" } },
            y: { ticks: { color: "#92a8c7" }, grid: { color: "rgba(148,163,184,0.08)" } },
        },
    };
}

function renderPriceChart(charts) {
    priceChart = new Chart(document.getElementById("priceChart").getContext("2d"), {
        type: "line",
        data: {
            labels: charts.price.dates,
            datasets: [
                { label: "Close", data: charts.price.close, borderColor: "#39d0b6", borderWidth: 2.5, tension: 0.25 },
                { label: "MA20", data: charts.price.ma20, borderColor: "#f59e0b", borderDash: [6, 4], pointRadius: 0, borderWidth: 1.5 },
                { label: "MA50", data: charts.price.ma50, borderColor: "#8b5cf6", borderDash: [4, 4], pointRadius: 0, borderWidth: 1.5 },
            ],
        },
        options: chartOptions(currentLanguage === "zh" ? "价格结构" : "Price Structure"),
    });
}

function renderCompositeChart(charts) {
    compositeChart = new Chart(document.getElementById("compositeChart").getContext("2d"), {
        type: "line",
        data: {
            labels: charts.composite.dates,
            datasets: [
                { label: "Composite", data: charts.composite.values, borderColor: "#1d7cd6", borderWidth: 2.5, tension: 0.25 },
                { label: "Long Entry", data: charts.composite.long_entry, borderColor: "#22c55e", borderDash: [7, 4], pointRadius: 0 },
                { label: "Short Entry", data: charts.composite.short_entry, borderColor: "#f97316", borderDash: [7, 4], pointRadius: 0 },
            ],
        },
        options: chartOptions(currentLanguage === "zh" ? "综合信号" : "Composite Conviction"),
    });
}

function renderStrategyChart(charts) {
    strategyChart = new Chart(document.getElementById("strategyChart").getContext("2d"), {
        type: "line",
        data: {
            labels: charts.strategy.dates,
            datasets: [
                { label: "Promoted", data: charts.strategy.strategy, borderColor: "#22c55e", borderWidth: 2.5, tension: 0.25 },
                { label: charts.strategy.comparison_name, data: charts.strategy.comparison_strategy, borderColor: "#f59e0b", borderWidth: 1.8, tension: 0.25 },
                { label: "Buy & Hold", data: charts.strategy.buyhold, borderColor: "#94a3b8", borderDash: [6, 4], pointRadius: 0, borderWidth: 1.5 },
            ],
        },
        options: chartOptions(currentLanguage === "zh" ? "策略比较" : "Promoted vs Candidate Strategy"),
    });
}

function summarizeChange(newRun) {
    if (!lastRunSummary || lastRunSummary.symbol !== newRun.summary.symbol) {
        return currentLanguage === "zh" ? "这是该标的的首次运行。" : "This is the first run for this symbol.";
    }
    const prev = lastRunSummary;
    return currentLanguage === "zh"
        ? `与上次相比：推荐动作 ${prev.recommended_action} -> ${newRun.summary.recommended_action}，研究评分变化 ${(newRun.summary.research_score - prev.research_score).toFixed(2)}。`
        : `Since the last run: recommended action ${prev.recommended_action} -> ${newRun.summary.recommended_action}, research score delta ${(newRun.summary.research_score - prev.research_score).toFixed(2)}.`;
}

function renderHeadlines(data) {
    document.getElementById("headlineRecommendation").textContent = data.summary.recommendation;
    document.getElementById("headlineMeta").textContent = currentLanguage === "zh"
        ? `${data.summary.symbol} | 最新收盘 ${data.summary.last_close} | 单日变化 ${formatPct(data.summary.day_change_pct)}`
        : `${data.summary.symbol} | Last close ${data.summary.last_close} | Daily change ${formatPct(data.summary.day_change_pct)}`;
    document.getElementById("headlineConviction").textContent = `${data.summary.composite_index}/100`;
    document.getElementById("headlineState").textContent = currentLanguage === "zh"
        ? `${data.summary.promoted_strategy} | 当前状态 ${data.summary.target_state}`
        : `${data.summary.promoted_strategy} | state ${data.summary.target_state}`;
    document.getElementById("headlineValidation").textContent = formatPct(data.validation.walk_forward.rolling_accuracy * 100);
    document.getElementById("headlineQuality").textContent = `${currentLanguage === "zh" ? "数据质量" : "Data quality"} ${data.data_quality.score}/100`;
    document.getElementById("headlineAlpha").textContent = formatPct(data.strategy.performance.alpha_vs_buy_hold);
    document.getElementById("headlineRisk").textContent = data.strategy.latest_state.risk_flags.join(" | ");
}

function renderPanels(data) {
    renderKvGrid("overviewSnapshot", [
        { label: currentLanguage === "zh" ? "研究评分" : "Research Score", value: data.summary.research_score },
        { label: currentLanguage === "zh" ? "推荐策略" : "Promoted Policy", value: data.summary.promoted_strategy },
        { label: currentLanguage === "zh" ? "建议动作" : "Recommended Action", value: data.summary.recommended_action },
        { label: currentLanguage === "zh" ? "当前价格" : "Current Price", value: data.strategy.latest_state.current_price },
        { label: currentLanguage === "zh" ? "当前仓位" : "Current Position", value: data.strategy.latest_state.current_position },
        { label: currentLanguage === "zh" ? "信号置信度" : "Conviction", value: formatPct(data.strategy.latest_state.conviction * 100) },
    ]);

    renderNarrative("overviewNotes", [
        data.explanations.overview,
        data.explanations.why_now,
        summarizeChange(data),
    ]);
    renderNarrative("lastRunDelta", [summarizeChange(data)]);

    renderKvGrid("regimeProbabilities", Object.entries(data.models.regime_classifier.probabilities).map(([key, value]) => ({
        label: key.replace(/_/g, " "),
        value: formatPct(value * 100),
    })));

    renderTable("regimeSlices", [
        { key: "regime", label: currentLanguage === "zh" ? "市场状态" : "Regime" },
        { key: "accuracy", label: currentLanguage === "zh" ? "准确率" : "Accuracy" },
        { key: "sample_size", label: currentLanguage === "zh" ? "样本数" : "Samples" },
    ], data.validation.regime_slices.map((row) => ({ regime: row.regime, accuracy: formatPct(row.accuracy * 100), sample_size: row.sample_size })));

    renderTable("alphaSleeves", [
        { key: "name", label: currentLanguage === "zh" ? "因子组" : "Sleeve" },
        { key: "direction", label: currentLanguage === "zh" ? "方向" : "Direction" },
        { key: "score", label: currentLanguage === "zh" ? "得分" : "Score" },
        { key: "strength", label: currentLanguage === "zh" ? "强度" : "Strength" },
    ], data.signals.alpha_sleeves);

    renderNarrative("alphaConflict", [
        currentLanguage === "zh" ? `高优先级因子：${data.signals.alpha_sleeves.slice(0, 3).map((item) => item.name).join("、")}` : `Top sleeves: ${data.signals.alpha_sleeves.slice(0, 3).map((item) => item.name).join(", ")}`,
        currentLanguage === "zh" ? `模型状态：${data.models.confidence_governance.state}` : `Confidence state: ${data.models.confidence_governance.state}`,
    ]);

    renderTable("modelVotes", [
        { key: "name", label: currentLanguage === "zh" ? "模型" : "Model" },
        { key: "vote", label: currentLanguage === "zh" ? "投票" : "Vote" },
        { key: "prob", label: currentLanguage === "zh" ? "上涨概率" : "Prob. Up" },
        { key: "acc", label: currentLanguage === "zh" ? "测试准确率" : "Test Accuracy" },
    ], Object.entries(data.models.base_models).map(([name, model]) => ({
        name,
        vote: model.latest_vote,
        prob: formatPct(model.latest_probability_up * 100),
        acc: formatPct(model.test_accuracy * 100),
    })));

    renderTable("featureImportance", [
        { key: "feature", label: currentLanguage === "zh" ? "特征" : "Feature" },
        { key: "importance", label: currentLanguage === "zh" ? "重要性" : "Importance" },
    ], data.models.feature_importance);

    renderKvGrid("strategySummary", [
        { label: currentLanguage === "zh" ? "策略收益" : "Strategy Return", value: formatPct(data.strategy.performance.strategy_total_return) },
        { label: currentLanguage === "zh" ? "相对买入持有 Alpha" : "Alpha vs Buy & Hold", value: formatPct(data.strategy.performance.alpha_vs_buy_hold) },
        { label: currentLanguage === "zh" ? "最大回撤" : "Max Drawdown", value: formatPct(data.strategy.performance.max_drawdown) },
        { label: currentLanguage === "zh" ? "稳定性" : "Stability", value: data.strategy.performance.stability },
        { label: currentLanguage === "zh" ? "建议动作" : "Recommended Action", value: data.strategy.latest_state.recommended_action },
        { label: currentLanguage === "zh" ? "提升原因" : "Promotion Reason", value: data.strategy.promoted_policy.promotion_reason },
    ]);
    renderTable("strategyCandidates", [
        { key: "name", label: currentLanguage === "zh" ? "策略" : "Strategy" },
        { key: "score", label: currentLanguage === "zh" ? "评分" : "Score" },
        { key: "alpha", label: currentLanguage === "zh" ? "Alpha" : "Alpha" },
        { key: "max_drawdown", label: currentLanguage === "zh" ? "最大回撤" : "Max DD" },
        { key: "latest_action", label: currentLanguage === "zh" ? "最新动作" : "Latest Action" },
    ], data.strategy.candidates);

    renderTable("tradeLog", [
        { key: "entry_date", label: currentLanguage === "zh" ? "开仓" : "Entry" },
        { key: "exit_date", label: currentLanguage === "zh" ? "平仓" : "Exit" },
        { key: "side", label: currentLanguage === "zh" ? "方向" : "Side" },
        { key: "return_pct", label: currentLanguage === "zh" ? "收益 %" : "Return %" },
        { key: "holding_days", label: currentLanguage === "zh" ? "天数" : "Days" },
    ], data.strategy.trade_log);
    renderNarrative("metaLearnerSummary", [
        data.strategy.meta_learner.available
            ? `${currentLanguage === "zh" ? "元学习器已参与排名" : "Meta learner participated in ranking"}`
            : `${currentLanguage === "zh" ? "元学习器未被提升或不可用" : "Meta learner was unavailable or not promoted"}`,
        data.strategy.meta_learner.error ? data.strategy.meta_learner.error : "",
        data.strategy.promoted_policy.promotion_reason,
    ].filter(Boolean));

    renderKvGrid("riskSummary", [
        { label: currentLanguage === "zh" ? "尾部损失" : "Tail Loss", value: formatPct(data.strategy.performance.tail_loss_5pct) },
        { label: currentLanguage === "zh" ? "总暴露" : "Gross Exposure", value: data.strategy.performance.gross_exposure },
        { label: currentLanguage === "zh" ? "风险标记" : "Risk Flags", value: data.strategy.latest_state.risk_flags.join(" | ") },
        { label: currentLanguage === "zh" ? "验证警告" : "Validation Warnings", value: data.strategy.warnings.join(" | ") || "-" },
    ]);

    renderTable("calibrationTable", [
        { key: "bucket", label: currentLanguage === "zh" ? "分桶" : "Bucket" },
        { key: "avg_probability_up", label: currentLanguage === "zh" ? "平均上涨概率" : "Avg Prob Up" },
        { key: "actual_up_rate", label: currentLanguage === "zh" ? "实际上涨率" : "Actual Up Rate" },
        { key: "count", label: currentLanguage === "zh" ? "数量" : "Count" },
    ], data.validation.calibration);

    renderNarrative("explainWhy", [data.explanations.overview, data.explanations.why_now, data.explanations.risk]);
    renderNarrative("explainExecution", [data.explanations.execution]);
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
    lastRunSummary = { symbol: data.summary.symbol, recommended_action: data.summary.recommended_action, research_score: data.summary.research_score };
}

async function handlePredict() {
    const symbol = stockSymbolInput.value.trim().toUpperCase();
    const period = timePeriodSelect.value;
    const optimizeStrategy = optimizeStrategyCheckbox.checked;
    const mode = executionModeSelect.value;

    if (!symbol) return showError(t("error_enter_symbol"));
    if (!isPythonReady) return showError(t("error_wait_python"));

    hideError();
    hideResults();
    loadingSection.classList.remove("hidden");
    predictBtn.disabled = true;

    const cached = readRunFromCache(symbol, period, mode);
    if (cached) displayResearch(cached);

    try {
        markDebugStatus("debug_running_research");
        updateProgress(12, stageLabels()[1], 1);
        appendDebug(`Running ${symbol} / ${period} / ${mode}`);
        const data = await runResearch(symbol, period, optimizeStrategy, mode);
        updateProgress(100, stageLabels()[5], 5);
        saveRunToCache(symbol, period, mode, data);
        displayResearch(data);
        markDebugStatus("debug_research_ready");
    } catch (error) {
        markDebugStatus("debug_research_failed");
        showError(error.message || "Research failed");
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
            buttons.forEach((item) => item.classList.toggle("active", item === button));
            panels.forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === tab));
        });
    });
}

function setLanguage(language) {
    currentLanguage = language === "zh" ? "zh" : "en";
    localStorage.setItem("quant-workstation-language", currentLanguage);
    applyStaticTranslations();
    if (currentRun) displayResearch(currentRun);
}

predictBtn.addEventListener("click", handlePredict);
stockSymbolInput.addEventListener("keypress", (event) => event.key === "Enter" && handlePredict());
languageSelect.addEventListener("change", (event) => setLanguage(event.target.value));

setupTabs();
applyStaticTranslations();
initPython();
