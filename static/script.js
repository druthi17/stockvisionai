/* ==========================================================================
   StockVision AI – Frontend JavaScript Application Logic
   ========================================================================== */

// Global State
let currentStockData = null;
let priceChartInstance = null;
let comparisonChartInstance = null;
let currentTimeframe = '1Y';

// DOM Elements
const searchForm = document.getElementById('search-form');
const tickerInput = document.getElementById('ticker-input');
const searchBtn = document.getElementById('search-btn');
const btnText = document.getElementById('btn-text');
const btnSpinner = document.getElementById('btn-spinner');

const alertBox = document.getElementById('alert-box');
const alertMessage = document.getElementById('alert-message');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingTitle = document.getElementById('loading-title');
const loadingSubtitle = document.getElementById('loading-subtitle');

const dashboardContent = document.getElementById('dashboard-content');
const conversionBanner = document.getElementById('conversion-banner');
const conversionBannerText = document.getElementById('conversion-banner-text');
const recentChipsContainer = document.getElementById('recent-chips');

const currNativeRadio = document.getElementById('curr-native');
const currInrRadio = document.getElementById('curr-inr');
const nativeCurrLabel = document.getElementById('native-curr-label');

const tfButtons = document.querySelectorAll('.tf-btn');
const toggleSma20 = document.getElementById('toggle-sma20');
const toggleSma50 = document.getElementById('toggle-sma50');

const compBtn = document.getElementById('comp-btn');

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    renderRecentSearches();
    setupEventListeners();
});

function setupEventListeners() {
    // Search form submit
    searchForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const ticker = tickerInput.value.trim().toUpperCase();
        if (ticker) {
            analyzeStock(ticker);
        }
    });

    // Display currency radio toggle
    currNativeRadio.addEventListener('change', () => updateDisplayValues());
    currInrRadio.addEventListener('change', () => updateDisplayValues());

    // Chart Timeframe Buttons
    tfButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            tfButtons.forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentTimeframe = e.target.getAttribute('data-tf');
            if (currentStockData) {
                renderHistoricalChart(currentStockData.chart_data);
            }
        });
    });

    // Technical indicator checkbox overlays
    toggleSma20.addEventListener('change', () => {
        if (currentStockData) renderHistoricalChart(currentStockData.chart_data);
    });
    toggleSma50.addEventListener('change', () => {
        if (currentStockData) renderHistoricalChart(currentStockData.chart_data);
    });

    // Stock Comparison Button
    compBtn.addEventListener('click', () => {
        const t1 = document.getElementById('comp-ticker-1').value.trim();
        const t2 = document.getElementById('comp-ticker-2').value.trim();
        const t3 = document.getElementById('comp-ticker-3').value.trim();
        const tickers = [t1, t2, t3].filter(t => t.length > 0);
        
        if (tickers.length === 0) {
            showAlert("Please enter at least one ticker symbol to compare.");
            return;
        }
        runStockComparison(tickers);
    });
}

/**
 * Main function to fetch stock data and prediction from Flask backend.
 */
async function analyzeStock(ticker) {
    hideAlert();
    showLoading(true, "Fetching Market Data...", `Downloading 5 years of daily market history for ${ticker}...`);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || `Server responded with status ${response.status}`);
        }

        // Save to global state
        currentStockData = data;

        // Save search to LocalStorage
        saveRecentSearch(data.stock_info.ticker);

        // Update Dashboard Interface
        dashboardContent.classList.remove('hidden');
        renderStockDashboard(data);

        // Scroll smoothly to dashboard
        dashboardContent.scrollIntoView({ behavior: 'smooth' });

    } catch (err) {
        console.error("Analysis error:", err);
        showAlert(err.message || "Failed to analyze stock. Please verify the ticker symbol and try again.");
    } finally {
        showLoading(false);
    }
}

/**
 * Render the entire dashboard with fetched data.
 */
function renderStockDashboard(data) {
    const info = data.stock_info;
    const pred = data.prediction;
    const model = data.model_info;
    const ind = data.technical_indicators;

    // Set Native Currency Toggle Label
    nativeCurrLabel.innerText = `Native (${info.native_currency})`;

    // Header & Company Info
    document.getElementById('kpi-company-name').innerText = info.company_name;
    document.getElementById('kpi-ticker-badge').innerText = info.ticker;
    document.getElementById('kpi-exchange-badge').innerText = info.exchange;

    // Forecast Date
    document.getElementById('kpi-forecast-date').innerText = pred.next_business_day;

    // Volume & Latest Date
    document.getElementById('kpi-volume').innerText = info.latest_volume.toLocaleString();
    
    // Model Status & Metrics
    const modelBadge = document.getElementById('kpi-model-badge');
    if (model.model_source === 'cached') {
        modelBadge.className = 'badge badge-cached';
        modelBadge.innerHTML = '<i class="fa-solid fa-database"></i> Cached Model';
    } else {
        modelBadge.className = 'badge badge-new';
        modelBadge.innerHTML = '<i class="fa-solid fa-fire"></i> Newly Trained';
    }

    document.getElementById('kpi-last-trained').innerText = model.last_trained_date;
    document.getElementById('kpi-epochs').innerText = `${model.epochs_completed} Epochs`;

    // Model Performance Metrics
    document.getElementById('metric-mae').innerText = model.metrics.mae;
    document.getElementById('metric-rmse').innerText = model.metrics.rmse;
    document.getElementById('metric-mape').innerText = `${model.metrics.mape}%`;
    document.getElementById('metric-mse').innerText = model.metrics.mse;

    document.getElementById('spec-train-samples').innerText = `${model.training_samples} days`;
    document.getElementById('spec-test-samples').innerText = `${model.testing_samples} days`;

    // Technical Indicators
    renderTechnicalIndicators(ind, info.native_currency);

    // Update Values based on selected Currency (Native vs INR)
    updateDisplayValues();

    // Render Historical Interactive Chart
    renderHistoricalChart(data.chart_data);
}

/**
 * Format currency amount cleanly with appropriate symbol.
 */
function formatCurrency(amount, currencyCode, isConverted = false) {
    if (amount === null || amount === undefined || isNaN(amount)) return '--';
    
    const symbols = {
        'USD': '$',
        'INR': '₹',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'CAD': 'CA$',
        'AUD': 'A$',
        'HKD': 'HK$',
        'SGD': 'S$'
    };

    const code = isConverted ? 'INR' : (currencyCode || 'USD');
    const symbol = symbols[code] || `${code} `;
    
    // Formatting with 2 decimals
    const formattedNum = Number(amount).toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });

    return `${symbol}${formattedNum}`;
}

/**
 * Update dynamic currency numbers on KPI cards depending on presentation mode.
 */
function updateDisplayValues() {
    if (!currentStockData) return;

    const info = currentStockData.stock_info;
    const pred = currentStockData.prediction;
    const isINRMode = currInrRadio.checked;
    const isNativeINR = (info.native_currency === 'INR');

    // Multiplier calculation
    let multiplier = 1.0;
    let targetCurr = info.native_currency;
    let isConverted = false;

    if (isINRMode && !isNativeINR) {
        multiplier = info.fx_to_inr;
        targetCurr = 'INR';
        isConverted = true;

        conversionBanner.classList.remove('hidden');
        conversionBannerText.innerText = `Prices are displayed in INR (₹) converted from ${info.native_currency} at 1 ${info.native_currency} = ₹${info.fx_to_inr.toFixed(2)}. LSTM model forecasts are computed in native stock currency.`;
    } else {
        conversionBanner.classList.add('hidden');
    }

    // Multiply numbers
    const currentPrice = info.latest_close * multiplier;
    const prevClose = info.prev_close * multiplier;
    const priceDiff = (info.latest_close - info.prev_close) * multiplier;
    const pricePctDiff = ((info.latest_close - info.prev_close) / info.prev_close) * 100;

    const predPrice = pred.predicted_price * multiplier;
    const predDiff = pred.predicted_change * multiplier;
    const predRangeLow = pred.range_low * multiplier;
    const predRangeHigh = pred.range_high * multiplier;
    const rmseVal = pred.residual_rmse * multiplier;

    const high52 = info.fifty_two_high * multiplier;
    const low52 = info.fifty_two_low * multiplier;

    // Populate Current Price Card
    document.getElementById('kpi-current-price').innerText = formatCurrency(currentPrice, targetCurr, isConverted);
    document.getElementById('kpi-prev-close').innerText = formatCurrency(prevClose, targetCurr, isConverted);
    
    const changeElem = document.getElementById('kpi-price-change');
    const sign = priceDiff >= 0 ? '+' : '';
    const colorClass = priceDiff >= 0 ? 'green-text' : 'red-text';
    const icon = priceDiff >= 0 ? '<i class="fa-solid fa-arrow-trend-up"></i>' : '<i class="fa-solid fa-arrow-trend-down"></i>';
    changeElem.className = colorClass;
    changeElem.innerHTML = `${icon} ${sign}${formatCurrency(priceDiff, targetCurr, isConverted)} (${sign}${pricePctDiff.toFixed(2)}%)`;

    // Populate Prediction Card
    document.getElementById('kpi-predicted-price').innerText = formatCurrency(predPrice, targetCurr, isConverted);
    
    const predChangeElem = document.getElementById('kpi-predicted-change');
    const predSign = predDiff >= 0 ? '+' : '';
    const predColorClass = predDiff >= 0 ? 'green-text' : 'red-text';
    const predIcon = predDiff >= 0 ? '<i class="fa-solid fa-circle-chevron-up"></i>' : '<i class="fa-solid fa-circle-chevron-down"></i>';
    predChangeElem.className = predColorClass;
    predChangeElem.innerHTML = `${predIcon} Expected ${predSign}${formatCurrency(predDiff, targetCurr, isConverted)} (${predSign}${pred.predicted_pct_change.toFixed(2)}%)`;

    // Populate Error Range Card
    document.getElementById('kpi-error-range').innerText = `${formatCurrency(predRangeLow, targetCurr, isConverted)} – ${formatCurrency(predRangeHigh, targetCurr, isConverted)}`;
    document.getElementById('kpi-rmse-val').innerText = formatCurrency(rmseVal, targetCurr, isConverted);

    // 52-Week High / Low
    document.getElementById('kpi-52-high').innerText = formatCurrency(high52, targetCurr, isConverted);
    document.getElementById('kpi-52-low').innerText = formatCurrency(low52, targetCurr, isConverted);
}

/**
 * Render Technical Indicator values & RSI signal badge.
 */
function renderTechnicalIndicators(ind, currencyCode) {
    const isINRMode = currInrRadio.checked;
    const isNativeINR = (currencyCode === 'INR');
    const multiplier = (isINRMode && !isNativeINR && currentStockData) ? currentStockData.stock_info.fx_to_inr : 1.0;
    const targetCurr = (isINRMode && !isNativeINR) ? 'INR' : currencyCode;
    const isConverted = (isINRMode && !isNativeINR);

    document.getElementById('ind-sma20').innerText = ind.sma_20 ? formatCurrency(ind.sma_20 * multiplier, targetCurr, isConverted) : '--';
    document.getElementById('ind-sma50').innerText = ind.sma_50 ? formatCurrency(ind.sma_50 * multiplier, targetCurr, isConverted) : '--';
    document.getElementById('ind-ema20').innerText = ind.ema_20 ? formatCurrency(ind.ema_20 * multiplier, targetCurr, isConverted) : '--';
    
    document.getElementById('ind-macd').innerText = ind.macd_line !== null ? ind.macd_line.toFixed(2) : '--';
    document.getElementById('ind-macd-signal').innerText = ind.macd_signal !== null ? ind.macd_signal.toFixed(2) : '--';

    // RSI 14 Logic & Badge
    const rsiVal = ind.rsi_14;
    const rsiElem = document.getElementById('ind-rsi14');
    const rsiBadge = document.getElementById('rsi-badge');

    if (rsiVal !== null && rsiVal !== undefined) {
        rsiElem.innerText = rsiVal.toFixed(2);
        if (rsiVal > 70) {
            rsiBadge.className = 'rsi-badge rsi-overbought';
            rsiBadge.innerText = 'Overbought';
        } else if (rsiVal < 30) {
            rsiBadge.className = 'rsi-badge rsi-oversold';
            rsiBadge.innerText = 'Oversold';
        } else {
            rsiBadge.className = 'rsi-badge rsi-neutral';
            rsiBadge.innerText = 'Neutral';
        }
    } else {
        rsiElem.innerText = '--';
        rsiBadge.className = 'rsi-badge hidden';
    }
}

/**
 * Render Chart.js Historical Interactive Chart.
 */
function renderHistoricalChart(chartData) {
    const ctx = document.getElementById('priceChart').getContext('2d');

    // Filter slices based on timeframe
    let dataPoints = 252; // default 1Y
    if (currentTimeframe === '1M') dataPoints = 21;
    else if (currentTimeframe === '3M') dataPoints = 63;
    else if (currentTimeframe === '6M') dataPoints = 126;
    else if (currentTimeframe === '1Y') dataPoints = 252;
    else if (currentTimeframe === '5Y') dataPoints = chartData.dates.length;

    const dates = chartData.dates.slice(-dataPoints);
    const closePrices = chartData.close.slice(-dataPoints);
    const sma20Prices = chartData.sma20.slice(-dataPoints);
    const sma50Prices = chartData.sma50.slice(-dataPoints);

    // Apply currency multiplier if in INR presentation mode
    const isINRMode = currInrRadio.checked;
    const info = currentStockData ? currentStockData.stock_info : null;
    const multiplier = (isINRMode && info && info.native_currency !== 'INR') ? info.fx_to_inr : 1.0;

    const finalClose = closePrices.map(p => p !== null ? Number((p * multiplier).toFixed(2)) : null);
    const finalSma20 = sma20Prices.map(p => p !== null ? Number((p * multiplier).toFixed(2)) : null);
    const finalSma50 = sma50Prices.map(p => p !== null ? Number((p * multiplier).toFixed(2)) : null);

    // Gradient background
    const gradient = ctx.createLinearGradient(0, 0, 0, 350);
    gradient.addColorStop(0, 'rgba(56, 189, 248, 0.35)');
    gradient.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

    const datasets = [
        {
            label: `Closing Price (${isINRMode && info.native_currency !== 'INR' ? '₹ INR' : info.currency_symbol})`,
            data: finalClose,
            borderColor: '#38bdf8',
            borderWidth: 2.5,
            backgroundColor: gradient,
            fill: true,
            tension: 0.15,
            pointRadius: 0,
            pointHoverRadius: 6,
            pointHoverBackgroundColor: '#38bdf8',
            pointHoverBorderColor: '#fff',
            pointHoverBorderWidth: 2
        }
    ];

    if (toggleSma20.checked) {
        datasets.push({
            label: 'SMA 20',
            data: finalSma20,
            borderColor: '#f59e0b',
            borderWidth: 1.8,
            borderDash: [4, 4],
            fill: false,
            pointRadius: 0
        });
    }

    if (toggleSma50.checked) {
        datasets.push({
            label: 'SMA 50',
            data: finalSma50,
            borderColor: '#a855f7',
            borderWidth: 1.8,
            fill: false,
            pointRadius: 0
        });
    }

    if (priceChartInstance) {
        priceChartInstance.destroy();
    }

    priceChartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: dates, datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 } }
                },
                tooltip: {
                    backgroundColor: '#131b2e',
                    titleColor: '#f8fafc',
                    bodyColor: '#94a3b8',
                    borderColor: '#23314d',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.raw !== null ? context.raw : 'N/A'}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(35, 49, 77, 0.5)' },
                    ticks: { color: '#64748b', maxTicksLimit: 10, font: { family: 'Inter', size: 11 } }
                },
                y: {
                    grid: { color: 'rgba(35, 49, 77, 0.5)' },
                    ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } }
                }
            }
        }
    });
}

/**
 * Execute Stock Comparison Request (Feature #22).
 */
async function runStockComparison(tickers) {
    showLoading(true, "Comparing Stocks...", `Fetching 1-year historical percentage returns for ${tickers.join(', ')}...`);

    try {
        const response = await fetch('/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ tickers: tickers })
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || "Stock comparison failed.");
        }

        renderComparisonChart(data);

    } catch (err) {
        console.error("Comparison error:", err);
        showAlert(err.message || "Failed to execute stock comparison.");
    } finally {
        showLoading(false);
    }
}

/**
 * Render multi-line Chart.js comparison plot.
 */
function renderComparisonChart(compData) {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    const colors = ['#38bdf8', '#10b981', '#a855f7'];

    const datasets = [];
    let idx = 0;

    for (const [ticker, item] of Object.entries(compData.tickers_data)) {
        datasets.push({
            label: `${ticker} (% Change)`,
            data: item.pct_change,
            borderColor: colors[idx % colors.length],
            borderWidth: 2,
            fill: false,
            tension: 0.1,
            pointRadius: 0
        });
        idx++;
    }

    if (comparisonChartInstance) {
        comparisonChartInstance.destroy();
    }

    comparisonChartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels: compData.dates, datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 } }
                },
                tooltip: {
                    backgroundColor: '#131b2e',
                    borderColor: '#23314d',
                    borderWidth: 1,
                    callbacks: {
                        label: (context) => `${context.dataset.label}: ${context.raw >= 0 ? '+' : ''}${context.raw}%`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(35, 49, 77, 0.5)' },
                    ticks: { color: '#64748b', maxTicksLimit: 10 }
                },
                y: {
                    grid: { color: 'rgba(35, 49, 77, 0.5)' },
                    ticks: {
                        color: '#64748b',
                        callback: (val) => `${val >= 0 ? '+' : ''}${val}%`
                    }
                }
            }
        }
    });
}

/**
 * LocalStorage Recent Searches Management (Feature #23).
 */
function saveRecentSearch(ticker) {
    if (!ticker) return;
    let searches = JSON.parse(localStorage.getItem('stockvision_recent') || '[]');
    // Remove duplicate if exists
    searches = searches.filter(t => t !== ticker);
    // Add to front
    searches.unshift(ticker);
    // Keep top 5
    searches = searches.slice(0, 5);
    localStorage.setItem('stockvision_recent', JSON.stringify(searches));
    renderRecentSearches();
}

function renderRecentSearches() {
    const searches = JSON.parse(localStorage.getItem('stockvision_recent') || '[]');
    recentChipsContainer.innerHTML = '';

    if (searches.length === 0) {
        recentChipsContainer.innerHTML = '<span class="chip-empty">No recent searches</span>';
        return;
    }

    searches.forEach(ticker => {
        const chip = document.createElement('span');
        chip.className = 'chip';
        chip.innerText = ticker;
        chip.addEventListener('click', () => {
            tickerInput.value = ticker;
            analyzeStock(ticker);
        });
        recentChipsContainer.appendChild(chip);
    });
}

/**
 * Loading Modal Overlay & UI Spinner controls.
 */
function showLoading(show, title = "Analyzing Market Data...", subtitle = "") {
    if (show) {
        loadingTitle.innerText = title;
        loadingSubtitle.innerText = subtitle;
        loadingOverlay.classList.remove('hidden');
        searchBtn.disabled = true;
        btnText.classList.add('hidden');
        btnSpinner.classList.remove('hidden');
    } else {
        loadingOverlay.classList.add('hidden');
        searchBtn.disabled = false;
        btnText.classList.remove('hidden');
        btnSpinner.classList.add('hidden');
    }
}

/**
 * Toast Alert Box controls.
 */
function showAlert(msg) {
    alertMessage.innerText = msg;
    alertBox.classList.remove('hidden');
}

function hideAlert() {
    alertBox.classList.add('hidden');
}
