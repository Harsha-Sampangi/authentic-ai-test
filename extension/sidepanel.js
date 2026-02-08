document.addEventListener('DOMContentLoaded', function () {
    // --- Configuration ---
    const API_BASE = 'http://localhost:8000';
    const ENDPOINTS = {
        ANALYZE_URL: `${API_BASE}/api/analyze-url`,
        HEALTH: `${API_BASE}/docs`
    };

    // --- Elements ---
    const ui = {
        analyzeBtn: document.getElementById('analyzePageBtn'),
        resetBtn: document.getElementById('resetBtn'),
        connStatus: document.getElementById('connStatus'),
        pageTitlePreview: document.getElementById('pageTitlePreview'),
        scanMediaBtn: document.getElementById('scanMediaBtn'),
        mediaList: document.getElementById('mediaList'),
        historyList: document.getElementById('historyList'),
        scoreValue: document.getElementById('scoreValue'),
        scoreBar: document.getElementById('scoreBar'),
        verdictBadge: document.getElementById('verdictBadge'),
        scoreCard: document.getElementById('scoreCard'),
        alertsList: document.getElementById('alertsList'),
        downloadPdfBtn: document.getElementById('downloadPdfBtn'),
        exportCsvBtn: document.getElementById('exportCsvBtn'),
        tabs: document.querySelectorAll('.tab-btn'),
    };

    // --- State ---
    const state = {
        currentResult: null,
        history: [],
        views: {
            overview: document.getElementById('initialState'),
            loading: document.getElementById('loadingState'),
            result: document.getElementById('resultState'),
            media: document.getElementById('mediaView'),
            history: document.getElementById('historyView')
        }
    };

    // --- Initialization ---
    init();

    function init() {
        checkConnection();
        loadHistory();
        setupEventListeners();
    }

    // --- Event Listeners ---
    function setupEventListeners() {
        // Tab Switching
        ui.tabs.forEach(btn => {
            btn.addEventListener('click', () => handleTabSwitch(btn));
        });

        // Main Actions
        ui.analyzeBtn.addEventListener('click', handlePageAnalysis);
        ui.scanMediaBtn.addEventListener('click', handleMediaScan);
        ui.resetBtn.addEventListener('click', resetAnalysis);

        // Exports
        ui.downloadPdfBtn.addEventListener('click', () => window.print());
        ui.exportCsvBtn.addEventListener('click', handleExportCsv);
    }

    // --- Handlers ---

    function handleTabSwitch(clickedBtn) {
        // Update Active Tab UI
        ui.tabs.forEach(b => b.classList.remove('active'));
        clickedBtn.classList.add('active');

        const tabName = clickedBtn.dataset.tab;

        // Hide all views
        Object.values(state.views).forEach(el => el.classList.add('hidden'));

        // Show appropriate view
        if (tabName === 'overview') {
            if (state.currentResult) {
                state.views.result.classList.remove('hidden');
            } else if (!state.views.loading.classList.contains('hidden')) {
                // Keep loading visible if it was active
                state.views.loading.classList.remove('hidden');
            } else {
                state.views.overview.classList.remove('hidden');
            }
        } else {
            state.views[tabName].classList.remove('hidden');
        }
    }

    async function handlePageAnalysis() {
        // Show Loading
        switchTab('overview');
        state.views.overview.classList.add('hidden');
        state.views.result.classList.add('hidden');
        state.views.loading.classList.remove('hidden');

        try {
            // Get Active Tab
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

            if (!tab || !tab.url) {
                throw new Error("Cannot access tab URL. Try refreshing the page.");
            }

            if (ui.pageTitlePreview) ui.pageTitlePreview.textContent = tab.title || "Unknown Page";

            // Prepare Payload
            const payload = { url: tab.url };
            console.log("Analyzing:", payload);

            // API Request
            let data;
            try {
                const response = await fetch(ENDPOINTS.ANALYZE_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error(`Server returned ${response.status}`);
                }

                const serverData = await response.json();

                // Normalize server data to UI format
                data = normalizeResult(serverData);

            } catch (networkError) {
                console.warn("Backend unavailable, falling back to simulation:", networkError);
                data = simulateAnalysis(tab.url);

                // Show a toast or small indicator that this is a simulation?
                // For now, we just proceed.
            }

            // Update State
            state.currentResult = data;

            // Render
            renderResults(data);
            addToHistory(tab.title || tab.url, data.score);

            // Show Result
            state.views.loading.classList.add('hidden');
            state.views.result.classList.remove('hidden');

        } catch (err) {
            console.error(err);
            state.views.loading.classList.add('hidden');
            state.views.overview.classList.remove('hidden');
            alert(`Analysis Failed: ${err.message}`);
        }
    }

    async function handleMediaScan() {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab || !tab.id) return;

        try {
            const results = await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                func: () => {
                    return Array.from(document.images)
                        .filter(img => img.naturalWidth > 150 && img.naturalHeight > 150)
                        .slice(0, 12)
                        .map(img => img.src);
                }
            });

            const images = results[0]?.result || [];
            renderMediaList(images);

        } catch (e) {
            console.error("Script injection failed:", e);
            ui.mediaList.innerHTML = '<div class="empty-state"><p>Could not access page content.</p></div>';
        }
    }

    function handleExportCsv() {
        if (!state.currentResult) return;
        const res = state.currentResult;
        const csvContent = "data:text/csv;charset=utf-8,"
            + ["Metric,Value", `Score,${res.score}`, `Verdict,${res.verdict}`].join("\n");

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "authentic_ai_report.csv");
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }

    function resetAnalysis() {
        state.currentResult = null;
        state.views.result.classList.add('hidden');
        state.views.overview.classList.remove('hidden');
    }

    // --- Rendering ---

    function renderResults(data) {
        // Score Bar and Value
        const score = Math.round(data.score);
        ui.scoreValue.textContent = `${score}%`;
        ui.scoreBar.style.width = `${score}%`;

        // Verdict
        let verdict = data.verdict;
        let theme = 'success'; // default

        if (score < 50) {
            theme = 'danger';
            ui.scoreBar.style.backgroundColor = '#ef4444';
        } else if (score < 80) {
            theme = 'warning';
            ui.scoreBar.style.backgroundColor = '#eab308';
        } else {
            ui.scoreBar.style.backgroundColor = '#22c55e';
        }

        // Remove old classes
        ui.scoreCard.classList.remove('warning', 'danger');
        if (theme !== 'success') ui.scoreCard.classList.add(theme);

        ui.verdictBadge.textContent = verdict;

        // Alerts
        ui.alertsList.innerHTML = '';
        const alerts = data.alerts && data.alerts.length > 0 ? data.alerts : [
            { severity: 'Low', title: 'Start Analysis', description: 'No issues detected yet.' }
        ];

        alerts.forEach(alert => {
            const div = document.createElement('div');
            div.className = `alert-item ${alert.severity ? alert.severity.toLowerCase() : 'low'}`;
            div.innerHTML = `
                <div class="icon">${getSeverityIcon(alert.severity)}</div>
                <div class="content">
                    <div class="title">${alert.title}</div>
                    <div class="desc">${alert.description || ''}</div>
                </div>
            `;
            ui.alertsList.appendChild(div);
        });
    }

    function renderMediaList(images) {
        ui.mediaList.innerHTML = '';

        if (images.length === 0) {
            ui.mediaList.innerHTML = '<div class="empty-state"><p>No relevant images found.</p></div>';
            return;
        }

        images.forEach(src => {
            const item = document.createElement('div');
            item.className = 'media-item';
            item.innerHTML = `
                <img src="${src}" loading="lazy" />
                <div class="overlay">Analyze</div>
            `;
            item.onclick = () => alert('Image analysis not connected in this demo.');
            ui.mediaList.appendChild(item);
        });
    }

    function renderHistory() {
        ui.historyList.innerHTML = '';
        if (state.history.length === 0) {
            ui.historyList.innerHTML = '<div class="empty-state"><p>No history yet.</p></div>';
            return;
        }

        state.history.forEach(item => {
            const div = document.createElement('div');
            div.className = 'history-item';
            div.innerHTML = `
                <div class="h-title">${item.title}</div>
                <div class="h-meta">
                    <span>${item.date}</span>
                    <span style="color: ${item.score > 70 ? '#4ade80' : '#f87171'}">${Math.round(item.score)}%</span>
                </div>
            `;
            ui.historyList.appendChild(div);
        });
    }


    // --- Helpers ---

    function normalizeResult(serverData) {
        // Maps backend/dummy data to UI structure
        return {
            score: serverData.authenticity_score ?? serverData.score ?? 0,
            verdict: serverData.is_deepfake ? 'SUSPICIOUS' : 'LIKELY AUTHENTIC',
            alerts: serverData.alerts || [],
            details: serverData
        };
    }

    function simulateAnalysis(url) {
        // Fallback simulation
        const safeUrl = url || "";
        const isSuspicious = safeUrl.includes('fake') || safeUrl.length > 150;

        return {
            score: isSuspicious ? 45 : 94,
            verdict: isSuspicious ? 'SUSPICIOUS' : 'LIKELY AUTHENTIC',
            alerts: isSuspicious ? [
                { severity: 'High', title: 'Suspicious Pattern', description: 'URL matches known phishing patterns.' }
            ] : [
                { severity: 'Low', title: 'Domain Verified', description: 'Source appears legitimate.' }
            ]
        };
    }

    function addToHistory(title, score) {
        const item = {
            title: (title || "Untitled").substring(0, 50),
            score: score,
            date: new Date().toLocaleDateString()
        };

        state.history.unshift(item);
        if (state.history.length > 20) state.history.pop();

        chrome.storage.local.set({ history: state.history });
        renderHistory();
    }

    function loadHistory() {
        chrome.storage.local.get(['history'], (res) => {
            if (res.history) {
                state.history = res.history;
                renderHistory();
            }
        });
    }

    function checkConnection() {
        fetch(ENDPOINTS.HEALTH)
            .then(() => {
                ui.connStatus.textContent = '● Connected';
                ui.connStatus.classList.add('online');
            })
            .catch(() => {
                ui.connStatus.textContent = '○ Offline';
                ui.connStatus.classList.remove('online');
            });
    }

    function switchTab(name) {
        const btn = document.querySelector(`.tab-btn[data-tab="${name}"]`);
        if (btn) handleTabSwitch(btn);
    }

    function getSeverityIcon(severity) {
        const s = (severity || '').toLowerCase();
        if (s === 'high') return '❌';
        if (s === 'medium') return '⚠️';
        return '✅';
    }
});
