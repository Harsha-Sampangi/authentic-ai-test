// Authentic.AI Background Service Worker
// Handles context menu creation and API communication

const API_URL = 'http://localhost:8000';

// Create context menu on extension install
chrome.runtime.onInstalled.addListener(() => {
    // Set open side panel on click
    chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });

    chrome.contextMenus.create({
        id: 'analyzeWithAuthentic.AI',
        title: '🛡️ Analyze with Authentic.AI',
        contexts: ['image']
    });

    // Initialize storage
    chrome.storage.local.set({
        analysesToday: 0,
        authentic: 0,
        suspicious: 0,
        uncertain: 0,
        lastReset: new Date().toDateString()
    });
});

// Handle context menu click
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    if (info.menuItemId === 'analyzeWithAuthentic.AI') {
        const imageUrl = info.srcUrl;

        // Send start message to content script
        await chrome.tabs.sendMessage(tab.id, {
            action: 'ANALYSIS_START',
            imageUrl: imageUrl
        });

        try {
            // Call the API
            const response = await fetch(`${API_URL}/api/analyze-url`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ url: imageUrl })
            });

            if (!response.ok) {
                throw new Error(`Analysis failed: ${response.status}`);
            }

            const result = await response.json();

            // Update stats
            await updateStats(result.authenticity_score);

            // Send result to content script
            await chrome.tabs.sendMessage(tab.id, {
                action: 'ANALYSIS_COMPLETE',
                result: {
                    score: result.authenticity_score,
                    isDeepfake: result.is_deepfake,
                    confidence: result.confidence,
                    alerts: result.alerts || [],
                    report: result.report,
                    forensicBreakdown: result.forensic_breakdown,
                    imageUrl: imageUrl,
                    sourceUrl: tab.url,
                    timestamp: new Date().toISOString()
                }
            });

        } catch (error) {
            console.error('Analysis error:', error);
            await chrome.tabs.sendMessage(tab.id, {
                action: 'ANALYSIS_ERROR',
                error: error.message
            });
        }
    }
});

// Update daily stats
async function updateStats(score) {
    const data = await chrome.storage.local.get(['analysesToday', 'authentic', 'suspicious', 'uncertain', 'lastReset']);

    // Reset if new day
    const today = new Date().toDateString();
    if (data.lastReset !== today) {
        data.analysesToday = 0;
        data.authentic = 0;
        data.suspicious = 0;
        data.uncertain = 0;
        data.lastReset = today;
    }

    data.analysesToday++;

    if (score >= 70) {
        data.authentic++;
    } else if (score >= 40) {
        data.uncertain++;
    } else {
        data.suspicious++;
    }

    await chrome.storage.local.set(data);
}

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'GET_STATS') {
        chrome.storage.local.get(['analysesToday', 'authentic', 'suspicious', 'uncertain']).then(sendResponse);
        return true; // Keep channel open for async response
    }
});
