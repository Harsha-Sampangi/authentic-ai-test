// Authentic.AI Popup Script
// Displays stats and handles popup interactions

document.addEventListener('DOMContentLoaded', async () => {
    // Get stats from background script
    const stats = await chrome.runtime.sendMessage({ action: 'GET_STATS' });

    if (stats) {
        document.getElementById('total-count').textContent = stats.analysesToday || 0;
        document.getElementById('authentic-count').textContent = stats.authentic || 0;
        document.getElementById('suspicious-count').textContent = stats.suspicious || 0;
        document.getElementById('uncertain-count').textContent = stats.uncertain || 0;
    }

    // Button handlers
    document.getElementById('open-dashboard').addEventListener('click', () => {
        chrome.tabs.create({ url: 'http://localhost:5173' });
    });

    document.getElementById('open-settings').addEventListener('click', () => {
        // For now, just show an alert. Could be extended to a settings page.
        alert('Settings coming soon!');
    });
});
