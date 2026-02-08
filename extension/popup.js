document.addEventListener('DOMContentLoaded', function () {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const detailsBtn = document.getElementById('detailsBtn');
    const connStatus = document.getElementById('connStatus');

    // Check Backend Connection
    fetch('http://localhost:8000/docs')
        .then(r => {
            connStatus.textContent = 'Connected';
            connStatus.style.color = '#4ade80';
        })
        .catch(e => {
            connStatus.textContent = 'Offline';
            connStatus.style.color = '#f87171';
        });

    analyzeBtn.addEventListener('click', async () => {
        // Get current tab
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

        if (!tab) return;

        // UI Update
        document.getElementById('initialState').classList.add('hidden');
        document.getElementById('loadingState').classList.remove('hidden');

        try {
            // Analyze URL via Backend
            const response = await fetch('http://localhost:8000/api/analyze-news', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url: tab.url })
            });

            if (!response.ok) {
                if (response.status === 404) throw new Error("Feature not enabled in Backend (V1)");
                throw new Error("Analysis failed");
            }

            const data = await response.json();
            showResults(data);

        } catch (err) {
            showError(err.message);
        }
    });

    detailsBtn.addEventListener('click', () => {
        chrome.tabs.create({ url: 'http://localhost:5173' });
    });
});

function showResults(data) {
    document.getElementById('loadingState').classList.add('hidden');
    document.getElementById('resultState').classList.remove('hidden');
    document.getElementById('scoreCircle').classList.remove('hidden');

    document.getElementById('scoreValue').textContent = Math.round(data.credibility_score || 0) + '%';
    document.getElementById('verdict').textContent = data.verdict || 'Unknown';
    document.getElementById('details').textContent = data.recommendation || 'No details available.';

    const score = data.credibility_score || 0;
    const circle = document.getElementById('scoreCircle');
    if (score > 70) circle.style.borderColor = '#4ade80';
    else if (score > 40) circle.style.borderColor = '#facc15';
    else circle.style.borderColor = '#f87171';
}

function showError(msg) {
    document.getElementById('loadingState').classList.add('hidden');
    document.getElementById('resultState').classList.remove('hidden');
    document.getElementById('verdict').textContent = 'Error';
    document.getElementById('details').textContent = msg;
    document.getElementById('resultState').querySelector('h3').style.color = '#f87171';
}
