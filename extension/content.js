// Authentic.AI Content Script
// Renders the analysis overlay on web pages

let overlay = null;

// Listen for messages from background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  switch (message.action) {
    case 'ANALYSIS_START':
      showLoadingOverlay(message.imageUrl);
      break;
    case 'ANALYSIS_COMPLETE':
      showResultOverlay(message.result);
      break;
    case 'ANALYSIS_ERROR':
      showErrorOverlay(message.error);
      break;
  }
});

function createOverlay() {
  // Remove existing overlay if any
  if (overlay) {
    overlay.remove();
  }

  overlay = document.createElement('div');
  overlay.id = 'authentiai-overlay';
  overlay.innerHTML = `
    <div class="authentiai-modal">
      <div class="authentiai-header">
        <div class="authentiai-logo">
          <span class="authentiai-icon">🛡️</span>
          <span>Authentic.AI</span>
        </div>
        <button class="authentiai-close" id="authentiai-close">×</button>
      </div>
      <div class="authentiai-content" id="authentiai-content">
        <!-- Content will be injected here -->
      </div>
    </div>
  `;

  document.body.appendChild(overlay);

  // Close button handler
  document.getElementById('authentiai-close').addEventListener('click', () => {
    overlay.remove();
    overlay = null;
  });

  // Click outside to close
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) {
      overlay.remove();
      overlay = null;
    }
  });

  return overlay;
}

function showLoadingOverlay(imageUrl) {
  createOverlay();
  const content = document.getElementById('authentiai-content');

  content.innerHTML = `
    <div class="authentiai-loading">
      <div class="authentiai-preview">
        <img src="${imageUrl}" alt="Analyzing..." />
      </div>
      <div class="authentiai-status">
        <div class="authentiai-spinner"></div>
        <p>🔍 Analyzing image...</p>
      </div>
      <div class="authentiai-progress-container">
        <div class="authentiai-progress-bar">
          <div class="authentiai-progress-fill" id="authentiai-progress"></div>
        </div>
        <span id="authentiai-progress-text">0%</span>
      </div>
    </div>
  `;

  // Animate progress bar
  let progress = 0;
  const progressFill = document.getElementById('authentiai-progress');
  const progressText = document.getElementById('authentiai-progress-text');

  const interval = setInterval(() => {
    progress = Math.min(progress + Math.random() * 15, 90);
    if (progressFill) {
      progressFill.style.width = `${progress}%`;
      progressText.textContent = `${Math.round(progress)}%`;
    } else {
      clearInterval(interval);
    }
  }, 200);
}

function showResultOverlay(result) {
  if (!overlay) {
    createOverlay();
  }

  const content = document.getElementById('authentiai-content');
  const scoreColor = result.score >= 70 ? '#22c55e' : result.score >= 40 ? '#eab308' : '#ef4444';
  const statusText = result.isDeepfake ? '🚨 HIGHLY SUSPICIOUS' : '✅ LIKELY AUTHENTIC';
  const statusClass = result.isDeepfake ? 'suspicious' : 'authentic';

  // Get quick findings from forensic breakdown
  let findings = [];
  if (result.forensicBreakdown && result.forensicBreakdown.sections) {
    result.forensicBreakdown.sections.forEach(section => {
      if (section.findings) {
        section.findings.slice(0, 3).forEach(f => {
          findings.push({ status: f.status, text: f.text });
        });
      }
    });
  }
  findings = findings.slice(0, 3); // Show max 3 findings

  content.innerHTML = `
    <div class="authentiai-result">
      <div class="authentiai-preview">
        <img src="${result.imageUrl}" alt="Analyzed" />
      </div>

      <div class="authentiai-score-section">
        <p class="authentiai-label">Authenticity Score</p>
        <div class="authentiai-score-bar">
          <div class="authentiai-score-fill" style="width: ${result.score}%; background: ${scoreColor}"></div>
        </div>
        <div class="authentiai-score-value" style="color: ${scoreColor}">${Math.round(result.score)}%</div>
      </div>

      <div class="authentiai-status-badge ${statusClass}">
        ${statusText}
      </div>

      <div class="authentiai-meta">
        <p><strong>Confidence:</strong> ${result.confidence}</p>
      </div>

      ${findings.length > 0 ? `
        <div class="authentiai-findings">
          <p class="authentiai-label">Quick Findings:</p>
          <ul>
            ${findings.map(f => `
              <li class="${f.status}">
                ${f.status === 'pass' ? '✅' : f.status === 'fail' ? '❌' : '⚠️'} ${f.text}
              </li>
            `).join('')}
          </ul>
        </div>
      ` : ''}

      <div class="authentiai-actions">
        <button class="authentiai-btn primary" id="authentiai-full-report">Full Report</button>
        <button class="authentiai-btn secondary" id="authentiai-save">Save Analysis</button>
      </div>

      <div class="authentiai-footer">
        <p>Analyzed from: ${new URL(result.sourceUrl).hostname}</p>
        <p>Timestamp: ${new Date(result.timestamp).toLocaleString()}</p>
      </div>
    </div>
  `;

  // Button handlers
  document.getElementById('authentiai-full-report').addEventListener('click', () => {
    window.open('http://localhost:5173', '_blank');
  });

  document.getElementById('authentiai-save').addEventListener('click', () => {
    // Save to storage
    chrome.storage.local.get(['savedAnalyses']).then(data => {
      const analyses = data.savedAnalyses || [];
      analyses.push(result);
      chrome.storage.local.set({ savedAnalyses: analyses });
      alert('Analysis saved!');
    });
  });
}

function showErrorOverlay(error) {
  if (!overlay) {
    createOverlay();
  }

  const content = document.getElementById('authentiai-content');

  content.innerHTML = `
    <div class="authentiai-error">
      <div class="authentiai-error-icon">❌</div>
      <h3>Analysis Failed</h3>
      <p>${error}</p>
      <button class="authentiai-btn primary" id="authentiai-retry">Try Again</button>
    </div>
  `;

  document.getElementById('authentiai-retry').addEventListener('click', () => {
    overlay.remove();
    overlay = null;
  });
}
