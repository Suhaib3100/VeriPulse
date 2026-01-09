// VeriPulse Chrome Extension - Popup Script

const API_BASE = "http://localhost:8000";

document.addEventListener('DOMContentLoaded', async () => {
  // Check backend status
  await checkBackendStatus();
  
  // Check current page
  await checkCurrentPage();
  
  // Setup scan button
  document.getElementById('scanBtn').addEventListener('click', startScan);
});

async function checkBackendStatus() {
  const statusEl = document.getElementById('backendStatus');
  
  try {
    const response = await fetch(`${API_BASE}/health`, { 
      method: 'GET',
      signal: AbortSignal.timeout(3000)
    });
    
    if (response.ok) {
      statusEl.textContent = 'Online';
      statusEl.className = 'status-value online';
    } else {
      throw new Error('Not OK');
    }
  } catch (e) {
    statusEl.textContent = 'Offline';
    statusEl.className = 'status-value offline';
  }
}

async function checkCurrentPage() {
  const statusEl = document.getElementById('pageStatus');
  
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tab?.url || '';
    
    const platforms = {
      'youtube.com': { name: 'YouTube', supported: true },
      'instagram.com': { name: 'Instagram', supported: true },
      'meet.google.com': { name: 'Google Meet', supported: true },
      'zoom.us': { name: 'Zoom', supported: true },
      'teams.microsoft.com': { name: 'MS Teams', supported: true }
    };
    
    let detected = null;
    for (const [domain, info] of Object.entries(platforms)) {
      if (url.includes(domain)) {
        detected = info;
        break;
      }
    }
    
    if (detected) {
      statusEl.textContent = detected.name;
      statusEl.className = 'status-value supported';
    } else if (url.includes('http')) {
      statusEl.textContent = 'Other Site';
      statusEl.className = 'status-value unsupported';
    } else {
      statusEl.textContent = 'Not a webpage';
      statusEl.className = 'status-value unsupported';
    }
  } catch (e) {
    statusEl.textContent = 'Unknown';
    statusEl.className = 'status-value unsupported';
  }
}

async function startScan() {
  const scanBtn = document.getElementById('scanBtn');
  const scanBtnText = document.getElementById('scanBtnText');
  const scanSection = document.getElementById('scanSection');
  const resultSection = document.getElementById('resultSection');
  
  // Disable button and show scanning state
  scanBtn.disabled = true;
  scanBtn.classList.add('scanning');
  scanBtnText.textContent = 'Scanning...';
  scanSection.classList.add('scanning');
  resultSection.classList.remove('visible');
  
  const startTime = Date.now();
  
  try {
    // Get current tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab?.id) {
      throw new Error('No active tab found');
    }
    
    // Try to trigger scan via content script
    let response;
    try {
      response = await chrome.tabs.sendMessage(tab.id, { action: 'triggerScan' });
    } catch (e) {
      // Content script might not be loaded, try direct capture
      response = await captureAndAnalyze(tab);
    }
    
    if (response?.success) {
      // Wait for result from background
      await waitForResult(startTime);
    } else {
      throw new Error(response?.error || 'Scan failed');
    }
    
  } catch (error) {
    console.error('Scan error:', error);
    showResult({
      status: 'error',
      verdict: 'ERROR',
      confidence: 0,
      message: error.message || 'Failed to scan video',
      scanTime: Date.now() - startTime
    });
  } finally {
    scanBtn.disabled = false;
    scanBtn.classList.remove('scanning');
    scanBtnText.textContent = 'Scan Again';
    scanSection.classList.remove('scanning');
  }
}

async function captureAndAnalyze(tab) {
  // Capture visible tab as image
  const screenshotUrl = await chrome.tabs.captureVisibleTab(null, { format: 'jpeg', quality: 80 });
  
  // Send to background for analysis
  const response = await chrome.runtime.sendMessage({
    action: 'quickScan',
    frames: [screenshotUrl]
  });
  
  if (response.success) {
    showResult({
      status: response.result.isReal ? 'real' : 'fake',
      verdict: response.result.verdict,
      confidence: response.result.confidence,
      message: response.result.message,
      scanTime: response.result.scanTime
    });
  }
  
  return response;
}

async function waitForResult(startTime) {
  // Poll for result or use callback
  return new Promise((resolve) => {
    const checkResult = () => {
      const elapsed = Date.now() - startTime;
      if (elapsed > 10000) {
        showResult({
          status: 'error',
          verdict: 'TIMEOUT',
          confidence: 0,
          message: 'Scan took too long. Try again.',
          scanTime: elapsed
        });
        resolve();
      } else {
        // Result should be shown by content script via message
        setTimeout(resolve, 2000);
      }
    };
    checkResult();
  });
}

function showResult(result) {
  const resultSection = document.getElementById('resultSection');
  const resultCard = document.getElementById('resultCard');
  const resultIcon = document.getElementById('resultIcon');
  const resultVerdict = document.getElementById('resultVerdict');
  const resultConfidence = document.getElementById('resultConfidence');
  const resultMessage = document.getElementById('resultMessage');
  const resultTime = document.getElementById('resultTime');
  
  // Set card class
  resultCard.className = `result-card ${result.status}`;
  
  // Set icon
  const icons = {
    real: '✅',
    fake: '⚠️',
    error: '❌'
  };
  resultIcon.textContent = icons[result.status] || '❓';
  
  // Set verdict
  resultVerdict.textContent = result.verdict;
  
  // Set confidence
  if (result.confidence > 0) {
    resultConfidence.textContent = `${(result.confidence * 100).toFixed(0)}% confidence`;
    resultConfidence.style.display = 'block';
  } else {
    resultConfidence.style.display = 'none';
  }
  
  // Set message
  resultMessage.textContent = result.message;
  
  // Set time
  resultTime.textContent = `Scanned in ${result.scanTime}ms`;
  
  // Show result section
  resultSection.classList.add('visible');
}

// Listen for results from background/content scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'scanResult') {
    showResult(request.result);
  }
});
