// VeriPulse Chrome Extension - Content Script
// Injects into YouTube, Instagram, Google Meet, Zoom, Teams

(function() {
  'use strict';
  
  const SCAN_FRAME_COUNT = 5; // Fast scan with 5 frames
  const SCAN_INTERVAL = 200; // Capture frame every 200ms
  
  let isScanning = false;
  let scanButton = null;
  let resultOverlay = null;
  
  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  
  function init() {
    console.log("VeriPulse content script loaded");
    
    // Detect platform and inject UI
    const platform = detectPlatform();
    if (platform) {
      injectScanButton(platform);
      setupVideoObserver();
    }
  }
  
  function detectPlatform() {
    const url = window.location.href;
    if (url.includes('youtube.com')) return 'youtube';
    if (url.includes('instagram.com')) return 'instagram';
    if (url.includes('meet.google.com')) return 'meet';
    if (url.includes('zoom.us')) return 'zoom';
    if (url.includes('teams.microsoft.com')) return 'teams';
    return null;
  }
  
  function injectScanButton(platform) {
    // Remove existing button if any
    const existing = document.getElementById('veripulse-scan-btn');
    if (existing) existing.remove();
    
    // Create scan button
    scanButton = document.createElement('div');
    scanButton.id = 'veripulse-scan-btn';
    scanButton.innerHTML = `
      <div class="vp-btn-content">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          <path d="M9 12l2 2 4-4"/>
        </svg>
        <span>Scan</span>
      </div>
    `;
    
    scanButton.addEventListener('click', () => startScan(platform));
    document.body.appendChild(scanButton);
    
    // Create result overlay
    resultOverlay = document.createElement('div');
    resultOverlay.id = 'veripulse-result';
    resultOverlay.style.display = 'none';
    document.body.appendChild(resultOverlay);
  }
  
  function setupVideoObserver() {
    // Watch for video elements being added
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        for (const node of mutation.addedNodes) {
          if (node.tagName === 'VIDEO' || (node.querySelector && node.querySelector('video'))) {
            console.log("VeriPulse: Video detected");
          }
        }
      }
    });
    
    observer.observe(document.body, { childList: true, subtree: true });
  }
  
  async function startScan(platform) {
    if (isScanning) return;
    isScanning = true;
    
    updateButton('scanning');
    showResult({ status: 'scanning', message: 'Analyzing video...' });
    
    try {
      const video = findVideo(platform);
      if (!video) {
        throw new Error('No video found on this page');
      }
      
      // Capture frames quickly
      const frames = await captureFrames(video, SCAN_FRAME_COUNT, SCAN_INTERVAL);
      
      if (frames.length === 0) {
        throw new Error('Could not capture video frames');
      }
      
      // Send to background for analysis
      const response = await chrome.runtime.sendMessage({
        action: 'quickScan',
        frames: frames
      });
      
      if (response.success) {
        showResult({
          status: response.result.isReal ? 'real' : 'fake',
          verdict: response.result.verdict,
          confidence: response.result.confidence,
          message: response.result.message,
          scanTime: response.result.scanTime
        });
        updateButton(response.result.isReal ? 'real' : 'fake');
      } else {
        throw new Error(response.error || 'Analysis failed');
      }
      
    } catch (error) {
      console.error('VeriPulse scan error:', error);
      showResult({ status: 'error', message: error.message });
      updateButton('error');
    } finally {
      isScanning = false;
    }
  }
  
  function findVideo(platform) {
    let video = null;
    
    switch (platform) {
      case 'youtube':
        video = document.querySelector('video.html5-main-video') || 
                document.querySelector('video');
        break;
      case 'instagram':
        video = document.querySelector('video');
        break;
      case 'meet':
      case 'zoom':
      case 'teams':
        // Find the largest video (usually the main speaker)
        const videos = Array.from(document.querySelectorAll('video'));
        if (videos.length > 0) {
          video = videos.reduce((a, b) => {
            const aArea = (a.videoWidth || a.clientWidth) * (a.videoHeight || a.clientHeight);
            const bArea = (b.videoWidth || b.clientWidth) * (b.videoHeight || b.clientHeight);
            return aArea > bArea ? a : b;
          });
        }
        break;
      default:
        video = document.querySelector('video');
    }
    
    return video;
  }
  
  async function captureFrames(video, count, interval) {
    const frames = [];
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Use video dimensions, max 640px for speed
    const scale = Math.min(1, 640 / Math.max(video.videoWidth || 640, video.videoHeight || 480));
    canvas.width = (video.videoWidth || 640) * scale;
    canvas.height = (video.videoHeight || 480) * scale;
    
    for (let i = 0; i < count; i++) {
      try {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frameData = canvas.toDataURL('image/jpeg', 0.8);
        frames.push(frameData);
      } catch (e) {
        console.error('Frame capture error:', e);
      }
      
      if (i < count - 1) {
        await new Promise(resolve => setTimeout(resolve, interval));
      }
    }
    
    return frames;
  }
  
  function updateButton(status) {
    if (!scanButton) return;
    
    scanButton.className = `vp-status-${status}`;
    
    const content = scanButton.querySelector('.vp-btn-content span');
    if (content) {
      const labels = {
        scanning: 'Scanning...',
        real: '✓ Real',
        fake: '⚠ Fake',
        error: 'Retry'
      };
      content.textContent = labels[status] || 'Scan';
    }
  }
  
  function showResult(result) {
    if (!resultOverlay) return;
    
    const statusColors = {
      scanning: '#F59E0B',
      real: '#22C55E',
      fake: '#EF4444',
      error: '#6B7280'
    };
    
    const statusIcons = {
      scanning: '⏳',
      real: '✅',
      fake: '⚠️',
      error: '❌'
    };
    
    resultOverlay.innerHTML = `
      <div class="vp-result-card vp-${result.status}">
        <div class="vp-result-header">
          <span class="vp-icon">${statusIcons[result.status]}</span>
          <span class="vp-verdict">${result.verdict || result.status.toUpperCase()}</span>
          ${result.confidence ? `<span class="vp-confidence">${(result.confidence * 100).toFixed(0)}%</span>` : ''}
        </div>
        <div class="vp-result-message">${result.message}</div>
        ${result.scanTime ? `<div class="vp-scan-time">Scanned in ${result.scanTime}ms</div>` : ''}
        <button class="vp-close-btn" onclick="this.parentElement.parentElement.style.display='none'">×</button>
      </div>
    `;
    
    resultOverlay.style.display = 'block';
    
    // Auto-hide after 10 seconds for real/fake results
    if (result.status !== 'scanning') {
      setTimeout(() => {
        if (resultOverlay) resultOverlay.style.display = 'none';
      }, 10000);
    }
  }
  
  // Listen for messages from popup
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'triggerScan') {
      const platform = detectPlatform();
      if (platform) {
        startScan(platform);
        sendResponse({ success: true });
      } else {
        sendResponse({ success: false, error: 'Unsupported platform' });
      }
    }
    return true;
  });
  
})();
