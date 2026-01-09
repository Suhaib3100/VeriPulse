// VeriPulse Chrome Extension - Content Script
// Injects into YouTube, Instagram, Google Meet, Zoom, Teams

(function() {
  'use strict';
  
  // Full video scan settings - more frames for better accuracy
  const SCAN_FRAME_COUNT = 15; // Analyze 15 frames across video
  const SCAN_INTERVAL = 300; // Capture frame every 300ms = 4.5 seconds of video
  
  let isScanning = false;
  let scanButton = null;
  let resultOverlay = null;
  let lastScanResult = null; // Track last result for re-scan
  
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
    
    // Remove existing result overlay
    const existingResult = document.getElementById('veripulse-result');
    if (existingResult) existingResult.remove();
    
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
    
    scanButton.addEventListener('click', () => {
      // Always reset button for new scan
      scanButton.className = '';
      const content = scanButton.querySelector('.vp-btn-content span');
      if (content) content.textContent = 'Scan';
      startScan(platform);
    });
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
    
    // Clear previous results for fresh scan
    lastScanResult = null;
    if (resultOverlay) {
      resultOverlay.style.display = 'none';
    }
    
    updateButton('scanning');
    showResult({ status: 'scanning', message: 'Scanning video... 0%', progress: 0 });
    
    try {
      const video = findVideo(platform);
      if (!video) {
        throw new Error('No video found on this page');
      }
      
      // Capture frames across the video for thorough analysis
      const frames = await captureFramesWithProgress(video, SCAN_FRAME_COUNT, SCAN_INTERVAL, (progress) => {
        showResult({ status: 'scanning', message: `Scanning video... ${Math.round(progress * 100)}%`, progress });
      });
      
      if (frames.length === 0) {
        throw new Error('Could not capture video frames');
      }
      
      // Send to background for analysis
      const response = await chrome.runtime.sendMessage({
        action: 'quickScan',
        frames: frames
      });
      
      if (response.success) {
        const result = response.result;
        
        // Map verdict to status for UI
        let status = 'uncertain';
        const v = (result.verdict || '').toLowerCase();
        if (v === 'real') status = 'real';
        else if (v === 'likely_real') status = 'likely_real';
        else if (v === 'uncertain') status = 'uncertain';
        else if (v === 'likely_fake') status = 'likely_fake';
        else if (v === 'fake') status = 'fake';
        
        showResult({
          status: status,
          verdict: result.verdict,
          confidence: result.confidence,
          trustScore: result.trustScore,
          framesAnalyzed: result.framesAnalyzed,
          facesDetected: result.facesDetected,
          verdictBreakdown: result.verdictBreakdown,
          reasons: result.reasons,
          message: result.message,
          scanTime: result.scanTime
        });
        updateButton(status);
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
  
  async function captureFramesWithProgress(video, count, interval, onProgress) {
    const frames = [];
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Use higher resolution for better analysis (max 800px)
    const scale = Math.min(1, 800 / Math.max(video.videoWidth || 640, video.videoHeight || 480));
    canvas.width = (video.videoWidth || 640) * scale;
    canvas.height = (video.videoHeight || 480) * scale;
    
    // For YouTube/Instagram, try to sample different parts of the video
    const videoDuration = video.duration || 0;
    const isLiveVideo = !videoDuration || videoDuration === Infinity;
    
    for (let i = 0; i < count; i++) {
      try {
        // Report progress
        if (onProgress) onProgress(i / count);
        
        // For recorded videos, seek to different timestamps
        if (!isLiveVideo && videoDuration > 5) {
          const seekTime = (videoDuration * i) / count;
          // Don't actually seek as it might disrupt playback
          // Just capture current frame
        }
        
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frameData = canvas.toDataURL('image/jpeg', 0.85); // Higher quality
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
        likely_real: '✓ Likely Real',
        uncertain: '? Uncertain',
        likely_fake: '⚠ Suspicious',
        fake: '⚠ Fake',
        error: 'Retry'
      };
      content.textContent = labels[status] || 'Scan';
    }
  }
  
  function showResult(result) {
    if (!resultOverlay) return;
    
    // Map status to display properties
    const statusConfig = {
      scanning: { color: '#F59E0B', icon: '⏳', label: 'Scanning...' },
      real: { color: '#22C55E', icon: '✅', label: 'REAL' },
      likely_real: { color: '#10B981', icon: '✓', label: 'LIKELY REAL' },
      uncertain: { color: '#F59E0B', icon: '❓', label: 'UNCERTAIN' },
      likely_fake: { color: '#F97316', icon: '⚠️', label: 'SUSPICIOUS' },
      fake: { color: '#EF4444', icon: '🚫', label: 'POTENTIAL DEEPFAKE' },
      error: { color: '#6B7280', icon: '❌', label: 'ERROR' }
    };
    
    // Determine display status from result
    let displayStatus = result.status;
    if (result.verdict) {
      const v = result.verdict.toLowerCase().replace('_', '_');
      if (v.includes('real') && !v.includes('likely')) displayStatus = 'real';
      else if (v.includes('likely_real') || v.includes('likely real')) displayStatus = 'likely_real';
      else if (v.includes('uncertain')) displayStatus = 'uncertain';
      else if (v.includes('likely_fake') || v.includes('likely fake')) displayStatus = 'likely_fake';
      else if (v.includes('fake')) displayStatus = 'fake';
    }
    
    const config = statusConfig[displayStatus] || statusConfig.error;
    const trustScore = result.trustScore || result.confidence || 0;
    
    // Scanning progress view
    if (displayStatus === 'scanning') {
      const progress = result.progress || 0;
      resultOverlay.innerHTML = `
        <div class="vp-result-card vp-scanning">
          <div class="vp-result-header">
            <span class="vp-icon">⏳</span>
            <span class="vp-verdict" style="color: ${config.color}">Scanning Video...</span>
          </div>
          <div class="vp-scan-progress">
            <div class="vp-progress-track">
              <div class="vp-progress-fill" style="width: ${progress * 100}%"></div>
            </div>
            <div class="vp-progress-text">${Math.round(progress * 100)}% - Capturing frames</div>
          </div>
        </div>
      `;
      resultOverlay.style.display = 'block';
      return;
    }
    
    // Build reasons HTML
    let reasonsHtml = '';
    if (result.reasons && result.reasons.length > 0) {
      reasonsHtml = `
        <div class="vp-reasons">
          ${result.reasons.slice(0, 5).map(r => `<div class="vp-reason">${r}</div>`).join('')}
        </div>
      `;
    }
    
    // Build verdict breakdown if available
    let breakdownHtml = '';
    if (result.verdictBreakdown) {
      const vb = result.verdictBreakdown;
      const faceCount = result.facesDetected || 0;
      breakdownHtml = `
        <div class="vp-breakdown">
          <span class="vp-breakdown-item vp-real">✅ ${(vb.REAL || 0) + (vb.LIKELY_REAL || 0)}</span>
          <span class="vp-breakdown-item vp-uncertain">❓ ${vb.UNCERTAIN || 0}</span>
          <span class="vp-breakdown-item vp-fake">⚠️ ${(vb.FAKE || 0) + (vb.LIKELY_FAKE || 0)}</span>
          <span class="vp-breakdown-item vp-faces">👤 ${faceCount}</span>
        </div>
      `;
    }
    
    resultOverlay.innerHTML = `
      <div class="vp-result-card vp-${displayStatus}">
        <button class="vp-close-btn" onclick="this.parentElement.parentElement.style.display='none'">×</button>
        <div class="vp-result-header">
          <span class="vp-icon">${config.icon}</span>
          <span class="vp-verdict" style="color: ${config.color}">${config.label}</span>
        </div>
        <div class="vp-trust-bar">
          <div class="vp-trust-label">Trust Score</div>
          <div class="vp-trust-track">
            <div class="vp-trust-fill" style="width: ${trustScore * 100}%; background: ${config.color}"></div>
          </div>
          <div class="vp-trust-value">${(trustScore * 100).toFixed(0)}%</div>
        </div>
        ${breakdownHtml}
        ${reasonsHtml}
        <div class="vp-result-footer">
          <span class="vp-frames">${result.framesAnalyzed || 0} frames analyzed</span>
          ${result.scanTime ? `<span class="vp-time">${(result.scanTime / 1000).toFixed(1)}s</span>` : ''}
        </div>
      </div>
    `;
    
    resultOverlay.style.display = 'block';
    
    // Auto-hide after 15 seconds for non-scanning results
    if (result.status !== 'scanning') {
      setTimeout(() => {
        if (resultOverlay) resultOverlay.style.display = 'none';
      }, 15000);
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
