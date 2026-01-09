// VeriPulse Chrome Extension - Content Script
// Auto-scans videos on YouTube, Instagram, Google Meet, Zoom, Teams

(function() {
  'use strict';
  
  // Auto-scan settings
  const SCAN_FRAME_COUNT = 15;
  const SCAN_INTERVAL = 300;
  const AUTO_SCAN_DELAY = 1500; // Wait 1.5s after video starts
  
  let isScanning = false;
  let resultOverlay = null;
  let currentVideoSrc = null;
  let currentVideoElement = null;
  let scanTimeout = null;
  let lastScanTime = 0;
  
  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  
  function init() {
    console.log("VeriPulse: Auto-scan initialized");
    
    const platform = detectPlatform();
    if (platform) {
      createResultOverlay();
      setupAutoScan(platform);
      
      // Also watch for URL changes (SPA navigation)
      setupNavigationWatcher(platform);
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
  
  function createResultOverlay() {
    // Remove existing
    const existing = document.getElementById('veripulse-result');
    if (existing) existing.remove();
    
    resultOverlay = document.createElement('div');
    resultOverlay.id = 'veripulse-result';
    resultOverlay.style.display = 'none';
    document.body.appendChild(resultOverlay);
  }
  
  function setupAutoScan(platform) {
    console.log(`VeriPulse: Setting up auto-scan for ${platform}`);
    
    // Initial check for existing videos
    setTimeout(() => checkForNewVideo(platform), 1000);
    
    // Watch for new videos and video changes
    const observer = new MutationObserver(() => {
      checkForNewVideo(platform);
    });
    
    observer.observe(document.body, { 
      childList: true, 
      subtree: true,
      attributes: true,
      attributeFilter: ['src', 'currentSrc']
    });
    
    // Platform-specific watchers
    if (platform === 'youtube') {
      setupYouTubeWatcher();
    } else if (platform === 'instagram') {
      setupInstagramWatcher();
    }
  }
  
  function setupYouTubeWatcher() {
    // YouTube uses SPA - watch for video ID changes
    let lastVideoId = getYouTubeVideoId();
    
    setInterval(() => {
      const newVideoId = getYouTubeVideoId();
      if (newVideoId && newVideoId !== lastVideoId) {
        console.log(`VeriPulse: New YouTube video detected: ${newVideoId}`);
        lastVideoId = newVideoId;
        resetAndScan('youtube');
      }
    }, 500);
    
    // Also watch for Shorts scrolling
    document.addEventListener('scroll', debounce(() => {
      if (window.location.href.includes('/shorts/')) {
        checkForNewVideo('youtube');
      }
    }, 300), true);
  }
  
  function setupInstagramWatcher() {
    // Instagram Reels - scroll detection
    let lastScrollTop = 0;
    
    document.addEventListener('scroll', debounce(() => {
      const scrollTop = window.scrollY || document.documentElement.scrollTop;
      if (Math.abs(scrollTop - lastScrollTop) > 200) {
        lastScrollTop = scrollTop;
        checkForNewVideo('instagram');
      }
    }, 300), true);
  }
  
  function setupNavigationWatcher(platform) {
    // Watch for URL changes (SPA navigation)
    let lastUrl = window.location.href;
    
    const urlObserver = new MutationObserver(() => {
      if (window.location.href !== lastUrl) {
        console.log(`VeriPulse: URL changed to ${window.location.href}`);
        lastUrl = window.location.href;
        resetAndScan(platform);
      }
    });
    
    urlObserver.observe(document.body, { childList: true, subtree: true });
    
    // Also listen for popstate (back/forward)
    window.addEventListener('popstate', () => {
      resetAndScan(platform);
    });
  }
  
  function getYouTubeVideoId() {
    const url = window.location.href;
    // Regular video
    const match = url.match(/[?&]v=([^&]+)/);
    if (match) return match[1];
    // Shorts
    const shortsMatch = url.match(/\/shorts\/([^?&]+)/);
    if (shortsMatch) return shortsMatch[1];
    return null;
  }
  
  function checkForNewVideo(platform) {
    const video = findVideo(platform);
    if (!video) return;
    
    // Check if this is a different video
    const videoSrc = video.src || video.currentSrc || video.baseURI;
    const isNewVideo = videoSrc !== currentVideoSrc || video !== currentVideoElement;
    
    if (isNewVideo && video.readyState >= 2 && !video.paused) {
      console.log(`VeriPulse: New video playing - triggering scan`);
      currentVideoSrc = videoSrc;
      currentVideoElement = video;
      
      // Clear any pending scan
      if (scanTimeout) clearTimeout(scanTimeout);
      
      // Delay scan slightly to let video stabilize
      scanTimeout = setTimeout(() => {
        startAutoScan(platform, video);
      }, AUTO_SCAN_DELAY);
    }
    
    // Also watch for video play event
    if (!video._veripulseWatching) {
      video._veripulseWatching = true;
      
      video.addEventListener('play', () => {
        const src = video.src || video.currentSrc || video.baseURI;
        if (src !== currentVideoSrc) {
          currentVideoSrc = src;
          currentVideoElement = video;
          
          if (scanTimeout) clearTimeout(scanTimeout);
          scanTimeout = setTimeout(() => {
            startAutoScan(platform, video);
          }, AUTO_SCAN_DELAY);
        }
      });
      
      video.addEventListener('loadeddata', () => {
        const src = video.src || video.currentSrc || video.baseURI;
        if (src !== currentVideoSrc && !video.paused) {
          currentVideoSrc = src;
          currentVideoElement = video;
          
          if (scanTimeout) clearTimeout(scanTimeout);
          scanTimeout = setTimeout(() => {
            startAutoScan(platform, video);
          }, AUTO_SCAN_DELAY);
        }
      });
    }
  }
  
  function resetAndScan(platform) {
    // Reset state for new video
    currentVideoSrc = null;
    currentVideoElement = null;
    isScanning = false;
    
    if (scanTimeout) clearTimeout(scanTimeout);
    if (resultOverlay) resultOverlay.style.display = 'none';
    
    // Check for video after short delay
    setTimeout(() => checkForNewVideo(platform), 500);
  }
  
  async function startAutoScan(platform, video) {
    // Debounce - don't scan too frequently
    const now = Date.now();
    if (now - lastScanTime < 3000) {
      console.log("VeriPulse: Scan debounced (too soon)");
      return;
    }
    
    if (isScanning) {
      console.log("VeriPulse: Scan already in progress");
      return;
    }
    
    if (!video || video.paused || video.ended) {
      console.log("VeriPulse: Video not playing");
      return;
    }
    
    isScanning = true;
    lastScanTime = now;
    
    console.log("VeriPulse: Starting auto-scan...");
    showResult({ status: 'scanning', message: 'Analyzing video...', progress: 0 });
    
    try {
      // Capture frames
      const frames = await captureFramesWithProgress(video, SCAN_FRAME_COUNT, SCAN_INTERVAL, (progress) => {
        showResult({ status: 'scanning', message: `Analyzing... ${Math.round(progress * 100)}%`, progress });
      });
      
      if (frames.length === 0) {
        throw new Error('Could not capture frames');
      }
      
      // Send to background for analysis
      const response = await chrome.runtime.sendMessage({
        action: 'quickScan',
        frames: frames
      });
      
      if (response.success) {
        const result = response.result;
        
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
      } else {
        throw new Error(response.error || 'Analysis failed');
      }
      
    } catch (error) {
      console.error('VeriPulse scan error:', error);
      showResult({ status: 'error', message: error.message });
    } finally {
      isScanning = false;
    }
  }
  
  function findVideo(platform) {
    let video = null;
    
    switch (platform) {
      case 'youtube':
        // Try main video first, then any video
        video = document.querySelector('video.html5-main-video') || 
                document.querySelector('ytd-player video') ||
                document.querySelector('#shorts-player video') ||
                document.querySelector('video');
        break;
      case 'instagram':
        // Find visible video (for Reels)
        const videos = Array.from(document.querySelectorAll('video'));
        video = videos.find(v => {
          const rect = v.getBoundingClientRect();
          return rect.top >= -100 && rect.top < window.innerHeight / 2;
        }) || videos[0];
        break;
      case 'meet':
      case 'zoom':
      case 'teams':
        // Find the largest video
        const allVideos = Array.from(document.querySelectorAll('video'));
        if (allVideos.length > 0) {
          video = allVideos.reduce((a, b) => {
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
  
  // Debounce helper for scroll events
  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
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
    
    // Auto-hide after 10 seconds
    if (result.status !== 'scanning') {
      setTimeout(() => {
        if (resultOverlay) resultOverlay.style.display = 'none';
      }, 10000);
    }
  }
  
  // Listen for messages from popup (manual re-scan)
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'triggerScan') {
      const platform = detectPlatform();
      if (platform) {
        // Force re-scan
        currentVideoSrc = null;
        currentVideoElement = null;
        isScanning = false;
        const video = findVideo(platform);
        if (video) {
          startAutoScan(platform, video);
          sendResponse({ success: true });
        } else {
          sendResponse({ success: false, error: 'No video found' });
        }
      } else {
        sendResponse({ success: false, error: 'Unsupported platform' });
      }
    }
    return true;
  });
  
})();
