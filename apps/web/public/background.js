// VeriPulse Chrome Extension - Background Service Worker

const API_BASE = "http://localhost:8000/api/v1/veripulse";

// Listen for messages from content script or popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "analyzeFrame") {
    analyzeFrame(request.frameData)
      .then(result => sendResponse({ success: true, result }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true; // Keep channel open for async response
  }
  
  if (request.action === "quickScan") {
    quickScan(request.frames)
      .then(result => sendResponse({ success: true, result }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true;
  }
  
  if (request.action === "getStatus") {
    sendResponse({ status: "ready", version: "1.0.0" });
  }
});

// Quick scan - analyze few frames fast
async function quickScan(frames) {
  const startTime = Date.now();
  
  // Analyze only 3 key frames for speed
  const keyFrames = selectKeyFrames(frames, 3);
  const results = [];
  
  for (const frame of keyFrames) {
    try {
      const result = await analyzeFrame(frame);
      results.push(result);
    } catch (e) {
      console.error("Frame analysis failed:", e);
    }
  }
  
  // Aggregate results
  const verdict = aggregateResults(results);
  verdict.scanTime = Date.now() - startTime;
  
  return verdict;
}

// Select key frames (first, middle, last)
function selectKeyFrames(frames, count) {
  if (frames.length <= count) return frames;
  
  const indices = [];
  for (let i = 0; i < count; i++) {
    indices.push(Math.floor(i * (frames.length - 1) / (count - 1)));
  }
  
  return indices.map(i => frames[i]);
}

// Analyze single frame
async function analyzeFrame(frameData) {
  // Convert base64 to blob
  const response = await fetch(frameData);
  const blob = await response.blob();
  
  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");
  
  const apiResponse = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: formData
  });
  
  if (!apiResponse.ok) {
    throw new Error(`API error: ${apiResponse.status}`);
  }
  
  return await apiResponse.json();
}

// Aggregate multiple frame results
function aggregateResults(results) {
  if (results.length === 0) {
    return {
      verdict: "UNKNOWN",
      confidence: 0,
      isReal: null,
      message: "No frames analyzed"
    };
  }
  
  // Count verdicts
  let realCount = 0;
  let fakeCount = 0;
  let totalConfidence = 0;
  
  for (const r of results) {
    if (r.verdict === "REAL" || r.verdict === "LIKELY_REAL") {
      realCount++;
    } else if (r.verdict === "FAKE" || r.verdict === "LIKELY_FAKE") {
      fakeCount++;
    }
    totalConfidence += r.confidence || 0.5;
  }
  
  const avgConfidence = totalConfidence / results.length;
  const isReal = realCount > fakeCount;
  
  return {
    verdict: isReal ? "REAL" : "FAKE",
    confidence: avgConfidence,
    isReal: isReal,
    realVotes: realCount,
    fakeVotes: fakeCount,
    framesAnalyzed: results.length,
    message: isReal 
      ? `Appears genuine (${realCount}/${results.length} frames)`
      : `Potential deepfake detected (${fakeCount}/${results.length} frames)`
  };
}

// Badge updates
function updateBadge(status) {
  const colors = {
    scanning: "#FFA500",
    real: "#22C55E",
    fake: "#EF4444",
    error: "#6B7280"
  };
  
  const texts = {
    scanning: "...",
    real: "✓",
    fake: "!",
    error: "?"
  };
  
  chrome.action.setBadgeBackgroundColor({ color: colors[status] || colors.error });
  chrome.action.setBadgeText({ text: texts[status] || "" });
}

console.log("VeriPulse background service worker loaded");
