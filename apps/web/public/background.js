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
    fullVideoScan(request.frames)
      .then(result => sendResponse({ success: true, result }))
      .catch(error => sendResponse({ success: false, error: error.message }));
    return true;
  }
  
  if (request.action === "getStatus") {
    sendResponse({ status: "ready", version: "1.0.0" });
  }
});

// Full video scan - analyze ALL frames for maximum accuracy
async function fullVideoScan(frames) {
  const startTime = Date.now();
  
  // Analyze more frames for better accuracy (up to 10)
  const framesToAnalyze = Math.min(frames.length, 10);
  const selectedFrames = selectDistributedFrames(frames, framesToAnalyze);
  const results = [];
  
  console.log(`VeriPulse: Analyzing ${framesToAnalyze} frames for accuracy...`);
  
  // Analyze frames in parallel for speed (batches of 3)
  for (let i = 0; i < selectedFrames.length; i += 3) {
    const batch = selectedFrames.slice(i, i + 3);
    const batchResults = await Promise.all(
      batch.map(frame => analyzeFrame(frame).catch(e => {
        console.error("Frame analysis failed:", e);
        return null;
      }))
    );
    results.push(...batchResults.filter(r => r !== null));
  }
  
  // Aggregate results with weighted scoring
  const verdict = aggregateResults(results);
  verdict.scanTime = Date.now() - startTime;
  
  console.log(`VeriPulse: Scan complete in ${verdict.scanTime}ms - ${verdict.verdict}`);
  
  return verdict;
}

// Select frames distributed across the video
function selectDistributedFrames(frames, count) {
  if (frames.length <= count) return frames;
  
  const selected = [];
  const step = frames.length / count;
  
  for (let i = 0; i < count; i++) {
    const index = Math.min(Math.floor(i * step), frames.length - 1);
    selected.push(frames[index]);
  }
  
  return selected;
}

// Analyze single frame - uses frame-only endpoint (no audio)
async function analyzeFrame(frameData) {
  // Convert base64 data URL to blob
  const response = await fetch(frameData);
  const blob = await response.blob();
  
  const formData = new FormData();
  formData.append("file", blob, "frame.jpg");
  
  // Use the frame-only analysis endpoint (optimized for video frames)
  const apiResponse = await fetch(`${API_BASE}/analyze-frame`, {
    method: "POST",
    body: formData
  });
  
  if (!apiResponse.ok) {
    throw new Error(`API error: ${apiResponse.status}`);
  }
  
  return await apiResponse.json();
}

// Aggregate multiple frame results with improved accuracy
function aggregateResults(results) {
  if (results.length === 0) {
    return {
      verdict: "UNKNOWN",
      confidence: 0,
      isReal: null,
      message: "No frames analyzed"
    };
  }
  
  // Collect all scores and analyze consistency
  const trustScores = [];
  const verdictCounts = { REAL: 0, LIKELY_REAL: 0, UNCERTAIN: 0, LIKELY_FAKE: 0, FAKE: 0 };
  const allReasons = new Map(); // Use Map to count reason frequency
  const componentAverages = {};
  let facesDetected = 0;
  
  for (const r of results) {
    const verdict = r.verdict || 'UNCERTAIN';
    const trustScore = r.trust_score || 0.5;
    
    // Track face detection
    if (r.face_detected) {
      facesDetected++;
    }
    
    trustScores.push(trustScore);
    verdictCounts[verdict] = (verdictCounts[verdict] || 0) + 1;
    
    // Collect reasons with frequency
    if (r.reasons) {
      for (const reason of r.reasons) {
        allReasons.set(reason, (allReasons.get(reason) || 0) + 1);
      }
    }
    
    // Aggregate component scores
    if (r.components) {
      for (const [key, value] of Object.entries(r.components)) {
        if (!componentAverages[key]) componentAverages[key] = [];
        componentAverages[key].push(value);
      }
    }
  }
  
  // Calculate statistics
  const avgTrustScore = trustScores.reduce((a, b) => a + b, 0) / trustScores.length;
  const minTrustScore = Math.min(...trustScores);
  const maxTrustScore = Math.max(...trustScores);
  const scoreVariance = trustScores.reduce((acc, s) => acc + Math.pow(s - avgTrustScore, 2), 0) / trustScores.length;
  
  // High variance might indicate mixed content or unreliable detection
  const isConsistent = scoreVariance < 0.05;
  
  // Count verdict distribution
  const totalFrames = results.length;
  const realVotes = verdictCounts.REAL + verdictCounts.LIKELY_REAL;
  const fakeVotes = verdictCounts.FAKE + verdictCounts.LIKELY_FAKE;
  const uncertainVotes = verdictCounts.UNCERTAIN;
  
  // Determine final verdict based on majority voting AND average score
  let finalVerdict;
  let confidence;
  
  // If most frames are consistent, trust the majority
  if (realVotes > totalFrames * 0.6) {
    finalVerdict = avgTrustScore >= 0.7 ? "REAL" : "LIKELY_REAL";
    confidence = avgTrustScore;
  } else if (fakeVotes > totalFrames * 0.6) {
    finalVerdict = avgTrustScore <= 0.3 ? "FAKE" : "LIKELY_FAKE";
    confidence = 1 - avgTrustScore;
  } else if (uncertainVotes > totalFrames * 0.5) {
    finalVerdict = "UNCERTAIN";
    confidence = 0.5;
  } else {
    // Mixed results - use average trust score
    if (avgTrustScore >= 0.65) {
      finalVerdict = "LIKELY_REAL";
      confidence = avgTrustScore;
    } else if (avgTrustScore >= 0.45) {
      finalVerdict = "UNCERTAIN";
      confidence = 0.5;
    } else {
      finalVerdict = "LIKELY_FAKE";
      confidence = 1 - avgTrustScore;
    }
  }
  
  // Sort reasons by frequency (most common first)
  const sortedReasons = Array.from(allReasons.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([reason]) => reason);
  
  // Calculate average component scores
  const avgComponents = {};
  for (const [key, values] of Object.entries(componentAverages)) {
    avgComponents[key] = (values.reduce((a, b) => a + b, 0) / values.length).toFixed(3);
  }
  
  const isReal = finalVerdict === "REAL" || finalVerdict === "LIKELY_REAL";
  
  // Build message with face detection info
  let message;
  const faceInfo = facesDetected > 0 ? `, ${facesDetected} faces` : `, no faces`;
  
  if (isReal) {
    message = `Video appears genuine (${realVotes}/${totalFrames} frames${faceInfo}, trust: ${(avgTrustScore * 100).toFixed(0)}%)`;
  } else if (finalVerdict === "UNCERTAIN") {
    message = `Inconclusive result (${totalFrames} frames${faceInfo}, trust: ${(avgTrustScore * 100).toFixed(0)}%)`;
  } else {
    message = `Potential deepfake detected (${fakeVotes}/${totalFrames} frames flagged${faceInfo})`;
  }
  
  return {
    verdict: finalVerdict,
    confidence: parseFloat(confidence.toFixed(3)),
    isReal: isReal,
    trustScore: parseFloat(avgTrustScore.toFixed(3)),
    framesAnalyzed: totalFrames,
    facesDetected: facesDetected,
    verdictBreakdown: verdictCounts,
    components: avgComponents,
    consistency: isConsistent ? "high" : "low",
    scoreRange: { min: minTrustScore.toFixed(2), max: maxTrustScore.toFixed(2) },
    reasons: sortedReasons,
    message: message
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
