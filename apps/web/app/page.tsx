"use client";

import { useEffect, useRef, useState } from "react";

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [result, setResult] = useState<any>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'live' | 'upload'>('live');
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    return () => {
      if (socket) {
        socket.close();
      }
    };
  }, [socket]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    
    const file = e.target.files[0];
    setIsUploading(true);
    setUploadResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/api/v1/analyze/video', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Analysis failed');
      
      const data = await response.json();
      setUploadResult(data);
    } catch (err) {
      console.error(err);
      setError("Failed to analyze file. Ensure backend is running.");
    } finally {
      setIsUploading(false);
    }
  };

  const startScanning = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: true,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // Initialize WebSocket
      const ws = new WebSocket("ws://localhost:8000/ws/liveness");
      
      ws.onopen = () => {
        console.log("Connected to VeriPulse Engine");
        setIsScanning(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        setResult(data);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        setError("Connection error. Is the backend running?");
        setIsScanning(false);
      };

      ws.onclose = () => {
        console.log("Disconnected");
        setIsScanning(false);
      };

      setSocket(ws);

      // Audio Processing
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(4096, 1, 1);

      source.connect(processor);
      processor.connect(audioContext.destination);

      processor.onaudioprocess = (e) => {
        if (ws.readyState !== WebSocket.OPEN) return;

        // Get raw PCM data (float32)
        const inputData = e.inputBuffer.getChannelData(0);
        
        // Convert to Int16
        const pcm16 = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          const s = Math.max(-1, Math.min(1, inputData[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Convert to Base64
        // We need to convert the buffer to a binary string first
        const bytes = new Uint8Array(pcm16.buffer);
        let binary = "";
        for (let i = 0; i < bytes.byteLength; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        const audioBase64 = btoa(binary);

        // Capture Video Frame
        if (videoRef.current && canvasRef.current) {
          const ctx = canvasRef.current.getContext("2d");
          if (ctx) {
            ctx.drawImage(videoRef.current, 0, 0, 640, 480);
            const imageBase64 = canvasRef.current.toDataURL("image/jpeg", 0.7);

            // Send both
            ws.send(JSON.stringify({
              image: imageBase64,
              audio: audioBase64
            }));
          }
        }
      };

    } catch (err) {
      console.error("Error accessing media devices:", err);
      setError("Could not access camera/microphone.");
    }
  };

  const stopScanning = () => {
    if (socket) {
      socket.close();
      setSocket(null);
    }
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }
    setIsScanning(false);
    setResult(null);
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-zinc-950 text-white p-8">
      <div className="max-w-4xl w-full space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight text-emerald-400">VeriPulse Engine</h1>
          <p className="text-zinc-400">Real-time Multimodal Liveness Detection</p>
          
          <div className="flex justify-center gap-4 mt-4">
            <button 
              onClick={() => setActiveTab('live')}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${activeTab === 'live' ? 'bg-emerald-600 text-white' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'}`}
            >
              Live Camera
            </button>
            <button 
              onClick={() => setActiveTab('upload')}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-all ${activeTab === 'upload' ? 'bg-emerald-600 text-white' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'}`}
            >
              Upload File
            </button>
          </div>
        </div>

        {activeTab === 'live' ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Video Feed */}
          <div className="relative aspect-video bg-black rounded-2xl overflow-hidden border border-zinc-800 shadow-2xl">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover transform scale-x-[-1]"
            />
            <canvas ref={canvasRef} width={640} height={480} className="hidden" />
            
            {/* Overlay */}
            {result?.bbox && (
              <>
                <div
                  className="absolute border-2 border-emerald-500 rounded-lg transition-all duration-100"
                  style={{
                    left: `${(result.bbox[0] / 640) * 100}%`,
                    top: `${(result.bbox[1] / 480) * 100}%`,
                    width: `${(result.bbox[2] / 640) * 100}%`,
                    height: `${(result.bbox[3] / 480) * 100}%`,
                    transform: 'scaleX(-1)' // Mirror fix
                  }}
                />
                {/* Progress Bar for Initial Collection */}
                {result.status === "collecting" && result.progress < 1.0 && (
                  <div className="absolute bottom-4 left-1/2 -translate-x-1/2 w-1/2 bg-black/50 backdrop-blur rounded-full h-2 overflow-hidden border border-zinc-700">
                    <div 
                      className="h-full bg-emerald-500 transition-all duration-200"
                      style={{ width: `${result.progress * 100}%` }}
                    />
                  </div>
                )}
                {result.status === "collecting" && (
                  <div className="absolute top-4 left-4 bg-black/60 backdrop-blur px-3 py-1 rounded-full text-xs font-mono text-emerald-400 border border-emerald-500/30">
                    Analyzing... {Math.round(result.progress * 100)}%
                  </div>
                )}
              </>
            )}

            {/* Final Verdict Overlay */}
            {result?.status === "completed" && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 backdrop-blur-md z-20 animate-in fade-in duration-300">
                <div className="text-6xl mb-6 animate-bounce">
                  {result.classification === "REAL HUMAN" ? "✅" : "🚫"}
                </div>
                <h2 className={`text-4xl font-bold mb-2 tracking-tight ${
                  result.classification === "REAL HUMAN" ? "text-emerald-400" : "text-red-500"
                }`}>
                  {result.classification}
                </h2>
                <div className="flex gap-4 text-sm text-zinc-400 mb-8">
                  <span>Score: {(result.score * 100).toFixed(1)}%</span>
                  <span>•</span>
                  <span>{result.liveness} CONFIDENCE</span>
                </div>
                <button
                  onClick={() => { 
                    stopScanning(); 
                    setTimeout(startScanning, 100); 
                  }}
                  className="px-8 py-3 bg-white text-black hover:bg-zinc-200 rounded-full font-bold transition-all transform hover:scale-105"
                >
                  Start New Scan
                </button>
              </div>
            )}
            
            {!isScanning && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm">
                <button
                  onClick={startScanning}
                  className="px-8 py-4 bg-emerald-600 hover:bg-emerald-500 text-white rounded-full font-semibold transition-all shadow-lg hover:shadow-emerald-500/20"
                >
                  Start Live Scan
                </button>
              </div>
            )}
          </div>

          {/* Metrics Panel */}
          <div className="space-y-6 p-6 bg-zinc-900/50 rounded-2xl border border-zinc-800">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Analysis Results</h2>
              {isScanning && (
                <button onClick={stopScanning} className="text-sm text-red-400 hover:text-red-300">
                  Stop Scanning
                </button>
              )}
            </div>

            {error && (
              <div className="p-4 bg-red-900/20 border border-red-800 rounded-lg text-red-200">
                {error}
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div className="col-span-2 p-4 bg-zinc-900 rounded-xl border border-zinc-800 text-center">
                <div className="text-xs text-zinc-500 uppercase tracking-wider mb-1">Final Classification</div>
                <div className={`text-3xl font-bold ${
                  result?.classification === "REAL HUMAN" ? "text-emerald-400" : 
                  result?.classification?.includes("AI") ? "text-red-500" : "text-yellow-400"
                }`}>
                  {result?.classification || "WAITING..."}
                </div>
              </div>

              <MetricCard 
                label="Liveness Level" 
                value={result?.liveness || "--"} 
                color={result?.liveness === "HIGH" ? "text-emerald-400" : result?.liveness === "LOW" ? "text-red-400" : "text-yellow-400"}
              />
              <MetricCard 
                label="Authenticity Score" 
                value={result?.score ? `${(result.score * 100).toFixed(1)}%` : "--"} 
                color={result?.score > 0.8 ? "text-emerald-400" : result?.score < 0.5 ? "text-red-400" : "text-yellow-400"}
              />
              <MetricCard 
                label="Heart Rate" 
                value={result?.bpm ? `${Math.round(result.bpm)} BPM` : "--"} 
              />
              <div className="col-span-2 grid grid-cols-2 gap-4 p-4 bg-zinc-900 rounded-xl border border-zinc-800">
                <div className="space-y-1">
                  <div className="text-xs text-zinc-500 uppercase tracking-wider">Video Realness</div>
                  <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-500 ${result?.video_score > 0.7 ? 'bg-emerald-500' : 'bg-red-500'}`}
                      style={{ width: `${(result?.video_score || 0) * 100}%` }}
                    />
                  </div>
                  <div className="text-right text-xs text-zinc-400">{(result?.video_score * 100).toFixed(0)}%</div>
                </div>
                <div className="space-y-1">
                  <div className="text-xs text-zinc-500 uppercase tracking-wider">Audio Realness</div>
                  <div className="h-2 bg-zinc-800 rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-500 ${result?.audio_score > 0.7 ? 'bg-emerald-500' : 'bg-red-500'}`}
                      style={{ width: `${(result?.audio_score || 0) * 100}%` }}
                    />
                  </div>
                  <div className="text-right text-xs text-zinc-400">{(result?.audio_score * 100).toFixed(0)}%</div>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <h3 className="text-sm font-medium text-zinc-500">Detection Log</h3>
              <div className="h-32 overflow-y-auto space-y-1 text-sm font-mono bg-black/20 p-2 rounded">
                {result?.reasons?.map((reason: string, i: number) => (
                  <div key={i} className="text-zinc-300">• {reason}</div>
                ))}
                {!result && <div className="text-zinc-600 italic">Waiting for data...</div>}
              </div>
            </div>

            {/* Sentinel-X Agent Monitor */}
            <AgentMonitor />
          </div>
        </div>
        ) : (
          <div className="max-w-2xl mx-auto space-y-8">
            <div className="p-12 border-2 border-dashed border-zinc-800 rounded-2xl bg-zinc-900/30 text-center hover:bg-zinc-900/50 transition-all">
              <input 
                type="file" 
                accept="video/*,audio/*" 
                onChange={handleFileUpload}
                className="hidden" 
                id="file-upload"
                disabled={isUploading}
              />
              <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center gap-4">
                <div className="w-16 h-16 rounded-full bg-zinc-800 flex items-center justify-center text-2xl">
                  {isUploading ? '⏳' : '📁'}
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-white">
                    {isUploading ? 'Analyzing File...' : 'Drop Video or Audio File'}
                  </h3>
                  <p className="text-zinc-500 mt-2">Supports MP4, MOV, MP3, WAV</p>
                </div>
              </label>
            </div>

            {uploadResult && (
              <div className="space-y-6 p-6 bg-zinc-900/50 rounded-2xl border border-zinc-800 animate-in fade-in slide-in-from-bottom-4">
                <div className="text-center">
                  <div className="text-sm text-zinc-500 uppercase tracking-wider mb-2">Analysis Verdict</div>
                  <div className={`text-4xl font-bold ${
                    uploadResult.classification === "REAL HUMAN" ? "text-emerald-400" : "text-red-500"
                  }`}>
                    {uploadResult.classification}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <MetricCard 
                    label="Overall Score" 
                    value={`${(uploadResult.score * 100).toFixed(1)}%`}
                    color={uploadResult.score > 0.8 ? "text-emerald-400" : "text-red-400"}
                  />
                  <MetricCard 
                    label="Audio Score" 
                    value={`${(uploadResult.audio_score * 100).toFixed(1)}%`}
                  />
                  <MetricCard 
                    label="Video Score" 
                    value={`${(uploadResult.video_score * 100).toFixed(1)}%`}
                  />
                  <MetricCard 
                    label="Texture Score" 
                    value={`${(uploadResult.texture_score * 100).toFixed(1)}%`}
                  />
                </div>

                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-zinc-500">Detailed Findings</h3>
                  <div className="bg-black/20 p-4 rounded-lg space-y-2">
                    {uploadResult.reasons.length > 0 ? (
                      uploadResult.reasons.map((r: string, i: number) => (
                        <div key={i} className="flex items-center gap-2 text-red-300">
                          <span>⚠️</span> {r}
                        </div>
                      ))
                    ) : (
                      <div className="flex items-center gap-2 text-emerald-400">
                        <span>✅</span> No anomalies detected.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </main>
  );
}

function MetricCard({ label, value, color = "text-white" }: { label: string, value: string | number, color?: string }) {
  return (
    <div className="p-4 bg-zinc-900 rounded-xl border border-zinc-800">
      <div className="text-xs text-zinc-500 uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
    </div>
  );
}

function AgentMonitor() {
  const [agents, setAgents] = useState<any[]>([]);
  const [timeline, setTimeline] = useState<any[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [agentsRes, timelineRes] = await Promise.all([
          fetch('http://localhost:8000/api/sentinel/agents'),
          fetch('http://localhost:8000/api/sentinel/timeline')
        ]);
        
        if (agentsRes.ok) setAgents(await agentsRes.json());
        if (timelineRes.ok) setTimeline(await timelineRes.json());
      } catch (e) {
        console.error("Failed to fetch Sentinel data", e);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Poll every 5s
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-4 pt-4 border-t border-zinc-800">
      <h3 className="text-lg font-semibold text-blue-400">Sentinel-X Agent Monitor</h3>
      
      <div className="grid grid-cols-2 gap-2">
        {agents.map((agent) => (
          <div key={agent.id} className="p-2 bg-zinc-900 rounded border border-zinc-800 flex justify-between items-center">
            <span className="text-xs text-zinc-400">{agent.id}</span>
            <span className={`text-xs font-bold ${agent.status === 'NORMAL' ? 'text-emerald-500' : 'text-red-500'}`}>
              {agent.status}
            </span>
          </div>
        ))}
      </div>

      <div className="space-y-2">
        <h4 className="text-xs font-medium text-zinc-500 uppercase">Recent Incidents</h4>
        <div className="h-24 overflow-y-auto space-y-1 text-xs font-mono bg-black/20 p-2 rounded">
          {timeline.map((event, i) => (
            <div key={i} className="flex gap-2">
              <span className="text-zinc-500">[{event.time}]</span>
              <span className={event.severity === 'CRITICAL' ? 'text-red-400' : 'text-zinc-300'}>
                {event.event}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
