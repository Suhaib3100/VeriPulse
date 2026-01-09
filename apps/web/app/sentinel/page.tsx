"use client";

import { useEffect, useState, useCallback, useRef } from "react";

/**
 * Component 6: SENTINEL-X DASHBOARD (The Visibility)
 * 
 * Real-time monitoring dashboard showing:
 * - Agent status grid (Normal/Warning/Compromised)
 * - Risk score visualization
 * - Event timeline
 * - Quick remediation actions
 * - WebSocket for real-time updates
 */

interface Agent {
  agent_id: string;
  name: string;
  status: "NORMAL" | "WARNING" | "COMPROMISED";
  risk_score: number;
  last_check: string;
  queries_today: number;
  tables_accessed: string[];
  deployment: string;
  alert_reasons?: string[];
}

interface TimelineEvent {
  timestamp: string;
  agent_id: string;
  event_type: string;
  message: string;
  severity: "info" | "warning" | "critical";
}

interface Stats {
  total_agents: number;
  normal: number;
  warning: number;
  compromised: number;
  incidents_today: number;
}

export default function SentinelDashboard() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [remediating, setRemediating] = useState<string | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [liveAlerts, setLiveAlerts] = useState<string[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const API_BASE = "http://localhost:8000/api/sentinel";
  const WS_URL = "ws://localhost:8000/api/sentinel/ws";

  // WebSocket connection for real-time updates
  useEffect(() => {
    const connectWs = () => {
      try {
        const ws = new WebSocket(WS_URL);
        
        ws.onopen = () => {
          console.log("🛡️ Connected to Sentinel-X WebSocket");
          setWsConnected(true);
        };
        
        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          console.log("WS message:", data);
          
          if (data.type === "REMEDIATION") {
            setLiveAlerts(prev => [`🔥 ${data.agent_id} remediated (${data.duration_ms?.toFixed(0)}ms)`, ...prev.slice(0, 4)]);
            fetchData(); // Refresh data
          } else if (data.type === "STATUS_CHANGE") {
            setLiveAlerts(prev => [`⚠️ ${data.agent_id} status: ${data.status}`, ...prev.slice(0, 4)]);
            fetchData();
          } else if (data.type === "COMPROMISED") {
            setLiveAlerts(prev => [`🚨 ALERT: ${data.agent_id} COMPROMISED!`, ...prev.slice(0, 4)]);
            fetchData();
          }
        };
        
        ws.onclose = () => {
          console.log("WebSocket closed, reconnecting...");
          setWsConnected(false);
          setTimeout(connectWs, 3000);
        };
        
        ws.onerror = () => {
          setWsConnected(false);
        };
        
        wsRef.current = ws;
      } catch (err) {
        console.error("WebSocket error:", err);
      }
    };
    
    connectWs();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Fetch data
  const fetchData = useCallback(async () => {
    try {
      const [agentsRes, timelineRes, statsRes] = await Promise.all([
        fetch(`${API_BASE}/agents`),
        fetch(`${API_BASE}/timeline?limit=20`),
        fetch(`${API_BASE}/stats`),
      ]);

      if (!agentsRes.ok || !timelineRes.ok || !statsRes.ok) {
        throw new Error("Failed to fetch data");
      }

      setAgents(await agentsRes.json());
      setTimeline(await timelineRes.json());
      setStats(await statsRes.json());
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Failed to connect to Sentinel-X API. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  }, []);

  // Poll for updates every 5 seconds
  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [fetchData]);

  // Remediate agent
  const handleRemediate = async (agentId: string) => {
    if (!confirm(`Are you sure you want to remediate ${agentId}? This will revoke tokens, kill the pod, and revert code.`)) {
      return;
    }

    setRemediating(agentId);
    try {
      const response = await fetch(`${API_BASE}/remediate/${agentId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ reason: "Manual trigger from dashboard" }),
      });

      if (!response.ok) throw new Error("Remediation failed");

      const result = await response.json();
      alert(`Remediation ${result.success ? "successful" : "partially failed"} in ${result.total_duration_ms.toFixed(0)}ms`);
      fetchData();
    } catch (err) {
      console.error(err);
      alert("Remediation failed. Check console for details.");
    } finally {
      setRemediating(null);
    }
  };

  // Status badge component
  const StatusBadge = ({ status }: { status: string }) => {
    const colors = {
      NORMAL: "bg-green-500",
      WARNING: "bg-yellow-500",
      COMPROMISED: "bg-red-500 animate-pulse",
    };
    return (
      <span className={`px-3 py-1 rounded-full text-white text-sm font-semibold ${colors[status as keyof typeof colors] || "bg-gray-500"}`}>
        {status}
      </span>
    );
  };

  // Risk score bar
  const RiskBar = ({ score }: { score: number }) => {
    const color = score >= 50 ? "bg-red-500" : score >= 25 ? "bg-yellow-500" : "bg-green-500";
    return (
      <div className="w-full bg-gray-700 rounded-full h-2">
        <div className={`${color} h-2 rounded-full transition-all duration-300`} style={{ width: `${Math.min(score, 100)}%` }} />
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-2xl animate-pulse">🛡️ Loading Sentinel-X...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-3xl">🛡️</span>
            <div>
              <h1 className="text-2xl font-bold">Sentinel-X</h1>
              <p className="text-gray-400 text-sm">AI Agent Behavioral Security</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {/* Connection Status */}
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs ${wsConnected ? 'bg-green-900 text-green-400' : 'bg-gray-700 text-gray-400'}`}>
              <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`}></span>
              {wsConnected ? 'Live' : 'Connecting...'}
            </div>
            <a href="/" className="text-gray-400 hover:text-white transition">← Back to VeriPulse</a>
            {error && (
              <span className="text-red-400 text-sm">⚠️ {error}</span>
            )}
          </div>
        </div>
      </header>

      {/* Live Alerts Banner */}
      {liveAlerts.length > 0 && (
        <div className="bg-red-900/50 border-b border-red-700 px-4 py-2">
          <div className="max-w-7xl mx-auto flex items-center gap-4 overflow-x-auto">
            <span className="text-red-400 font-semibold text-sm whitespace-nowrap">🔴 Live:</span>
            {liveAlerts.map((alert, i) => (
              <span key={i} className="text-red-300 text-sm whitespace-nowrap">{alert}</span>
            ))}
          </div>
        </div>
      )}

      <main className="max-w-7xl mx-auto p-6">
        {/* Stats Cards */}
        {stats && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8">
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="text-3xl font-bold">{stats.total_agents}</div>
              <div className="text-gray-400 text-sm">Total Agents</div>
            </div>
            <div className="bg-gray-800 rounded-lg p-4 border border-green-700">
              <div className="text-3xl font-bold text-green-500">{stats.normal}</div>
              <div className="text-gray-400 text-sm">Normal</div>
            </div>
            <div className="bg-gray-800 rounded-lg p-4 border border-yellow-700">
              <div className="text-3xl font-bold text-yellow-500">{stats.warning}</div>
              <div className="text-gray-400 text-sm">Warning</div>
            </div>
            <div className="bg-gray-800 rounded-lg p-4 border border-red-700">
              <div className="text-3xl font-bold text-red-500">{stats.compromised}</div>
              <div className="text-gray-400 text-sm">Compromised</div>
            </div>
            <div className="bg-gray-800 rounded-lg p-4 border border-purple-700">
              <div className="text-3xl font-bold text-purple-500">{stats.incidents_today}</div>
              <div className="text-gray-400 text-sm">Incidents Today</div>
            </div>
          </div>
        )}

        <div className="grid md:grid-cols-3 gap-6">
          {/* Agents List */}
          <div className="md:col-span-2">
            <h2 className="text-xl font-bold mb-4">🤖 Monitored Agents</h2>
            <div className="space-y-4">
              {agents.map((agent) => (
                <div
                  key={agent.agent_id}
                  className={`bg-gray-800 rounded-lg p-4 border cursor-pointer transition hover:border-blue-500 ${
                    agent.status === "COMPROMISED" ? "border-red-500" :
                    agent.status === "WARNING" ? "border-yellow-500" : "border-gray-700"
                  } ${selectedAgent?.agent_id === agent.agent_id ? "ring-2 ring-blue-500" : ""}`}
                  onClick={() => setSelectedAgent(agent)}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <div className="font-semibold text-lg">{agent.name}</div>
                      <div className="text-gray-400 text-sm">{agent.agent_id}</div>
                    </div>
                    <div className="flex items-center gap-3">
                      <StatusBadge status={agent.status} />
                      {agent.status === "COMPROMISED" && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRemediate(agent.agent_id);
                          }}
                          disabled={remediating === agent.agent_id}
                          className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm font-semibold disabled:opacity-50"
                        >
                          {remediating === agent.agent_id ? "⏳ Remediating..." : "🔥 Remediate"}
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <div className="text-gray-400">Risk Score</div>
                      <div className="flex items-center gap-2">
                        <span className="font-semibold">{agent.risk_score}%</span>
                        <div className="flex-1">
                          <RiskBar score={agent.risk_score} />
                        </div>
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-400">Queries Today</div>
                      <div className="font-semibold">{agent.queries_today.toLocaleString()}</div>
                    </div>
                    <div>
                      <div className="text-gray-400">Deployment</div>
                      <div className="font-semibold">{agent.deployment}</div>
                    </div>
                  </div>

                  {agent.alert_reasons && agent.alert_reasons.length > 0 && (
                    <div className="mt-3 p-2 bg-red-900/30 rounded border border-red-800">
                      <div className="text-red-400 text-xs font-semibold mb-1">⚠️ Alerts:</div>
                      {agent.alert_reasons.map((reason, i) => (
                        <div key={i} className="text-red-300 text-sm">• {reason}</div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Timeline */}
          <div>
            <h2 className="text-xl font-bold mb-4">📜 Event Timeline</h2>
            <div className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
              <div className="max-h-[600px] overflow-y-auto">
                {timeline.map((event, i) => (
                  <div
                    key={i}
                    className={`p-3 border-b border-gray-700 last:border-0 ${
                      event.severity === "critical" ? "bg-red-900/20" :
                      event.severity === "warning" ? "bg-yellow-900/20" : ""
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className={`text-xs font-semibold ${
                        event.severity === "critical" ? "text-red-400" :
                        event.severity === "warning" ? "text-yellow-400" : "text-gray-400"
                      }`}>
                        {event.event_type}
                      </span>
                      <span className="text-xs text-gray-500">
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div className="text-sm text-gray-300">{event.message}</div>
                    <div className="text-xs text-gray-500 mt-1">{event.agent_id}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Agent Details Modal */}
        {selectedAgent && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setSelectedAgent(null)}>
            <div className="bg-gray-800 rounded-lg p-6 max-w-lg w-full mx-4 border border-gray-600" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold">{selectedAgent.name}</h3>
                <button onClick={() => setSelectedAgent(null)} className="text-gray-400 hover:text-white text-2xl">&times;</button>
              </div>
              
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <StatusBadge status={selectedAgent.status} />
                  <span className="text-gray-400">Risk Score: {selectedAgent.risk_score}%</span>
                </div>

                <div>
                  <div className="text-gray-400 text-sm mb-1">Tables Accessed</div>
                  <div className="flex flex-wrap gap-2">
                    {selectedAgent.tables_accessed.map((table) => (
                      <span key={table} className="px-2 py-1 bg-gray-700 rounded text-sm">
                        {table}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <div className="text-gray-400 text-sm mb-1">Last Check</div>
                  <div>{new Date(selectedAgent.last_check).toLocaleString()}</div>
                </div>

                {selectedAgent.status !== "NORMAL" && (
                  <button
                    onClick={() => handleRemediate(selectedAgent.agent_id)}
                    disabled={remediating === selectedAgent.agent_id}
                    className="w-full py-2 bg-red-600 hover:bg-red-700 rounded font-semibold disabled:opacity-50"
                  >
                    {remediating === selectedAgent.agent_id ? "⏳ Remediating..." : "🔥 Trigger Remediation"}
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
