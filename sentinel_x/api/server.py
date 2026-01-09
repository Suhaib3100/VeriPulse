"""
Component 5: API SERVER (The Interface)

FastAPI server providing endpoints for:
- Agent status and monitoring
- Timeline of events
- Manual remediation triggers
- Real-time WebSocket updates

Endpoints:
- GET /sentinel/agents - List all agents with status
- GET /sentinel/agents/{id} - Get specific agent details
- GET /sentinel/timeline - Get event timeline
- POST /sentinel/remediate/{id} - Manually trigger remediation
- WS /sentinel/ws - Real-time status updates
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sentinel_x.config import config

# Initialize router
router = APIRouter(prefix="/sentinel", tags=["Sentinel-X"])

# Try to import MongoDB (optional)
try:
    from pymongo import MongoClient
    mongo_client = MongoClient(config.mongo_uri, serverSelectionTimeoutMS=2000)
    db = mongo_client[config.db_name]
    agents_collection = db["agents"]
    logs_collection = db["agent_logs"]
    anomalies_collection = db["anomalies"]
    remediations_collection = db["remediations"]
    MONGO_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Sentinel API: MongoDB not available: {e}")
    MONGO_AVAILABLE = False


# Demo data for when MongoDB is not available
DEMO_AGENTS = [
    {
        "agent_id": "agent-001",
        "name": "SQL-Query-Agent",
        "status": "NORMAL",
        "risk_score": 5,
        "last_check": datetime.utcnow().isoformat(),
        "queries_today": 1247,
        "tables_accessed": ["orders", "products", "customers"],
        "deployment": "production"
    },
    {
        "agent_id": "agent-002",
        "name": "Report-Generator",
        "status": "NORMAL",
        "risk_score": 12,
        "last_check": datetime.utcnow().isoformat(),
        "queries_today": 523,
        "tables_accessed": ["sales", "revenue", "metrics"],
        "deployment": "production"
    },
    {
        "agent_id": "agent-003",
        "name": "Data-Sync-Agent",
        "status": "WARNING",
        "risk_score": 35,
        "last_check": datetime.utcnow().isoformat(),
        "queries_today": 3891,
        "tables_accessed": ["inventory", "suppliers", "pricing"],
        "deployment": "staging"
    },
    {
        "agent_id": "agent-007",
        "name": "Analytics-Agent",
        "status": "COMPROMISED",
        "risk_score": 85,
        "last_check": datetime.utcnow().isoformat(),
        "queries_today": 50000,
        "tables_accessed": ["users", "credentials", "payment_info", "ssn"],
        "deployment": "production",
        "alert_reasons": [
            "Row count 50x normal baseline",
            "Accessing sensitive columns: ssn, payment_info",
            "Activity at 3:00 AM (unusual hour)"
        ]
    }
]

DEMO_TIMELINE = [
    {
        "timestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
        "agent_id": "agent-007",
        "event_type": "COMPROMISED",
        "message": "Agent marked as COMPROMISED - Risk score 85",
        "severity": "critical"
    },
    {
        "timestamp": (datetime.utcnow() - timedelta(minutes=5, seconds=1)).isoformat(),
        "agent_id": "agent-007",
        "event_type": "REMEDIATION",
        "message": "Auto-remediation initiated - Tokens revoked, Pod killed",
        "severity": "warning"
    },
    {
        "timestamp": (datetime.utcnow() - timedelta(minutes=30)).isoformat(),
        "agent_id": "agent-003",
        "event_type": "WARNING",
        "message": "Elevated query volume detected - Risk score 35",
        "severity": "warning"
    },
    {
        "timestamp": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
        "agent_id": "agent-001",
        "event_type": "CHECK_PASSED",
        "message": "All checks passed - Status NORMAL",
        "severity": "info"
    },
    {
        "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
        "agent_id": "agent-002",
        "event_type": "CHECK_PASSED",
        "message": "All checks passed - Status NORMAL",
        "severity": "info"
    }
]


class AgentResponse(BaseModel):
    agent_id: str
    name: str
    status: str
    risk_score: int
    last_check: str
    queries_today: int = 0
    tables_accessed: List[str] = []
    deployment: str = "unknown"


class TimelineEvent(BaseModel):
    timestamp: str
    agent_id: str
    event_type: str
    message: str
    severity: str


class RemediateRequest(BaseModel):
    reason: str = "Manual trigger"


# WebSocket connections
active_connections: List[WebSocket] = []


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "sentinel-x",
        "mongo_available": MONGO_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/agents", response_model=List[Dict])
async def get_agents():
    """Get all monitored agents with their status."""
    if MONGO_AVAILABLE:
        try:
            agents = list(agents_collection.find({}, {"_id": 0}))
            if agents:
                return agents
        except Exception as e:
            print(f"Error fetching agents: {e}")
    
    # Return demo data
    return DEMO_AGENTS


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get detailed information for a specific agent."""
    if MONGO_AVAILABLE:
        try:
            agent = agents_collection.find_one({"agent_id": agent_id}, {"_id": 0})
            if agent:
                # Get recent logs
                recent_logs = list(logs_collection.find(
                    {"agent_id": agent_id},
                    {"_id": 0}
                ).sort("timestamp", -1).limit(10))
                agent["recent_logs"] = recent_logs
                
                # Get recent anomalies
                anomalies = list(anomalies_collection.find(
                    {"agent_id": agent_id},
                    {"_id": 0}
                ).sort("timestamp", -1).limit(5))
                agent["recent_anomalies"] = anomalies
                
                return agent
        except Exception as e:
            print(f"Error fetching agent: {e}")
    
    # Check demo data
    for agent in DEMO_AGENTS:
        if agent["agent_id"] == agent_id:
            return agent
    
    raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")


@router.get("/timeline", response_model=List[Dict])
async def get_timeline(
    limit: int = Query(default=50, le=200),
    severity: Optional[str] = Query(default=None)
):
    """Get timeline of security events."""
    if MONGO_AVAILABLE:
        try:
            query = {}
            if severity:
                query["severity"] = severity
            
            events = list(anomalies_collection.find(
                query,
                {"_id": 0}
            ).sort("timestamp", -1).limit(limit))
            
            if events:
                # Convert to timeline format
                timeline = []
                for event in events:
                    timeline.append({
                        "timestamp": event.get("timestamp", ""),
                        "agent_id": event.get("agent_id", ""),
                        "event_type": event.get("status", "CHECK"),
                        "message": ", ".join(event.get("reasons", [])),
                        "severity": "critical" if event.get("status") == "COMPROMISED" else 
                                   "warning" if event.get("status") == "WARNING" else "info"
                    })
                return timeline
        except Exception as e:
            print(f"Error fetching timeline: {e}")
    
    # Return demo data
    filtered = DEMO_TIMELINE
    if severity:
        filtered = [e for e in filtered if e["severity"] == severity]
    return filtered[:limit]


@router.post("/remediate/{agent_id}")
async def remediate_agent(agent_id: str, request: RemediateRequest = None):
    """Manually trigger remediation for an agent."""
    try:
        from sentinel_x.remediation.auto_remediate import AutoRemediator
        
        remediator = AutoRemediator()
        result = await remediator.remediate_async(agent_id)
        
        # Broadcast update
        await broadcast_update({
            "type": "REMEDIATION",
            "agent_id": agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "success": result.success,
            "duration_ms": result.total_duration_ms,
            "reason": request.reason if request else "Manual trigger"
        })
        
        return result.to_dict()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get summary statistics."""
    if MONGO_AVAILABLE:
        try:
            total_agents = agents_collection.count_documents({})
            normal = agents_collection.count_documents({"status": "NORMAL"})
            warning = agents_collection.count_documents({"status": "WARNING"})
            compromised = agents_collection.count_documents({"status": "COMPROMISED"})
            
            today = datetime.utcnow().replace(hour=0, minute=0, second=0)
            incidents_today = anomalies_collection.count_documents({
                "status": {"$in": ["WARNING", "COMPROMISED"]},
                "timestamp": {"$gte": today.isoformat()}
            })
            
            return {
                "total_agents": total_agents,
                "normal": normal,
                "warning": warning,
                "compromised": compromised,
                "incidents_today": incidents_today,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            print(f"Error fetching stats: {e}")
    
    # Demo stats
    return {
        "total_agents": len(DEMO_AGENTS),
        "normal": sum(1 for a in DEMO_AGENTS if a["status"] == "NORMAL"),
        "warning": sum(1 for a in DEMO_AGENTS if a["status"] == "WARNING"),
        "compromised": sum(1 for a in DEMO_AGENTS if a["status"] == "COMPROMISED"),
        "incidents_today": 2,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "CONNECTED",
            "message": "Connected to Sentinel-X",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        while True:
            # Wait for messages (heartbeat)
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        if websocket in active_connections:
            active_connections.remove(websocket)


async def broadcast_update(data: Dict):
    """Broadcast update to all connected clients."""
    for connection in active_connections:
        try:
            await connection.send_json(data)
        except Exception:
            pass


# Create standalone app for testing
from fastapi import FastAPI
app = FastAPI(title="Sentinel-X API", version="1.0.0")
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    print("Starting Sentinel-X API server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
