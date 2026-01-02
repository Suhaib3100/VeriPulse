from fastapi import APIRouter, HTTPException
from pymongo import MongoClient
import os
from typing import List, Dict, Any

router = APIRouter()

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "sentinel_x"
COLLECTION_NAME = "agent_logs"

# Initialize MongoDB Client
# In a production app, this should probably be a dependency
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]
except Exception as e:
    print(f"Warning: Could not connect to MongoDB: {e}")
    collection = None

@router.get("/agents")
def get_agents():
    """List all monitored agents and their status."""
    if collection is None:
        return []
        
    try:
        # Mock status for now, in real app we'd store status in a separate collection
        agents = collection.distinct("Properties.agent_id")
        return [{"id": a, "status": "NORMAL"} for a in agents]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/timeline")
def get_timeline():
    """Get incident timeline."""
    # Fetch recent anomalies (mocked for now as per original sentinel_x logic)
    return [
        {"time": "10:00:01", "event": "High Latency Detected", "agent": "agent-007", "severity": "WARNING"},
        {"time": "10:00:05", "event": "Unauthorized Access", "agent": "agent-007", "severity": "CRITICAL"},
        {"time": "10:00:06", "event": "Auto-Remediation Triggered", "agent": "agent-007", "severity": "INFO"}
    ]

@router.get("/agent/{agent_id}")
def get_agent_details(agent_id: str):
    """Get details for a specific agent."""
    if collection is None:
        return []

    try:
        logs = list(collection.find({"Properties.agent_id": agent_id}).sort("TimeGenerated", -1).limit(50))
        # Convert ObjectId to str for JSON serialization
        for log in logs:
            log["_id"] = str(log["_id"])
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
