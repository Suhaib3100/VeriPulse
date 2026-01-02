from flask import Flask, jsonify
from pymongo import MongoClient
import os

app = Flask(__name__)

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "sentinel_x"
COLLECTION_NAME = "agent_logs"

mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

@app.route('/api/agents', methods=['GET'])
def get_agents():
    """List all monitored agents and their status."""
    # Mock status for now, in real app we'd store status in a separate collection
    agents = collection.distinct("Properties.agent_id")
    return jsonify([{"id": a, "status": "NORMAL"} for a in agents])

@app.route('/api/timeline', methods=['GET'])
def get_timeline():
    """Get incident timeline."""
    # Fetch recent anomalies (mocked)
    return jsonify([
        {"time": "10:00:01", "event": "High Latency Detected", "agent": "agent-007", "severity": "WARNING"},
        {"time": "10:00:05", "event": "Unauthorized Access", "agent": "agent-007", "severity": "CRITICAL"},
        {"time": "10:00:06", "event": "Auto-Remediation Triggered", "agent": "agent-007", "severity": "INFO"}
    ])

@app.route('/api/agent/<agent_id>', methods=['GET'])
def get_agent_details(agent_id):
    """Get details for a specific agent."""
    logs = list(collection.find({"Properties.agent_id": agent_id}).sort("TimeGenerated", -1).limit(50))
    # Convert ObjectId to str
    for log in logs:
        log["_id"] = str(log["_id"])
    return jsonify(logs)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
