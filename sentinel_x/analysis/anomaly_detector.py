import time
import json
import os
import pandas as pd
from pymongo import MongoClient
import numpy as np

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "sentinel_x"
COLLECTION_NAME = "agent_logs"

class AnomalyDetector:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.baseline = self._load_baseline()

    def _load_baseline(self):
        filename = f"baseline_{self.agent_id}.json"
        if os.path.exists(filename):
            with open(filename, "r") as f:
                return json.load(f)
        return None

    def analyze_recent_activity(self):
        """
        Analyzes the last 30 seconds of activity against the baseline.
        """
        if not self.baseline:
            print("⚠️ No baseline found. Skipping analysis.")
            return "UNKNOWN"

        # Fetch recent logs (Mocking "last 30s" by taking last 10 records)
        cursor = self.collection.find({"Properties.agent_id": self.agent_id}).sort("TimeGenerated", -1).limit(10)
        recent_logs = list(cursor)
        
        if not recent_logs:
            return "NORMAL"

        anomalies = []

        for log in recent_logs:
            # Check 1: Z-Score (Duration)
            duration = log.get("DurationMs", 0)
            mean = self.baseline["metrics"].get("duration_mean", 0)
            std = self.baseline["metrics"].get("duration_std", 1)
            
            z_score = (duration - mean) / (std + 1e-6) # Avoid div by zero
            
            if z_score > 3: # 3 Sigma
                anomalies.append(f"High Latency (Z={z_score:.2f})")

            # Check 2: Access Pattern (New Action)
            action = log.get("Name")
            allowed = self.baseline["metrics"].get("allowed_actions", [])
            if action not in allowed:
                anomalies.append(f"Unauthorized Action: {action}")

            # Check 3: Semantic (Sensitive Data)
            # Mock check
            if "delete" in action.lower() or "download" in action.lower():
                # If this wasn't in baseline, it's flagged by Check 2.
                # But we can have a hard rule too.
                pass

        if anomalies:
            print(f"🚨 COMPROMISED: {anomalies}")
            return "COMPROMISED"
        
        print("✅ NORMAL")
        return "NORMAL"

    def run(self):
        print(f"🧠 Anomaly Detector running for {self.agent_id}...")
        while True:
            status = self.analyze_recent_activity()
            if status == "COMPROMISED":
                self._trigger_remediation()
            time.sleep(30)

    def _trigger_remediation(self):
        print("⚡ Triggering Auto-Remediation...")
        # Call the remediation module (Component 4)
        # For now, just print
        pass

if __name__ == "__main__":
    detector = AnomalyDetector("agent-007")
    detector.run()
