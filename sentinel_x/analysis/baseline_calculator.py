import json
import os
import pandas as pd
from pymongo import MongoClient
import numpy as np

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "sentinel_x"
COLLECTION_NAME = "agent_logs"

class BaselineCalculator:
    def __init__(self):
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]

    def calculate_baseline(self, agent_id, hours=24):
        """
        Analyzes the last N hours of logs to establish a baseline.
        """
        print(f"📊 Calculating baseline for {agent_id}...")
        
        # Fetch logs
        # In a real scenario, we'd filter by time. For now, fetch all.
        cursor = self.collection.find({"Properties.agent_id": agent_id})
        logs = list(cursor)
        
        if not logs:
            print("⚠️ No logs found for agent.")
            return None

        df = pd.DataFrame(logs)
        
        # Calculate Metrics
        baseline = {
            "agent_id": agent_id,
            "generated_at": pd.Timestamp.now().isoformat(),
            "metrics": {}
        }

        # 1. Response Time (DurationMs)
        if "DurationMs" in df.columns:
            baseline["metrics"]["duration_mean"] = float(df["DurationMs"].mean())
            baseline["metrics"]["duration_std"] = float(df["DurationMs"].std())
            baseline["metrics"]["duration_max"] = float(df["DurationMs"].max())

        # 2. Row Count / Volume (Mocked as we don't have row_count in logs yet)
        # We can use 'logs per minute' as a proxy for volume
        # df['TimeGenerated'] = pd.to_datetime(df['TimeGenerated'])
        # logs_per_min = df.set_index('TimeGenerated').resample('1T').count()['Name']
        # baseline["metrics"]["volume_mean"] = float(logs_per_min.mean())
        # baseline["metrics"]["volume_std"] = float(logs_per_min.std())
        
        # 3. Access Patterns (Unique Actions)
        if "Name" in df.columns:
            baseline["metrics"]["allowed_actions"] = df["Name"].unique().tolist()

        # Save to JSON
        filename = f"baseline_{agent_id}.json"
        with open(filename, "w") as f:
            json.dump(baseline, f, indent=4)
            
        print(f"✅ Baseline saved to {filename}")
        return baseline

if __name__ == "__main__":
    # Mock data insertion for testing
    calc = BaselineCalculator()
    calc.calculate_baseline("agent-007")
