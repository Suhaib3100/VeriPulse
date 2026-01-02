import time
import os
import json
from datetime import datetime, timedelta
from pymongo import MongoClient
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from azure.identity import DefaultAzureCredential
import pandas as pd

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "sentinel_x"
COLLECTION_NAME = "agent_logs"
AZURE_WORKSPACE_ID = os.getenv("AZURE_WORKSPACE_ID", "mock-workspace-id")

class LogCollector:
    def __init__(self):
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        
        # Initialize Azure Client (Mockable)
        if AZURE_WORKSPACE_ID != "mock-workspace-id":
            self.credential = DefaultAzureCredential()
            self.logs_client = LogsQueryClient(self.credential)
        else:
            self.logs_client = None
            print("⚠️ Running in MOCK mode (No Azure Workspace ID provided)")

    def fetch_logs(self):
        """
        Fetches logs from Azure Application Insights for the last 30 seconds.
        """
        query = """
        AppRequests
        | where TimeGenerated > ago(30s)
        | project TimeGenerated, Name, Success, DurationMs, Properties
        """
        
        if self.logs_client:
            try:
                response = self.logs_client.query_workspace(
                    workspace_id=AZURE_WORKSPACE_ID,
                    query=query,
                    timespan=timedelta(seconds=30)
                )
                
                if response.status == LogsQueryStatus.PARTIAL:
                    print("⚠️ Partial data received from Azure")
                
                data = [dict(zip([c.name for c in response.tables[0].columns], row)) 
                        for row in response.tables[0].rows]
                return data
                
            except Exception as e:
                print(f"❌ Error fetching logs: {e}")
                return []
        else:
            # Return Mock Data
            return self._generate_mock_data()

    def _generate_mock_data(self):
        """Generates synthetic logs for testing."""
        import random
        actions = ["get_user_profile", "list_emails", "download_file", "delete_user"]
        
        return [{
            "TimeGenerated": datetime.now().isoformat(),
            "Name": random.choice(actions),
            "Success": True,
            "DurationMs": random.randint(50, 5000),
            "Properties": {"agent_id": "agent-007", "table": "users"}
        } for _ in range(random.randint(1, 5))]

    def save_logs(self, logs):
        """Stores logs in MongoDB."""
        if not logs:
            return
        
        try:
            self.collection.insert_many(logs)
            print(f"✅ Saved {len(logs)} logs to MongoDB")
        except Exception as e:
            print(f"❌ Error saving to MongoDB: {e}")

    def run(self):
        print("🚀 Log Collector Started...")
        while True:
            logs = self.fetch_logs()
            self.save_logs(logs)
            time.sleep(30)

if __name__ == "__main__":
    collector = LogCollector()
    collector.run()
