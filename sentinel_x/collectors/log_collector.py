"""
Component 1: LOG COLLECTOR

Reads agent behavior logs from Azure Application Insights every 30 seconds.
Stores in MongoDB for analysis.

Data collected per agent action:
- API calls made
- Database queries executed
- Tables/columns accessed
- Row counts returned
- Response times
- Timestamps
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from pymongo import MongoClient
from pymongo.collection import Collection

from sentinel_x.config import config


@dataclass
class AgentLog:
    """Represents a single agent action log entry."""
    agent_id: str
    timestamp: datetime
    action_type: str  # "api_call", "db_query", "file_access", etc.
    
    # API call details
    endpoint: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    
    # Database query details
    query_type: Optional[str] = None  # SELECT, INSERT, UPDATE, DELETE
    tables_accessed: Optional[List[str]] = None
    columns_accessed: Optional[List[str]] = None
    row_count: Optional[int] = None
    
    # Resource access
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    
    # Context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    
    # Risk indicators (computed)
    sensitive_data_accessed: bool = False
    unusual_time: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LogCollector:
    """
    Collects agent behavior logs from Azure Application Insights.
    
    In production: Queries Azure Application Insights REST API
    For demo: Generates simulated logs or reads from local file
    """
    
    def __init__(self, demo_mode: bool = True):
        self.demo_mode = demo_mode
        self.running = False
        self._thread: Optional[threading.Thread] = None
        
        # MongoDB connection
        try:
            self.client = MongoClient(config.mongo_uri, serverSelectionTimeoutMS=2000)
            self.db = self.client[config.db_name]
            self.logs_collection: Collection = self.db["agent_logs"]
            self.agents_collection: Collection = self.db["agents"]
            print("✅ LogCollector connected to MongoDB")
        except Exception as e:
            print(f"⚠️ LogCollector: MongoDB not available: {e}")
            self.client = None
            self.logs_collection = None
            self.agents_collection = None
    
    def collect_from_azure(self) -> List[AgentLog]:
        """
        Query Azure Application Insights for recent agent logs.
        
        In production, this would use the Application Insights REST API:
        https://api.applicationinsights.io/v1/apps/{app_id}/query
        """
        if not config.azure_app_insights_app_id:
            return []
        
        # Production code would be:
        # import requests
        # headers = {"x-api-key": config.azure_app_insights_api_key}
        # query = """
        # customEvents
        # | where timestamp > ago(30s)
        # | where customDimensions.agent_id != ""
        # | project timestamp, customDimensions
        # """
        # response = requests.post(
        #     f"https://api.applicationinsights.io/v1/apps/{config.azure_app_insights_app_id}/query",
        #     headers=headers,
        #     json={"query": query}
        # )
        # return self._parse_azure_response(response.json())
        
        return []
    
    def generate_demo_logs(self, agent_id: str = "agent-007") -> List[AgentLog]:
        """Generate realistic demo logs for testing."""
        import random
        
        logs = []
        now = datetime.utcnow()
        
        # Normal operation: agent queries customer emails
        for i in range(random.randint(1, 3)):
            logs.append(AgentLog(
                agent_id=agent_id,
                timestamp=now - timedelta(seconds=random.randint(0, 25)),
                action_type="db_query",
                query_type="SELECT",
                tables_accessed=["customers_emails"],
                columns_accessed=["email", "name", "created_at"],
                row_count=random.randint(100, 500),
                response_time_ms=random.uniform(200, 600)
            ))
        
        # Normal API call
        logs.append(AgentLog(
            agent_id=agent_id,
            timestamp=now - timedelta(seconds=random.randint(0, 25)),
            action_type="api_call",
            endpoint="/api/v1/emails/send",
            method="POST",
            status_code=200,
            response_time_ms=random.uniform(100, 300)
        ))
        
        return logs
    
    def generate_anomalous_logs(self, agent_id: str = "agent-007") -> List[AgentLog]:
        """Generate anomalous logs to simulate a compromised agent."""
        import random
        
        logs = []
        now = datetime.utcnow()
        
        # ANOMALY 1: Massive data exfiltration (100x normal row count)
        logs.append(AgentLog(
            agent_id=agent_id,
            timestamp=now,
            action_type="db_query",
            query_type="SELECT",
            tables_accessed=["customers_emails", "customers_personal"],  # New table!
            columns_accessed=["email", "name", "ssn", "credit_card"],  # Sensitive!
            row_count=50000,  # Way more than normal 500
            response_time_ms=15000,  # Slow due to large query
            sensitive_data_accessed=True
        ))
        
        # ANOMALY 2: Accessing tables it never accessed before
        logs.append(AgentLog(
            agent_id=agent_id,
            timestamp=now,
            action_type="db_query",
            query_type="SELECT",
            tables_accessed=["admin_credentials"],  # Should never access this!
            columns_accessed=["username", "password_hash", "api_key"],
            row_count=100,
            sensitive_data_accessed=True
        ))
        
        # ANOMALY 3: Unusual time (if we're simulating 3 AM)
        logs.append(AgentLog(
            agent_id=agent_id,
            timestamp=now.replace(hour=3),  # 3 AM
            action_type="api_call",
            endpoint="/api/v1/data/export",  # Bulk export endpoint
            method="POST",
            status_code=200,
            response_time_ms=30000,
            unusual_time=True
        ))
        
        return logs
    
    def store_logs(self, logs: List[AgentLog]) -> int:
        """Store logs in MongoDB."""
        if self.logs_collection is None or not logs:
            return 0
        
        try:
            docs = [log.to_dict() for log in logs]
            result = self.logs_collection.insert_many(docs)
            return len(result.inserted_ids)
        except Exception as e:
            print(f"Error storing logs: {e}")
            return 0
    
    def register_agent(self, agent_id: str, metadata: Dict[str, Any] = None):
        """Register a new agent for monitoring."""
        if self.agents_collection is None:
            return
        
        doc = {
            "agent_id": agent_id,
            "registered_at": datetime.utcnow(),
            "status": "NORMAL",
            "last_seen": datetime.utcnow(),
            "metadata": metadata or {}
        }
        
        self.agents_collection.update_one(
            {"agent_id": agent_id},
            {"$set": doc},
            upsert=True
        )
    
    def update_agent_status(self, agent_id: str, status: str, reason: str = None):
        """Update agent status (NORMAL, WARNING, COMPROMISED)."""
        if self.agents_collection is None:
            return
        
        update = {
            "status": status,
            "last_status_change": datetime.utcnow(),
            "last_seen": datetime.utcnow()
        }
        if reason:
            update["status_reason"] = reason
        
        self.agents_collection.update_one(
            {"agent_id": agent_id},
            {"$set": update}
        )
    
    def get_recent_logs(self, agent_id: str, minutes: int = 5) -> List[Dict]:
        """Get recent logs for an agent."""
        if self.logs_collection is None:
            return []
        
        since = datetime.utcnow() - timedelta(minutes=minutes)
        cursor = self.logs_collection.find({
            "agent_id": agent_id,
            "timestamp": {"$gte": since}
        }).sort("timestamp", -1)
        
        return list(cursor)
    
    def _collection_loop(self):
        """Main collection loop - runs every 30 seconds."""
        while self.running:
            try:
                if self.demo_mode:
                    # Generate demo logs
                    logs = self.generate_demo_logs("agent-007")
                    logs.extend(self.generate_demo_logs("agent-DataProcessor"))
                else:
                    # Collect from Azure
                    logs = self.collect_from_azure()
                
                if logs:
                    count = self.store_logs(logs)
                    print(f"📊 Collected {count} logs")
                
            except Exception as e:
                print(f"Error in collection loop: {e}")
            
            time.sleep(config.collection_interval_seconds)
    
    def run(self):
        """Start the collector (blocking)."""
        self.running = True
        print("🔄 LogCollector started")
        self._collection_loop()
    
    def start(self):
        """Start the collector in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        print("🔄 LogCollector started (background)")
    
    def stop(self):
        """Stop the collector."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("⏹️ LogCollector stopped")


if __name__ == "__main__":
    # Test the collector
    collector = LogCollector(demo_mode=True)
    collector.register_agent("agent-007", {"type": "email_processor"})
    
    # Generate and store some logs
    logs = collector.generate_demo_logs("agent-007")
    print(f"Generated {len(logs)} logs")
    
    count = collector.store_logs(logs)
    print(f"Stored {count} logs")
    
    # Get recent logs
    recent = collector.get_recent_logs("agent-007")
    print(f"Recent logs: {len(recent)}")
