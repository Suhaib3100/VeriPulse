"""
Component 2: BASELINE CALCULATOR

Runs when an agent is first deployed.
Analyzes 24 hours of NORMAL operation to establish baseline metrics.

Output: baseline_{agent_id}.json
{
    "agent_id": "agent-007",
    "calculated_at": "2026-01-09T10:00:00Z",
    "metrics": {
        "normal_row_count": {"mean": 500, "stddev": 100, "min": 50, "max": 800},
        "normal_response_time_ms": {"mean": 420, "stddev": 150, "min": 100, "max": 1000},
        "tables_accessed": ["customers_emails"],
        "columns_accessed": ["email", "name", "created_at"],
        "sensitive_columns_never_accessed": ["credit_card", "ssn", "password"],
        "normal_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17],
        "query_types": {"SELECT": 0.9, "INSERT": 0.05, "UPDATE": 0.05},
        "avg_queries_per_minute": 2.5
    }
}
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
from pymongo import MongoClient

from sentinel_x.config import config


@dataclass
class BaselineMetrics:
    """Baseline metrics for an agent."""
    agent_id: str
    calculated_at: datetime
    baseline_hours: int
    
    # Row count statistics
    row_count_mean: float
    row_count_stddev: float
    row_count_min: int
    row_count_max: int
    
    # Response time statistics
    response_time_mean: float
    response_time_stddev: float
    response_time_min: float
    response_time_max: float
    
    # Access patterns
    tables_accessed: List[str]
    columns_accessed: List[str]
    sensitive_columns_never_accessed: List[str]
    
    # Temporal patterns
    normal_hours: List[int]  # Hours of day when agent normally operates
    
    # Query patterns
    query_type_distribution: Dict[str, float]
    avg_queries_per_minute: float
    
    # API patterns
    endpoints_accessed: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['calculated_at'] = self.calculated_at.isoformat()
        return d
    
    def save(self, filepath: str = None):
        """Save baseline to JSON file."""
        if filepath is None:
            filepath = f"baseline_{self.agent_id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        print(f"✅ Baseline saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaselineMetrics':
        """Load baseline from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if 'calculated_at' in data:
                data['calculated_at'] = datetime.fromisoformat(data['calculated_at'])
            else:
                data['calculated_at'] = datetime.utcnow()
            return cls(**data)
        except Exception as e:
            print(f"⚠️ Failed to load baseline from {filepath}: {e}")
            return None


class BaselineCalculator:
    """
    Calculates baseline behavior metrics for an agent.
    
    Analyzes historical logs to establish:
    - What's normal data access volume
    - What tables/columns are normally accessed
    - When the agent normally operates
    - What API endpoints it normally calls
    """
    
    def __init__(self):
        try:
            self.client = MongoClient(config.mongo_uri, serverSelectionTimeoutMS=2000)
            self.db = self.client[config.db_name]
            self.logs_collection = self.db["agent_logs"]
            self.baselines_collection = self.db["baselines"]
        except Exception as e:
            print(f"⚠️ BaselineCalculator: MongoDB not available: {e}")
            self.client = None
            self.logs_collection = None
            self.baselines_collection = None
    
    def calculate_baseline(self, agent_id: str, hours: int = 24) -> Optional[BaselineMetrics]:
        """
        Calculate baseline metrics for an agent based on historical data.
        
        Args:
            agent_id: The agent to analyze
            hours: How many hours of historical data to analyze (default: 24)
        
        Returns:
            BaselineMetrics object with calculated baselines
        """
        if not self.client:
            print("MongoDB not available, generating demo baseline")
            return self._generate_demo_baseline(agent_id)
        
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Fetch all logs for this agent in the time period
        logs = list(self.logs_collection.find({
            "agent_id": agent_id,
            "timestamp": {"$gte": since}
        }))
        
        if len(logs) < 10:
            print(f"Not enough data for {agent_id}, generating demo baseline")
            return self._generate_demo_baseline(agent_id)
        
        return self._calculate_from_logs(agent_id, logs, hours)
    
    def _calculate_from_logs(self, agent_id: str, logs: List[Dict], hours: int) -> BaselineMetrics:
        """Calculate baseline from actual log data."""
        
        # Extract metrics
        row_counts = []
        response_times = []
        tables = set()
        columns = set()
        hours_active = set()
        query_types = defaultdict(int)
        endpoints = set()
        
        for log in logs:
            # Row counts
            if log.get('row_count'):
                row_counts.append(log['row_count'])
            
            # Response times
            if log.get('response_time_ms'):
                response_times.append(log['response_time_ms'])
            
            # Tables and columns
            if log.get('tables_accessed'):
                tables.update(log['tables_accessed'])
            if log.get('columns_accessed'):
                columns.update(log['columns_accessed'])
            
            # Hours
            if log.get('timestamp'):
                ts = log['timestamp']
                if isinstance(ts, datetime):
                    hours_active.add(ts.hour)
            
            # Query types
            if log.get('query_type'):
                query_types[log['query_type']] += 1
            
            # Endpoints
            if log.get('endpoint'):
                endpoints.add(log['endpoint'])
        
        # Calculate statistics
        row_counts = row_counts or [0]
        response_times = response_times or [0]
        
        total_queries = sum(query_types.values()) or 1
        query_distribution = {k: v/total_queries for k, v in query_types.items()}
        
        # Determine sensitive columns never accessed
        all_sensitive = set(config.sensitive_columns)
        accessed_sensitive = columns.intersection(all_sensitive)
        never_accessed_sensitive = list(all_sensitive - accessed_sensitive)
        
        return BaselineMetrics(
            agent_id=agent_id,
            calculated_at=datetime.utcnow(),
            baseline_hours=hours,
            row_count_mean=float(np.mean(row_counts)),
            row_count_stddev=float(np.std(row_counts)) or 1.0,
            row_count_min=int(min(row_counts)),
            row_count_max=int(max(row_counts)),
            response_time_mean=float(np.mean(response_times)),
            response_time_stddev=float(np.std(response_times)) or 1.0,
            response_time_min=float(min(response_times)),
            response_time_max=float(max(response_times)),
            tables_accessed=list(tables),
            columns_accessed=list(columns),
            sensitive_columns_never_accessed=never_accessed_sensitive,
            normal_hours=sorted(list(hours_active)) or list(range(9, 18)),
            query_type_distribution=dict(query_distribution),
            avg_queries_per_minute=len(logs) / (hours * 60),
            endpoints_accessed=list(endpoints)
        )
    
    def _generate_demo_baseline(self, agent_id: str) -> BaselineMetrics:
        """Generate a realistic demo baseline for testing."""
        return BaselineMetrics(
            agent_id=agent_id,
            calculated_at=datetime.utcnow(),
            baseline_hours=24,
            row_count_mean=500.0,
            row_count_stddev=100.0,
            row_count_min=50,
            row_count_max=800,
            response_time_mean=420.0,
            response_time_stddev=150.0,
            response_time_min=100.0,
            response_time_max=1000.0,
            tables_accessed=["customers_emails", "email_templates"],
            columns_accessed=["email", "name", "created_at", "template_id"],
            sensitive_columns_never_accessed=["credit_card", "ssn", "password", "api_key"],
            normal_hours=[9, 10, 11, 12, 13, 14, 15, 16, 17],
            query_type_distribution={"SELECT": 0.9, "INSERT": 0.05, "UPDATE": 0.05},
            avg_queries_per_minute=2.5,
            endpoints_accessed=["/api/v1/emails/send", "/api/v1/templates/get"]
        )
    
    def store_baseline(self, baseline: BaselineMetrics):
        """Store baseline in MongoDB."""
        if self.baselines_collection is None:
            return
        
        self.baselines_collection.update_one(
            {"agent_id": baseline.agent_id},
            {"$set": baseline.to_dict()},
            upsert=True
        )
        print(f"✅ Baseline stored for {baseline.agent_id}")
    
    def get_baseline(self, agent_id: str) -> Optional[BaselineMetrics]:
        """Retrieve baseline from MongoDB or file."""
        # Try MongoDB first
        if self.baselines_collection is not None:
            doc = self.baselines_collection.find_one({"agent_id": agent_id})
            if doc:
                doc.pop('_id', None)
                doc['calculated_at'] = doc['calculated_at'] if isinstance(doc['calculated_at'], datetime) else datetime.fromisoformat(doc['calculated_at'])
                return BaselineMetrics(**doc)
        
        # Try file
        filepath = f"baseline_{agent_id}.json"
        if os.path.exists(filepath):
            return BaselineMetrics.load(filepath)
        
        return None


if __name__ == "__main__":
    # Test baseline calculation
    calculator = BaselineCalculator()
    
    # Generate demo baseline
    baseline = calculator.calculate_baseline("agent-007")
    print(f"\nBaseline for {baseline.agent_id}:")
    print(f"  Row count: {baseline.row_count_mean:.0f} ± {baseline.row_count_stddev:.0f}")
    print(f"  Response time: {baseline.response_time_mean:.0f}ms ± {baseline.response_time_stddev:.0f}ms")
    print(f"  Tables: {baseline.tables_accessed}")
    print(f"  Normal hours: {baseline.normal_hours}")
    print(f"  Sensitive never accessed: {baseline.sensitive_columns_never_accessed}")
    
    # Save to file
    baseline.save()
