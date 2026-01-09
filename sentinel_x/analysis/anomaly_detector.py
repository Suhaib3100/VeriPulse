"""
Component 3: ANOMALY DETECTOR (The Brain)

Runs every 30 seconds to analyze agent behavior.
Uses 4 detection methods:

1. Z-Score Analysis: Did row count spike 100x normal? → RED FLAG
2. Access Pattern: Accessing new tables/columns? → RED FLAG
3. Temporal Analysis: Running at 3 AM when normally 9 AM-5 PM? → RED FLAG
4. Semantic Analysis: Trying to access sensitive columns? → RED FLAG

Result: If all 4 pass → "NORMAL", if any fails → "COMPROMISED"
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from pymongo import MongoClient

from sentinel_x.config import config
from sentinel_x.analysis.baseline_calculator import BaselineCalculator, BaselineMetrics


class AgentStatus(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    COMPROMISED = "COMPROMISED"


@dataclass
class AnomalyResult:
    """Result of anomaly detection analysis."""
    agent_id: str
    timestamp: datetime
    status: AgentStatus
    
    # Individual check results
    zscore_passed: bool
    access_pattern_passed: bool
    temporal_passed: bool
    semantic_passed: bool
    
    # Details
    zscore_value: Optional[float] = None
    zscore_metric: Optional[str] = None
    new_tables: Optional[List[str]] = None
    new_columns: Optional[List[str]] = None
    sensitive_accessed: Optional[List[str]] = None
    unusual_hour: Optional[int] = None
    
    # Risk score (0-100)
    risk_score: int = 0
    
    # Human-readable explanation
    reasons: List[str] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "zscore_passed": self.zscore_passed,
            "access_pattern_passed": self.access_pattern_passed,
            "temporal_passed": self.temporal_passed,
            "semantic_passed": self.semantic_passed,
            "zscore_value": self.zscore_value,
            "zscore_metric": self.zscore_metric,
            "new_tables": self.new_tables,
            "new_columns": self.new_columns,
            "sensitive_accessed": self.sensitive_accessed,
            "unusual_hour": self.unusual_hour,
            "risk_score": self.risk_score,
            "reasons": self.reasons
        }


class AnomalyDetector:
    """
    The Brain of Sentinel-X.
    
    Continuously monitors agent behavior and detects anomalies
    using 4 complementary detection methods.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.baseline: Optional[BaselineMetrics] = None
        self.running = False
        self._thread: Optional[threading.Thread] = None
        
        # MongoDB connection
        try:
            self.client = MongoClient(config.mongo_uri, serverSelectionTimeoutMS=2000)
            self.db = self.client[config.db_name]
            self.logs_collection = self.db["agent_logs"]
            self.anomalies_collection = self.db["anomalies"]
            self.agents_collection = self.db["agents"]
            print(f"✅ AnomalyDetector initialized for {agent_id}")
        except Exception as e:
            print(f"⚠️ AnomalyDetector: MongoDB not available: {e}")
            self.client = None
            self.logs_collection = None
            self.anomalies_collection = None
            self.agents_collection = None
        
        # Load or calculate baseline
        self._load_baseline()
    
    def _load_baseline(self):
        """Load baseline for the agent."""
        calculator = BaselineCalculator()
        self.baseline = calculator.get_baseline(self.agent_id)
        
        if not self.baseline:
            print(f"No baseline found for {self.agent_id}, calculating...")
            self.baseline = calculator.calculate_baseline(self.agent_id)
            if self.baseline:
                calculator.store_baseline(self.baseline)
                self.baseline.save()
    
    def analyze(self, logs: List[Dict] = None) -> AnomalyResult:
        """
        Analyze recent logs for anomalies.
        
        Args:
            logs: List of log entries to analyze. If None, fetches recent logs.
        
        Returns:
            AnomalyResult with detection results
        """
        if logs is None:
            logs = self._get_recent_logs()
        
        if not logs:
            return AnomalyResult(
                agent_id=self.agent_id,
                timestamp=datetime.utcnow(),
                status=AgentStatus.NORMAL,
                zscore_passed=True,
                access_pattern_passed=True,
                temporal_passed=True,
                semantic_passed=True,
                risk_score=0,
                reasons=["No recent activity"]
            )
        
        # Run all 4 checks
        zscore_result = self._check_zscore(logs)
        access_result = self._check_access_pattern(logs)
        temporal_result = self._check_temporal(logs)
        semantic_result = self._check_semantic(logs)
        
        # Combine results
        all_passed = all([
            zscore_result[0],
            access_result[0],
            temporal_result[0],
            semantic_result[0]
        ])
        
        # Calculate risk score
        risk_score = 0
        if not zscore_result[0]:
            risk_score += 30
        if not access_result[0]:
            risk_score += 25
        if not temporal_result[0]:
            risk_score += 20
        if not semantic_result[0]:
            risk_score += 40  # Semantic violations are most serious
        
        # Determine status
        if risk_score >= 50:
            status = AgentStatus.COMPROMISED
        elif risk_score >= 25:
            status = AgentStatus.WARNING
        else:
            status = AgentStatus.NORMAL
        
        # Collect reasons
        reasons = []
        if not zscore_result[0]:
            reasons.append(f"Z-Score anomaly: {zscore_result[1]}")
        if not access_result[0]:
            reasons.append(f"Access pattern violation: {access_result[1]}")
        if not temporal_result[0]:
            reasons.append(f"Temporal anomaly: {temporal_result[1]}")
        if not semantic_result[0]:
            reasons.append(f"Sensitive data access: {semantic_result[1]}")
        
        if not reasons:
            reasons.append("All checks passed - Normal operation")
        
        result = AnomalyResult(
            agent_id=self.agent_id,
            timestamp=datetime.utcnow(),
            status=status,
            zscore_passed=zscore_result[0],
            access_pattern_passed=access_result[0],
            temporal_passed=temporal_result[0],
            semantic_passed=semantic_result[0],
            zscore_value=zscore_result[2] if len(zscore_result) > 2 else None,
            zscore_metric=zscore_result[3] if len(zscore_result) > 3 else None,
            new_tables=access_result[2] if len(access_result) > 2 else None,
            new_columns=access_result[3] if len(access_result) > 3 else None,
            sensitive_accessed=semantic_result[2] if len(semantic_result) > 2 else None,
            unusual_hour=temporal_result[2] if len(temporal_result) > 2 else None,
            risk_score=risk_score,
            reasons=reasons
        )
        
        # Store result
        self._store_result(result)
        
        return result
    
    def _check_zscore(self, logs: List[Dict]) -> Tuple[bool, str, float, str]:
        """
        Check 1: Z-Score Analysis
        
        Detects statistical anomalies in metrics like row counts and response times.
        A z-score > 3 indicates the value is 3+ standard deviations from normal.
        """
        if not self.baseline:
            return (True, "No baseline", 0, "")
        
        # Check row counts
        row_counts = [log.get('row_count', 0) for log in logs if log.get('row_count')]
        if row_counts:
            max_rows = max(row_counts)
            mean = self.baseline.row_count_mean
            std = self.baseline.row_count_stddev or 1
            zscore = (max_rows - mean) / std
            
            if zscore > config.zscore_threshold:
                return (
                    False,
                    f"Row count {max_rows} is {zscore:.1f} std devs above normal ({mean:.0f})",
                    zscore,
                    "row_count"
                )
        
        # Check response times
        response_times = [log.get('response_time_ms', 0) for log in logs if log.get('response_time_ms')]
        if response_times:
            max_time = max(response_times)
            mean = self.baseline.response_time_mean
            std = self.baseline.response_time_stddev or 1
            zscore = (max_time - mean) / std
            
            if zscore > config.zscore_threshold:
                return (
                    False,
                    f"Response time {max_time}ms is {zscore:.1f} std devs above normal ({mean:.0f}ms)",
                    zscore,
                    "response_time"
                )
        
        return (True, "Within normal range", 0, "")
    
    def _check_access_pattern(self, logs: List[Dict]) -> Tuple[bool, str, List[str], List[str]]:
        """
        Check 2: Access Pattern Analysis
        
        Detects access to new tables or columns that were never accessed in baseline.
        """
        if not self.baseline:
            return (True, "No baseline", [], [])
        
        baseline_tables = set(self.baseline.tables_accessed)
        baseline_columns = set(self.baseline.columns_accessed)
        
        current_tables = set()
        current_columns = set()
        
        for log in logs:
            if log.get('tables_accessed'):
                current_tables.update(log['tables_accessed'])
            if log.get('columns_accessed'):
                current_columns.update(log['columns_accessed'])
        
        new_tables = list(current_tables - baseline_tables)
        new_columns = list(current_columns - baseline_columns)
        
        if new_tables:
            return (
                False,
                f"Accessing new tables: {new_tables}",
                new_tables,
                new_columns
            )
        
        # New columns are less severe but still notable
        if new_columns and len(new_columns) > 3:
            return (
                False,
                f"Accessing many new columns: {new_columns}",
                new_tables,
                new_columns
            )
        
        return (True, "Access pattern normal", [], [])
    
    def _check_temporal(self, logs: List[Dict]) -> Tuple[bool, str, int]:
        """
        Check 3: Temporal Analysis
        
        Detects activity at unusual times (e.g., 3 AM when normally 9 AM-5 PM).
        """
        if not self.baseline:
            return (True, "No baseline", 0)
        
        normal_hours = set(self.baseline.normal_hours)
        
        for log in logs:
            timestamp = log.get('timestamp')
            if isinstance(timestamp, datetime):
                hour = timestamp.hour
                if hour not in normal_hours:
                    return (
                        False,
                        f"Activity at {hour}:00 (normal hours: {sorted(normal_hours)})",
                        hour
                    )
        
        return (True, "Activity within normal hours", 0)
    
    def _check_semantic(self, logs: List[Dict]) -> Tuple[bool, str, List[str]]:
        """
        Check 4: Semantic Analysis
        
        Detects access to sensitive columns that should NEVER be accessed.
        This is the most critical check.
        """
        if not self.baseline:
            return (True, "No baseline", [])
        
        sensitive_never_accessed = set(self.baseline.sensitive_columns_never_accessed)
        all_sensitive = set(config.sensitive_columns)
        
        accessed_sensitive = []
        
        for log in logs:
            columns = log.get('columns_accessed', [])
            for col in columns:
                if col in sensitive_never_accessed or col in all_sensitive:
                    accessed_sensitive.append(col)
            
            # Also check for explicit sensitive flag
            if log.get('sensitive_data_accessed'):
                return (
                    False,
                    "Explicit sensitive data access detected",
                    list(set(accessed_sensitive)) or ["unknown"]
                )
        
        if accessed_sensitive:
            return (
                False,
                f"Accessed sensitive columns: {list(set(accessed_sensitive))}",
                list(set(accessed_sensitive))
            )
        
        return (True, "No sensitive data access", [])
    
    def _get_recent_logs(self, seconds: int = 30) -> List[Dict]:
        """Get logs from the last N seconds."""
        if self.logs_collection is None:
            return []
        
        since = datetime.utcnow() - timedelta(seconds=seconds)
        cursor = self.logs_collection.find({
            "agent_id": self.agent_id,
            "timestamp": {"$gte": since}
        })
        
        return list(cursor)
    
    def _store_result(self, result: AnomalyResult):
        """Store anomaly detection result."""
        if self.anomalies_collection is None:
            return
        
        self.anomalies_collection.insert_one(result.to_dict())
        
        # Update agent status if changed
        if self.agents_collection is not None:
            self.agents_collection.update_one(
                {"agent_id": self.agent_id},
                {"$set": {
                    "status": result.status.value,
                    "risk_score": result.risk_score,
                    "last_check": result.timestamp,
                    "last_reasons": result.reasons
                }}
            )
    
    def _detection_loop(self):
        """Main detection loop - runs every 30 seconds."""
        while self.running:
            try:
                result = self.analyze()
                
                status_emoji = {
                    AgentStatus.NORMAL: "✅",
                    AgentStatus.WARNING: "⚠️",
                    AgentStatus.COMPROMISED: "🚨"
                }
                
                print(f"{status_emoji[result.status]} [{self.agent_id}] Status: {result.status.value} (Risk: {result.risk_score})")
                
                if result.status == AgentStatus.COMPROMISED:
                    print(f"   🔴 ALERT: {result.reasons}")
                    # Trigger auto-remediation
                    self._trigger_remediation(result)
                
            except Exception as e:
                print(f"Error in detection loop: {e}")
            
            time.sleep(config.collection_interval_seconds)
    
    def _trigger_remediation(self, result: AnomalyResult):
        """Trigger auto-remediation when compromised detected."""
        # Import here to avoid circular imports
        try:
            from sentinel_x.remediation.auto_remediate import AutoRemediator
            remediator = AutoRemediator()
            remediator.remediate(self.agent_id, result)
        except ImportError:
            print(f"⚠️ Auto-remediation not available")
        except Exception as e:
            print(f"Error triggering remediation: {e}")
    
    def run(self):
        """Start the detector (blocking)."""
        self.running = True
        print(f"🔍 AnomalyDetector started for {self.agent_id}")
        self._detection_loop()
    
    def start(self):
        """Start the detector in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True)
        self._thread.start()
        print(f"🔍 AnomalyDetector started for {self.agent_id} (background)")
    
    def stop(self):
        """Stop the detector."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        print(f"⏹️ AnomalyDetector stopped for {self.agent_id}")


if __name__ == "__main__":
    # Test the anomaly detector
    from sentinel_x.collectors.log_collector import LogCollector
    
    # Initialize
    detector = AnomalyDetector("agent-007")
    collector = LogCollector(demo_mode=True)
    
    # Test with normal logs
    print("\n=== Testing with NORMAL logs ===")
    normal_logs = collector.generate_demo_logs("agent-007")
    result = detector.analyze(normal_logs)
    print(f"Status: {result.status.value}")
    print(f"Risk Score: {result.risk_score}")
    print(f"Reasons: {result.reasons}")
    
    # Test with anomalous logs
    print("\n=== Testing with ANOMALOUS logs ===")
    anomalous_logs = collector.generate_anomalous_logs("agent-007")
    result = detector.analyze(anomalous_logs)
    print(f"Status: {result.status.value}")
    print(f"Risk Score: {result.risk_score}")
    print(f"Reasons: {result.reasons}")
