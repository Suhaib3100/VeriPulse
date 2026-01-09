"""
Sentinel-X: AI Agent Behavioral Security System

Monitors AI agents for anomalous behavior and automatically remediates threats.
Works alongside Entra ID - NOT replacing it, but adding the missing behavioral layer.

Components:
1. Log Collector - Reads agent behavior from Azure Application Insights
2. Baseline Calculator - Establishes normal behavior patterns
3. Anomaly Detector - Detects compromised agents using 4 detection methods
4. Auto-Remediation - Revokes tokens, kills pods, reverts code in 1.2 seconds
5. API Server - Serves agent status to dashboard
6. Dashboard Integration - Real-time agent monitoring UI
"""

__version__ = "1.0.0"
# Lazy imports to avoid circular dependencies
def get_log_collector():
    from sentinel_x.collectors.log_collector import LogCollector
    return LogCollector

def get_baseline_calculator():
    from sentinel_x.analysis.baseline_calculator import BaselineCalculator
    return BaselineCalculator

def get_anomaly_detector():
    from sentinel_x.analysis.anomaly_detector import AnomalyDetector
    return AnomalyDetector

def get_auto_remediator():
    from sentinel_x.remediation.auto_remediate import AutoRemediator
    return AutoRemediator

def get_api_router():
    from sentinel_x.api.server import router
    return router