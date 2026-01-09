"""
Sentinel-X Configuration

Environment variables and settings for all components.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SentinelConfig:
    """Central configuration for Sentinel-X."""
    
    # Demo Mode - uses simulated data when True
    demo_mode: bool = os.getenv("SENTINEL_DEMO_MODE", "true").lower() == "true"
    
    # MongoDB Settings
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    db_name: str = "sentinel_x"
    
    # Azure Application Insights
    azure_app_insights_connection: str = os.getenv("AZURE_APP_INSIGHTS_CONNECTION", "")
    azure_app_insights_app_id: str = os.getenv("AZURE_APP_INSIGHTS_APP_ID", "")
    azure_app_insights_api_key: str = os.getenv("AZURE_APP_INSIGHTS_API_KEY", "")
    
    # Microsoft Graph API (for Entra ID token revocation)
    azure_tenant_id: str = os.getenv("AZURE_TENANT_ID", "")
    azure_client_id: str = os.getenv("AZURE_CLIENT_ID", "")
    azure_client_secret: str = os.getenv("AZURE_CLIENT_SECRET", "")
    
    # Aliases for backward compatibility
    @property
    def graph_tenant_id(self):
        return self.azure_tenant_id
    
    @property
    def graph_client_id(self):
        return self.azure_client_id
    
    @property
    def graph_client_secret(self):
        return self.azure_client_secret
    
    # Kubernetes Settings
    k8s_namespace: str = os.getenv("K8S_NAMESPACE", "ai-agents")
    k8s_api_server: str = os.getenv("K8S_API_SERVER", "https://kubernetes.default.svc")
    k8s_token: str = os.getenv("K8S_TOKEN", "")
    k8s_in_cluster: bool = os.getenv("K8S_IN_CLUSTER", "false").lower() == "true"
    
    # Git/ArgoCD Settings
    git_repo_url: str = os.getenv("GIT_REPO_URL", "")
    git_token: str = os.getenv("GIT_TOKEN", "")
    argocd_server: str = os.getenv("ARGOCD_SERVER", "https://argocd.example.com")
    argocd_token: str = os.getenv("ARGOCD_TOKEN", "")
    last_known_good_revision: str = os.getenv("LAST_KNOWN_GOOD_REVISION", "main")
    
    # Teams Webhook (for alerts)
    teams_webhook_url: str = os.getenv("TEAMS_WEBHOOK_URL", "")
    
    # Detection Settings
    collection_interval_seconds: int = 30
    baseline_hours: int = 24
    zscore_threshold: float = 3.0  # Standard deviations for anomaly
    
    # Sensitive columns that should NEVER be accessed
    sensitive_columns: List[str] = field(default_factory=lambda: [
        "credit_card", "ssn", "social_security",
        "password", "secret", "api_key", "token",
        "private_key", "encryption_key", "payment_info"
    ])

# Global config instance
config = SentinelConfig()