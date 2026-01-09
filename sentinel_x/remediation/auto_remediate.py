"""
Component 4: AUTO-REMEDIATION (The Response)

When an agent is detected as COMPROMISED, execute in parallel:

Step 1: Call Microsoft Graph API → Revoke Entra tokens
Step 2: Call Kubernetes API → Kill the pod
Step 3: Call Git/ArgoCD API → Revert to clean version
Step 4: Call Teams webhook → Alert security team

Total time: 1.2 seconds (all steps run in parallel)
"""

import asyncio
import aiohttp
import time
import json
from datetime import datetime
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from sentinel_x.config import config


class RemediationStep(Enum):
    REVOKE_TOKENS = "REVOKE_TOKENS"
    KILL_POD = "KILL_POD"
    REVERT_CODE = "REVERT_CODE"
    ALERT_TEAM = "ALERT_TEAM"


@dataclass
class StepResult:
    """Result of a single remediation step."""
    step: RemediationStep
    success: bool
    duration_ms: float
    message: str
    details: Optional[Dict] = None


@dataclass
class RemediationResult:
    """Complete remediation result."""
    agent_id: str
    timestamp: datetime
    total_duration_ms: float
    success: bool
    steps: List[StepResult]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "total_duration_ms": self.total_duration_ms,
            "success": self.success,
            "steps": [
                {
                    "step": s.step.value,
                    "success": s.success,
                    "duration_ms": s.duration_ms,
                    "message": s.message,
                    "details": s.details
                }
                for s in self.steps
            ]
        }


class AutoRemediator:
    """
    Automated threat remediation engine.
    
    Executes 4 remediation steps in parallel to minimize response time.
    Target: Complete all steps in under 1.2 seconds.
    """
    
    def __init__(self):
        self.demo_mode = config.demo_mode
        print(f"✅ AutoRemediator initialized (demo_mode={self.demo_mode})")
    
    async def remediate_async(self, agent_id: str, anomaly_result=None) -> RemediationResult:
        """
        Execute all remediation steps in parallel (async version).
        
        Args:
            agent_id: The agent ID to remediate
            anomaly_result: Optional AnomalyResult with detection details
        
        Returns:
            RemediationResult with all step outcomes
        """
        start_time = time.time()
        print(f"\n🚨 INITIATING AUTO-REMEDIATION for {agent_id}")
        print(f"   Reason: {anomaly_result.reasons if anomaly_result else 'Manual trigger'}")
        
        # Run all 4 steps in parallel
        tasks = [
            self._step1_revoke_tokens(agent_id),
            self._step2_kill_pod(agent_id),
            self._step3_revert_code(agent_id),
            self._step4_alert_team(agent_id, anomaly_result)
        ]
        
        step_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for result in step_results:
            if isinstance(result, Exception):
                processed_results.append(StepResult(
                    step=RemediationStep.ALERT_TEAM,
                    success=False,
                    duration_ms=0,
                    message=str(result)
                ))
            else:
                processed_results.append(result)
        
        total_duration = (time.time() - start_time) * 1000
        overall_success = all(r.success for r in processed_results)
        
        result = RemediationResult(
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
            total_duration_ms=total_duration,
            success=overall_success,
            steps=processed_results
        )
        
        # Print summary
        status = "✅ SUCCESS" if overall_success else "⚠️ PARTIAL"
        print(f"\n{status} Remediation complete in {total_duration:.0f}ms")
        for step in processed_results:
            emoji = "✅" if step.success else "❌"
            print(f"   {emoji} {step.step.value}: {step.message} ({step.duration_ms:.0f}ms)")
        
        # Store result
        await self._store_result(result)
        
        return result
    
    def remediate(self, agent_id: str, anomaly_result=None) -> RemediationResult:
        """Execute remediation (sync wrapper)."""
        return asyncio.run(self.remediate_async(agent_id, anomaly_result))
    
    async def _step1_revoke_tokens(self, agent_id: str) -> StepResult:
        """
        Step 1: Revoke Entra ID tokens via Microsoft Graph API
        
        This immediately invalidates all access tokens for the compromised agent,
        preventing any further authenticated API calls.
        """
        start = time.time()
        
        if self.demo_mode:
            # Simulate API call
            await asyncio.sleep(0.15)  # Simulated network latency
            duration = (time.time() - start) * 1000
            return StepResult(
                step=RemediationStep.REVOKE_TOKENS,
                success=True,
                duration_ms=duration,
                message=f"Revoked tokens for {agent_id}",
                details={"service_principal": agent_id, "tokens_revoked": 3}
            )
        
        try:
            # Get access token for Graph API
            async with aiohttp.ClientSession() as session:
                # Get OAuth token
                token_url = f"https://login.microsoftonline.com/{config.azure_tenant_id}/oauth2/v2.0/token"
                token_data = {
                    "grant_type": "client_credentials",
                    "client_id": config.azure_client_id,
                    "client_secret": config.azure_client_secret,
                    "scope": "https://graph.microsoft.com/.default"
                }
                
                async with session.post(token_url, data=token_data) as resp:
                    token_response = await resp.json()
                    access_token = token_response.get("access_token")
                
                if not access_token:
                    raise Exception("Failed to get Graph API token")
                
                # Revoke sign-in sessions
                # This invalidates all refresh tokens and session tokens
                headers = {"Authorization": f"Bearer {access_token}"}
                revoke_url = f"https://graph.microsoft.com/v1.0/servicePrincipals/{agent_id}/revokeSignInSessions"
                
                async with session.post(revoke_url, headers=headers) as resp:
                    if resp.status in [200, 204]:
                        duration = (time.time() - start) * 1000
                        return StepResult(
                            step=RemediationStep.REVOKE_TOKENS,
                            success=True,
                            duration_ms=duration,
                            message=f"Revoked tokens for {agent_id}",
                            details={"status_code": resp.status}
                        )
                    else:
                        error = await resp.text()
                        raise Exception(f"Graph API error: {error}")
        
        except Exception as e:
            duration = (time.time() - start) * 1000
            return StepResult(
                step=RemediationStep.REVOKE_TOKENS,
                success=False,
                duration_ms=duration,
                message=f"Failed to revoke tokens: {str(e)}"
            )
    
    async def _step2_kill_pod(self, agent_id: str) -> StepResult:
        """
        Step 2: Kill the Kubernetes pod running the compromised agent
        
        This immediately terminates the running container, stopping any
        malicious operations in progress.
        """
        start = time.time()
        
        if self.demo_mode:
            await asyncio.sleep(0.2)  # Simulated K8s API latency
            duration = (time.time() - start) * 1000
            return StepResult(
                step=RemediationStep.KILL_POD,
                success=True,
                duration_ms=duration,
                message=f"Killed pod for {agent_id}",
                details={"pod": f"agent-{agent_id}-pod", "namespace": "production"}
            )
        
        try:
            async with aiohttp.ClientSession() as session:
                # Kubernetes API call to delete pod
                pod_name = f"agent-{agent_id}"
                namespace = config.k8s_namespace
                k8s_url = f"{config.k8s_api_server}/api/v1/namespaces/{namespace}/pods/{pod_name}"
                
                headers = {
                    "Authorization": f"Bearer {config.k8s_token}",
                    "Content-Type": "application/json"
                }
                
                # Delete pod (force kill with grace period 0)
                params = {"gracePeriodSeconds": 0}
                async with session.delete(k8s_url, headers=headers, params=params, ssl=False) as resp:
                    if resp.status in [200, 202, 404]:  # 404 = already deleted
                        duration = (time.time() - start) * 1000
                        return StepResult(
                            step=RemediationStep.KILL_POD,
                            success=True,
                            duration_ms=duration,
                            message=f"Killed pod {pod_name}",
                            details={"pod": pod_name, "status_code": resp.status}
                        )
                    else:
                        error = await resp.text()
                        raise Exception(f"K8s API error: {error}")
        
        except Exception as e:
            duration = (time.time() - start) * 1000
            return StepResult(
                step=RemediationStep.KILL_POD,
                success=False,
                duration_ms=duration,
                message=f"Failed to kill pod: {str(e)}"
            )
    
    async def _step3_revert_code(self, agent_id: str) -> StepResult:
        """
        Step 3: Revert to last known good version via ArgoCD
        
        This triggers ArgoCD to sync the application to the last verified
        safe commit, ensuring any malicious code changes are rolled back.
        """
        start = time.time()
        
        if self.demo_mode:
            await asyncio.sleep(0.3)  # Simulated ArgoCD sync
            duration = (time.time() - start) * 1000
            return StepResult(
                step=RemediationStep.REVERT_CODE,
                success=True,
                duration_ms=duration,
                message=f"Reverted {agent_id} to clean version",
                details={
                    "commit": "abc123def",
                    "branch": "main",
                    "app": f"agent-{agent_id}"
                }
            )
        
        try:
            async with aiohttp.ClientSession() as session:
                # ArgoCD API call to sync application
                app_name = f"agent-{agent_id}"
                argocd_url = f"{config.argocd_server}/api/v1/applications/{app_name}/sync"
                
                headers = {
                    "Authorization": f"Bearer {config.argocd_token}",
                    "Content-Type": "application/json"
                }
                
                # Sync to specific revision (last known good)
                sync_data = {
                    "revision": config.last_known_good_revision,
                    "prune": True,
                    "dryRun": False
                }
                
                async with session.post(argocd_url, headers=headers, json=sync_data) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        duration = (time.time() - start) * 1000
                        return StepResult(
                            step=RemediationStep.REVERT_CODE,
                            success=True,
                            duration_ms=duration,
                            message=f"Reverted {app_name} to clean version",
                            details=result
                        )
                    else:
                        error = await resp.text()
                        raise Exception(f"ArgoCD error: {error}")
        
        except Exception as e:
            duration = (time.time() - start) * 1000
            return StepResult(
                step=RemediationStep.REVERT_CODE,
                success=False,
                duration_ms=duration,
                message=f"Failed to revert code: {str(e)}"
            )
    
    async def _step4_alert_team(self, agent_id: str, anomaly_result=None) -> StepResult:
        """
        Step 4: Alert security team via Microsoft Teams webhook
        
        Sends an immediate alert to the security channel with all relevant
        information about the compromise and remediation actions taken.
        """
        start = time.time()
        
        # Build alert message
        alert = self._build_teams_alert(agent_id, anomaly_result)
        
        if self.demo_mode:
            await asyncio.sleep(0.1)  # Simulated webhook call
            duration = (time.time() - start) * 1000
            return StepResult(
                step=RemediationStep.ALERT_TEAM,
                success=True,
                duration_ms=duration,
                message="Security team alerted via Teams",
                details={"channel": "security-alerts"}
            )
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config.teams_webhook_url,
                    json=alert,
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    if resp.status == 200:
                        duration = (time.time() - start) * 1000
                        return StepResult(
                            step=RemediationStep.ALERT_TEAM,
                            success=True,
                            duration_ms=duration,
                            message="Security team alerted via Teams"
                        )
                    else:
                        raise Exception(f"Teams webhook failed: {resp.status}")
        
        except Exception as e:
            duration = (time.time() - start) * 1000
            return StepResult(
                step=RemediationStep.ALERT_TEAM,
                success=False,
                duration_ms=duration,
                message=f"Failed to alert team: {str(e)}"
            )
    
    def _build_teams_alert(self, agent_id: str, anomaly_result=None) -> Dict:
        """Build Microsoft Teams Adaptive Card alert."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        reasons = []
        risk_score = 0
        if anomaly_result:
            reasons = anomaly_result.reasons if hasattr(anomaly_result, 'reasons') else []
            risk_score = anomaly_result.risk_score if hasattr(anomaly_result, 'risk_score') else 0
        
        # Microsoft Teams Adaptive Card format
        return {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": "FF0000",
            "summary": f"🚨 SENTINEL-X: Agent {agent_id} COMPROMISED",
            "sections": [{
                "activityTitle": f"🚨 Agent Compromise Detected",
                "activitySubtitle": f"Agent ID: {agent_id}",
                "facts": [
                    {"name": "Time", "value": timestamp},
                    {"name": "Risk Score", "value": str(risk_score)},
                    {"name": "Status", "value": "COMPROMISED"},
                    {"name": "Reasons", "value": "\n".join(reasons) or "Unknown"},
                    {"name": "Actions Taken", "value": "Tokens revoked, Pod killed, Code reverted"}
                ],
                "markdown": True
            }],
            "potentialAction": [
                {
                    "@type": "OpenUri",
                    "name": "View Dashboard",
                    "targets": [{"os": "default", "uri": "http://localhost:3000/sentinel"}]
                }
            ]
        }
    
    async def _store_result(self, result: RemediationResult):
        """Store remediation result in database."""
        try:
            from pymongo import MongoClient
            client = MongoClient(config.mongo_uri, serverSelectionTimeoutMS=2000)
            db = client[config.db_name]
            db["remediations"].insert_one(result.to_dict())
        except Exception as e:
            print(f"⚠️ Failed to store remediation result: {e}")


# Standalone test
if __name__ == "__main__":
    print("Testing AutoRemediator...")
    
    remediator = AutoRemediator()
    
    # Test remediation
    result = remediator.remediate("agent-007")
    
    print(f"\nFinal Result:")
    print(json.dumps(result.to_dict(), indent=2, default=str))
