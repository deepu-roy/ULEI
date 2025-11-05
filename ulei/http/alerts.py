"""
Webhook alerting system for threshold breaches and performance regressions.

Sends HTTP notifications when evaluation thresholds are violated or
performance regressions are detected in online shadow evaluation.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from ulei.core.schemas import EvaluationReport
from ulei.http.models import AlertConfig
from ulei.utils.stats import detect_regressions

logger = logging.getLogger(__name__)


@dataclass
class AlertEvent:
    """Represents an alert event to be sent via webhook."""

    alert_id: str
    alert_type: str  # "threshold_breach", "regression", "system_error"
    severity: str  # "low", "medium", "high", "critical"
    title: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    run_id: Optional[str] = None
    metric_name: Optional[str] = None
    suite_name: Optional[str] = None


class AlertManager:
    """Manages threshold-based alerting with webhook notifications."""

    def __init__(self, config: AlertConfig):
        """
        Initialize alert manager.

        Args:
            config: Alert configuration
        """
        self.config = config
        self.recent_alerts: Dict[str, datetime] = {}  # Alert cooldown tracking
        self.consecutive_violations: Dict[str, int] = {}  # Violation count tracking
        self.baseline_reports: Dict[str, EvaluationReport] = {}  # For regression detection

        # HTTP client for webhook requests
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def check_report_for_alerts(
        self, report: EvaluationReport, previous_report: Optional[EvaluationReport] = None
    ):
        """
        Check evaluation report for alert conditions.

        Args:
            report: Current evaluation report
            previous_report: Previous report for regression detection
        """
        if not self.config.enabled:
            return

        alerts = []

        # Check threshold breaches
        if self.config.alert_on_threshold_breach:
            threshold_alerts = self._check_threshold_breaches(report)
            alerts.extend(threshold_alerts)

        # Check for performance regressions
        if self.config.alert_on_regression and previous_report:
            regression_alerts = self._check_performance_regressions(report, previous_report)
            alerts.extend(regression_alerts)

        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)

    def _check_threshold_breaches(self, report: EvaluationReport) -> List[AlertEvent]:
        """Check for threshold breaches in evaluation report."""
        alerts = []

        for metric_name, passed in report.threshold_status.items():
            if not passed:
                # Check if we need to track consecutive violations
                violation_key = f"{report.suite_name}:{metric_name}"

                if violation_key not in self.consecutive_violations:
                    self.consecutive_violations[violation_key] = 0

                self.consecutive_violations[violation_key] += 1

                # Only alert after required consecutive violations
                if (
                    self.consecutive_violations[violation_key]
                    >= self.config.threshold_violations_required
                ):
                    # Check cooldown
                    if self._is_in_cooldown(violation_key):
                        logger.debug(f"Alert for {violation_key} in cooldown, skipping")
                        continue

                    # Get actual vs expected values
                    actual_score = report.aggregates.get(metric_name, 0.0)
                    # Note: We'd need threshold config to get expected value

                    alert = AlertEvent(
                        alert_id=f"threshold_{report.run_id}_{metric_name}",
                        alert_type="threshold_breach",
                        severity="high",
                        title=f"Threshold Breach: {metric_name}",
                        message=(
                            f"Metric '{metric_name}' has failed threshold validation "
                            f"for {self.consecutive_violations[violation_key]} consecutive evaluations"
                        ),
                        details={
                            "metric_name": metric_name,
                            "suite_name": report.suite_name,
                            "run_id": report.run_id,
                            "actual_score": actual_score,
                            "consecutive_violations": self.consecutive_violations[violation_key],
                            "dataset_size": len(report.results),
                        },
                        timestamp=datetime.utcnow(),
                        run_id=report.run_id,
                        metric_name=metric_name,
                        suite_name=report.suite_name,
                    )

                    alerts.append(alert)

                    # Update cooldown tracking
                    self.recent_alerts[violation_key] = datetime.utcnow()

                    # Reset violation count after alerting
                    self.consecutive_violations[violation_key] = 0
            else:
                # Reset violation count on successful threshold
                violation_key = f"{report.suite_name}:{metric_name}"
                self.consecutive_violations[violation_key] = 0

        return alerts

    def _check_performance_regressions(
        self, current_report: EvaluationReport, previous_report: EvaluationReport
    ) -> List[AlertEvent]:
        """Check for performance regressions between reports."""
        alerts = []

        try:
            # Get all metrics to check
            current_metrics = {r.metric for r in current_report.results}
            previous_metrics = {r.metric for r in previous_report.results}
            common_metrics = current_metrics.intersection(previous_metrics)

            # Detect regressions
            regressions = detect_regressions(
                previous_report.results,
                current_report.results,
                list(common_metrics),
                regression_threshold=5.0,  # 5% decline threshold
                significance_level=0.05,
            )

            for metric_name, comparison in regressions.items():
                regression_key = f"regression_{current_report.suite_name}:{metric_name}"

                # Check cooldown
                if self._is_in_cooldown(regression_key):
                    continue

                severity = (
                    "critical"
                    if comparison.delta_percent < -20
                    else "high"
                    if comparison.delta_percent < -10
                    else "medium"
                )

                alert = AlertEvent(
                    alert_id=f"regression_{current_report.run_id}_{metric_name}",
                    alert_type="regression",
                    severity=severity,
                    title=f"Performance Regression: {metric_name}",
                    message=(
                        f"Metric '{metric_name}' has regressed by "
                        f"{abs(comparison.delta_percent):.1f}% "
                        f"(p-value: {comparison.p_value:.4f})"
                    ),
                    details={
                        "metric_name": metric_name,
                        "suite_name": current_report.suite_name,
                        "current_run_id": current_report.run_id,
                        "previous_run_id": previous_report.run_id,
                        "baseline_score": comparison.baseline_mean,
                        "current_score": comparison.comparison_mean,
                        "delta_percent": comparison.delta_percent,
                        "p_value": comparison.p_value,
                        "sample_sizes": comparison.sample_sizes,
                    },
                    timestamp=datetime.utcnow(),
                    run_id=current_report.run_id,
                    metric_name=metric_name,
                    suite_name=current_report.suite_name,
                )

                alerts.append(alert)
                self.recent_alerts[regression_key] = datetime.utcnow()

        except Exception as e:
            logger.error(f"Error checking for regressions: {e}")

        return alerts

    def _is_in_cooldown(self, alert_key: str) -> bool:
        """Check if alert is in cooldown period."""
        if alert_key not in self.recent_alerts:
            return False

        cooldown_until = self.recent_alerts[alert_key] + timedelta(
            minutes=self.config.cooldown_minutes
        )

        return datetime.utcnow() < cooldown_until

    async def _send_alert(self, alert: AlertEvent):
        """Send alert via webhook."""
        if not self.config.webhook_url:
            logger.warning("No webhook URL configured, alert not sent")
            return

        try:
            # Prepare webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "details": alert.details,
                "source": "ulei",
                "version": "1.0.0",
            }

            # Add optional fields
            if alert.run_id:
                payload["run_id"] = alert.run_id
            if alert.metric_name:
                payload["metric_name"] = alert.metric_name
            if alert.suite_name:
                payload["suite_name"] = alert.suite_name

            logger.info(f"Sending {alert.severity} alert: {alert.title}")

            # Send webhook request
            response = await self.http_client.post(
                self.config.webhook_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "ULEI-AlertManager/1.0.0",
                },
            )

            response.raise_for_status()

            logger.info(f"Alert {alert.alert_id} sent successfully")

        except httpx.HTTPStatusError as e:
            logger.error(f"Webhook returned status {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Failed to send webhook: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending alert: {e}")

    async def send_system_alert(
        self,
        title: str,
        message: str,
        severity: str = "medium",
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Send system-level alert (e.g., service errors, queue overflows).

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            details: Additional details
        """
        alert = AlertEvent(
            alert_id=f"system_{datetime.utcnow().timestamp()}",
            alert_type="system_error",
            severity=severity,
            title=title,
            message=message,
            details=details or {},
            timestamp=datetime.utcnow(),
        )

        await self._send_alert(alert)

    async def test_webhook(self) -> bool:
        """
        Test webhook connectivity.

        Returns:
            True if webhook is reachable, False otherwise
        """
        if not self.config.webhook_url:
            logger.warning("No webhook URL configured for testing")
            return False

        try:
            test_payload = {
                "alert_id": "test_alert",
                "alert_type": "test",
                "severity": "low",
                "title": "ULEI Webhook Test",
                "message": "This is a test alert to verify webhook connectivity",
                "timestamp": datetime.utcnow().isoformat(),
                "details": {"test": True},
                "source": "ulei",
                "version": "1.0.0",
            }

            response = await self.http_client.post(
                self.config.webhook_url,
                json=test_payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "ULEI-AlertManager/1.0.0",
                },
            )

            response.raise_for_status()
            logger.info("Webhook test successful")
            return True

        except Exception as e:
            logger.error(f"Webhook test failed: {e}")
            return False

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get statistics about recent alerts."""
        now = datetime.utcnow()
        cooldown_delta = timedelta(minutes=self.config.cooldown_minutes)

        # Count alerts in cooldown
        in_cooldown = sum(
            1 for alert_time in self.recent_alerts.values() if now - alert_time < cooldown_delta
        )

        # Count active violation tracking
        active_violations = sum(1 for count in self.consecutive_violations.values() if count > 0)

        return {
            "total_alerts_sent": len(self.recent_alerts),
            "alerts_in_cooldown": in_cooldown,
            "active_violation_sequences": active_violations,
            "webhook_configured": self.config.webhook_url is not None,
            "alerting_enabled": self.config.enabled,
            "cooldown_minutes": self.config.cooldown_minutes,
            "threshold_violations_required": self.config.threshold_violations_required,
        }

    async def cleanup_old_tracking(self, days: int = 7):
        """Clean up old alert tracking data."""
        cutoff_time = datetime.utcnow() - timedelta(days=days)

        # Clean up old alert timestamps
        old_keys = [key for key, timestamp in self.recent_alerts.items() if timestamp < cutoff_time]

        for key in old_keys:
            del self.recent_alerts[key]

        logger.info(f"Cleaned up {len(old_keys)} old alert records")

    async def close(self):
        """Close HTTP client and cleanup resources."""
        await self.http_client.aclose()


# Webhook payload schemas for common integrations


class SlackWebhookPayload:
    """Format alert for Slack webhook."""

    @staticmethod
    def format_alert(alert: AlertEvent) -> Dict[str, Any]:
        """Format alert as Slack message."""

        # Choose emoji based on severity
        emoji_map = {"low": "🟡", "medium": "🟠", "high": "🔴", "critical": "🚨"}

        emoji = emoji_map.get(alert.severity, "⚠️")

        # Build message blocks
        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": f"{emoji} {alert.title}"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": alert.message}},
        ]

        # Add details section
        if alert.details:
            detail_lines = []
            for key, value in alert.details.items():
                detail_lines.append(f"*{key.replace('_', ' ').title()}:* {value}")

            blocks.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(detail_lines)}}
            )

        return {"blocks": blocks}


class DiscordWebhookPayload:
    """Format alert for Discord webhook."""

    @staticmethod
    def format_alert(alert: AlertEvent) -> Dict[str, Any]:
        """Format alert as Discord embed."""

        # Color based on severity
        color_map = {
            "low": 0xFFFF00,  # Yellow
            "medium": 0xFF8000,  # Orange
            "high": 0xFF0000,  # Red
            "critical": 0x8B0000,  # Dark red
        }

        embed = {
            "title": alert.title,
            "description": alert.message,
            "color": color_map.get(alert.severity, 0x808080),
            "timestamp": alert.timestamp.isoformat(),
            "footer": {
                "text": "ULEI Alert System",
                "icon_url": "https://example.com/ulei-icon.png",
            },
            "fields": [],
        }

        # Add detail fields
        for key, value in alert.details.items():
            embed["fields"].append(
                {"name": key.replace("_", " ").title(), "value": str(value), "inline": True}
            )

        return {"embeds": [embed]}
