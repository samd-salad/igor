"""HA-backed notification commands.

Sends messages via HA's `notify.*` services — typically a phone push via the
HA Companion app, but also persistent_notification (HA UI banner) and any
other notify service the user has configured.

Targets are discovered at call time so newly-added phones / integrations
work without a code change.
"""
import logging
import threading
import time
import urllib.request
import json
import os
from typing import Optional

from .base import Command
from server.ha_client import HAError, get_client

logger = logging.getLogger(__name__)


# Cache the list of available notify.* services so we don't re-enumerate
# /api/services on every call. Refresh after CACHE_TTL seconds.
_CACHE_TTL = 300
_services_cache: list[str] = []
_services_cache_time: float = 0.0
_cache_lock = threading.Lock()


def _list_notify_services(force: bool = False) -> list[str]:
    """Return all `notify.<service>` names exposed by HA, sorted."""
    global _services_cache, _services_cache_time
    now = time.monotonic()
    with _cache_lock:
        if not force and _services_cache and (now - _services_cache_time) < _CACHE_TTL:
            return list(_services_cache)
    ha = get_client()
    base = ha.base_url
    token = ha.token
    try:
        req = urllib.request.Request(f"{base}/api/services",
                                     headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            services = json.loads(r.read())
    except Exception as e:
        logger.warning(f"Failed to enumerate notify services: {e}")
        return []
    notify_svcs = sorted({
        s_name for svc in services if svc.get("domain") == "notify"
        for s_name in svc.get("services", {}).keys()
    })
    with _cache_lock:
        _services_cache = notify_svcs
        _services_cache_time = now
    return list(notify_svcs)


def _resolve_notify_target(target: str = "") -> Optional[str]:
    """Resolve a friendly target to a notify service name.

    Order:
      1. Empty target → prefer a mobile_app_* service, then persistent_notification, then 'notify' (broadcast)
      2. 'phone' / 'mobile' / 'pphone' → first mobile_app_*
      3. 'banner' / 'persistent' / 'ui' → persistent_notification
      4. 'broadcast' / 'all' → 'notify' (sends to every notify service)
      5. Substring match against service names
    """
    services = _list_notify_services()
    if not services:
        return None
    target_lower = target.strip().lower()

    mobile_apps = [s for s in services if s.startswith("mobile_app_")]

    if not target_lower:
        return mobile_apps[0] if mobile_apps else (
            "persistent_notification" if "persistent_notification" in services else (
                "notify" if "notify" in services else services[0]
            )
        )

    if any(w in target_lower for w in ("phone", "mobile", "pphone")):
        return mobile_apps[0] if mobile_apps else None
    if any(w in target_lower for w in ("banner", "persistent", "ui")):
        return "persistent_notification" if "persistent_notification" in services else None
    if target_lower in ("broadcast", "all"):
        return "notify" if "notify" in services else None

    for s in services:
        if target_lower == s or target_lower in s:
            return s
    return None


class NotifyCommand(Command):
    name = "notify"
    description = (
        "Send a notification — typically a push to the user's phone via the HA Companion app. "
        "Use for reminders, alerts, or confirmations the user might want to see later."
    )

    @property
    def parameters(self) -> dict:
        return {
            "message": {"type": "string", "description": "The notification body"},
            "title": {"type": "string", "description": "Optional title shown above the message"},
            "target": {"type": "string", "description": "'phone' (default), 'banner' for HA UI, 'broadcast' for all targets, or a specific notify service name"},
        }

    @property
    def required_parameters(self) -> list:
        return ["message"]

    def execute(self, message: str, title: str = "", target: str = "") -> str:
        message = message.strip()
        if not message:
            return "Notification message is empty"
        service = _resolve_notify_target(target)
        if not service:
            available = ", ".join(_list_notify_services()) or "(none)"
            return f"No notify service for target '{target}'. Available: {available}"
        data: dict = {"message": message}
        if title:
            data["title"] = title
        try:
            get_client().call_service("notify", service, data)
        except HAError as e:
            return f"Notification failed: {e}"
        return f"Sent notification to {service}"


class ListNotifyTargetsCommand(Command):
    name = "list_notify_targets"
    description = "Show the available notification targets (phones, persistent banner, etc.)."

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, **_) -> str:
        services = _list_notify_services(force=True)
        if not services:
            return "No notify services available"
        return "Notify targets:\n" + "\n".join(f"- {s}" for s in services)
