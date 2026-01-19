"""Network monitoring for unknown device detection."""
import json
import logging
import subprocess
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DEVICES_FILE = Path(__file__).parent / "known_devices.json"
SCAN_INTERVAL = 300  # 5 minutes
REMINDER_DELAY = 1800  # 30 minutes


@dataclass
class Device:
    mac: str
    ip: str
    name: Optional[str] = None

    def __hash__(self):
        return hash(self.mac.lower())

    def __eq__(self, other):
        if isinstance(other, Device):
            return self.mac.lower() == other.mac.lower()
        return False


@dataclass
class PendingAlert:
    device: Device
    first_seen: float
    reminded: bool = False


def load_devices_data() -> dict:
    """Load the known devices JSON file."""
    if DEVICES_FILE.exists():
        try:
            return json.loads(DEVICES_FILE.read_text())
        except json.JSONDecodeError:
            log.error("Failed to parse known_devices.json")
    return {"devices": [], "pending_alerts": [], "dismissed_macs": []}


def save_devices_data(data: dict):
    """Save the known devices JSON file."""
    DEVICES_FILE.write_text(json.dumps(data, indent=2))


def get_known_devices() -> list[Device]:
    """Get list of known devices."""
    data = load_devices_data()
    return [Device(d["mac"], d["ip"], d.get("name")) for d in data.get("devices", [])]


def get_dismissed_macs() -> set[str]:
    """Get set of dismissed MAC addresses (lowercase)."""
    data = load_devices_data()
    return {mac.lower() for mac in data.get("dismissed_macs", [])}


def is_known_device(mac: str, ip: str) -> bool:
    """Check if a device is known by MAC or IP."""
    mac_lower = mac.lower()
    known = get_known_devices()
    dismissed = get_dismissed_macs()

    if mac_lower in dismissed:
        return True

    for device in known:
        if device.mac.lower() == mac_lower:
            return True
        if device.ip == ip:
            return True

    return False


def add_known_device(mac: str, ip: str, name: str) -> bool:
    """Add a device to the known list."""
    data = load_devices_data()
    mac_lower = mac.lower()

    # Check if already exists
    for d in data["devices"]:
        if d["mac"].lower() == mac_lower:
            # Update existing
            d["ip"] = ip
            d["name"] = name
            save_devices_data(data)
            return True

    # Add new
    data["devices"].append({"mac": mac_lower, "ip": ip, "name": name})

    # Remove from pending if present
    data["pending_alerts"] = [
        p for p in data.get("pending_alerts", [])
        if p["device"]["mac"].lower() != mac_lower
    ]

    save_devices_data(data)
    return True


def remove_known_device(identifier: str) -> bool:
    """Remove a device by MAC, IP, or name."""
    data = load_devices_data()
    identifier_lower = identifier.lower()
    original_count = len(data["devices"])

    data["devices"] = [
        d for d in data["devices"]
        if d["mac"].lower() != identifier_lower
        and d["ip"] != identifier
        and d.get("name", "").lower() != identifier_lower
    ]

    if len(data["devices"]) < original_count:
        save_devices_data(data)
        return True
    return False


def dismiss_alert(mac: str) -> bool:
    """Dismiss an alert without adding to known devices."""
    data = load_devices_data()
    mac_lower = mac.lower()

    # Add to dismissed list
    if mac_lower not in [m.lower() for m in data.get("dismissed_macs", [])]:
        data.setdefault("dismissed_macs", []).append(mac_lower)

    # Remove from pending
    original_count = len(data.get("pending_alerts", []))
    data["pending_alerts"] = [
        p for p in data.get("pending_alerts", [])
        if p["device"]["mac"].lower() != mac_lower
    ]

    save_devices_data(data)
    return len(data.get("pending_alerts", [])) < original_count


def get_pending_alerts() -> list[PendingAlert]:
    """Get list of pending alerts."""
    data = load_devices_data()
    alerts = []
    for p in data.get("pending_alerts", []):
        device = Device(p["device"]["mac"], p["device"]["ip"], p["device"].get("name"))
        alerts.append(PendingAlert(device, p["first_seen"], p.get("reminded", False)))
    return alerts


def add_pending_alert(device: Device):
    """Add a device to pending alerts."""
    data = load_devices_data()
    mac_lower = device.mac.lower()

    # Check if already pending
    for p in data.get("pending_alerts", []):
        if p["device"]["mac"].lower() == mac_lower:
            return  # Already pending

    data.setdefault("pending_alerts", []).append({
        "device": {"mac": device.mac, "ip": device.ip, "name": device.name},
        "first_seen": time.time(),
        "reminded": False
    })
    save_devices_data(data)


def mark_alert_reminded(mac: str):
    """Mark a pending alert as reminded."""
    data = load_devices_data()
    mac_lower = mac.lower()

    for p in data.get("pending_alerts", []):
        if p["device"]["mac"].lower() == mac_lower:
            p["reminded"] = True
            break

    save_devices_data(data)


def clear_reminded_alerts():
    """Remove alerts that have been reminded (cleanup)."""
    data = load_devices_data()
    now = time.time()

    # Keep alerts that haven't been reminded, or were reminded less than an hour ago
    data["pending_alerts"] = [
        p for p in data.get("pending_alerts", [])
        if not p.get("reminded") or (now - p["first_seen"]) < 3600
    ]
    save_devices_data(data)


def scan_network() -> list[Device]:
    """Scan the local network for devices."""
    devices = []

    # Try arp-scan first (most reliable)
    try:
        result = subprocess.run(
            ["sudo", "arp-scan", "--localnet", "--quiet"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            # Parse arp-scan output: IP\tMAC\tVendor
            for line in result.stdout.strip().split('\n'):
                match = re.match(r'(\d+\.\d+\.\d+\.\d+)\s+([0-9a-fA-F:]{17})', line)
                if match:
                    ip, mac = match.groups()
                    devices.append(Device(mac.lower(), ip))
            if devices:
                return devices
    except FileNotFoundError:
        log.debug("arp-scan not found, trying fallback")
    except subprocess.TimeoutExpired:
        log.warning("arp-scan timed out")
    except Exception as e:
        log.warning(f"arp-scan failed: {e}")

    # Fallback to ip neigh
    try:
        result = subprocess.run(
            ["ip", "neigh", "show"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # Parse ip neigh output: IP dev IFACE lladdr MAC STATE
            for line in result.stdout.strip().split('\n'):
                match = re.match(r'(\d+\.\d+\.\d+\.\d+)\s+.*lladdr\s+([0-9a-fA-F:]{17})', line)
                if match:
                    ip, mac = match.groups()
                    # Skip incomplete entries
                    if 'REACHABLE' in line or 'STALE' in line or 'DELAY' in line:
                        devices.append(Device(mac.lower(), ip))
    except Exception as e:
        log.warning(f"ip neigh failed: {e}")

    return devices


def check_for_unknown_devices() -> list[Device]:
    """Scan network and return list of unknown devices."""
    devices = scan_network()
    unknown = []

    for device in devices:
        if not is_known_device(device.mac, device.ip):
            unknown.append(device)

    return unknown


class NetworkMonitor:
    """Background network monitoring with alerts."""

    def __init__(self, alert_callback=None):
        self.alert_callback = alert_callback
        self.last_scan = 0
        self._running = False

    def scan_and_alert(self) -> list[Device]:
        """Perform a scan and trigger alerts for unknown devices."""
        unknown = check_for_unknown_devices()

        for device in unknown:
            # Add to pending alerts
            add_pending_alert(device)

            # Trigger alert callback
            if self.alert_callback:
                self.alert_callback(device)

        return unknown

    def check_reminders(self):
        """Check for pending alerts that need reminders."""
        now = time.time()
        alerts = get_pending_alerts()

        for alert in alerts:
            if not alert.reminded and (now - alert.first_seen) >= REMINDER_DELAY:
                # Time to remind
                mark_alert_reminded(alert.device.mac)
                if self.alert_callback:
                    self.alert_callback(alert.device, is_reminder=True)

    def tick(self):
        """Called periodically by event loop."""
        now = time.time()

        # Check if time for a scan
        if now - self.last_scan >= SCAN_INTERVAL:
            self.last_scan = now
            try:
                self.scan_and_alert()
            except Exception as e:
                log.error(f"Network scan failed: {e}")

        # Check for reminders
        try:
            self.check_reminders()
        except Exception as e:
            log.error(f"Reminder check failed: {e}")

        # Cleanup old reminded alerts
        clear_reminded_alerts()
