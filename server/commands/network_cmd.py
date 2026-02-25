"""Network monitoring commands for detecting unknown devices."""
import json
import logging
import subprocess
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .base import Command

logger = logging.getLogger(__name__)

# Data file for known devices
DEVICES_FILE = Path(__file__).parent.parent.parent / "data" / "known_devices.json"


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


# --- Data persistence functions ---

def _load_devices_data() -> dict:
    """Load the known devices JSON file."""
    if DEVICES_FILE.exists():
        try:
            return json.loads(DEVICES_FILE.read_text())
        except json.JSONDecodeError:
            logger.error("Failed to parse known_devices.json")
    return {"devices": [], "pending_alerts": [], "dismissed_macs": []}


def _save_devices_data(data: dict):
    """Save the known devices JSON file."""
    DEVICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEVICES_FILE.write_text(json.dumps(data, indent=2))


# --- Device management functions ---

def get_known_devices() -> list[Device]:
    """Get list of known devices."""
    data = _load_devices_data()
    return [Device(d["mac"], d["ip"], d.get("name")) for d in data.get("devices", [])]


def get_dismissed_macs() -> set[str]:
    """Get set of dismissed MAC addresses (lowercase)."""
    data = _load_devices_data()
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
    data = _load_devices_data()
    mac_lower = mac.lower()

    # Check if already exists
    for d in data["devices"]:
        if d["mac"].lower() == mac_lower:
            # Update existing
            d["ip"] = ip
            d["name"] = name
            _save_devices_data(data)
            return True

    # Add new
    data["devices"].append({"mac": mac_lower, "ip": ip, "name": name})

    # Remove from pending if present
    data["pending_alerts"] = [
        p for p in data.get("pending_alerts", [])
        if p["device"]["mac"].lower() != mac_lower
    ]

    _save_devices_data(data)
    return True


def remove_known_device(identifier: str) -> bool:
    """Remove a device by MAC, IP, or name."""
    data = _load_devices_data()
    identifier_lower = identifier.lower()
    original_count = len(data["devices"])

    data["devices"] = [
        d for d in data["devices"]
        if d["mac"].lower() != identifier_lower
        and d["ip"] != identifier
        and d.get("name", "").lower() != identifier_lower
    ]

    if len(data["devices"]) < original_count:
        _save_devices_data(data)
        return True
    return False


def dismiss_alert(mac: str) -> bool:
    """Dismiss an alert without adding to known devices."""
    data = _load_devices_data()
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

    _save_devices_data(data)
    return len(data.get("pending_alerts", [])) < original_count


def get_pending_alerts() -> list[PendingAlert]:
    """Get list of pending alerts."""
    data = _load_devices_data()
    alerts = []
    for p in data.get("pending_alerts", []):
        device = Device(p["device"]["mac"], p["device"]["ip"], p["device"].get("name"))
        alerts.append(PendingAlert(device, p["first_seen"], p.get("reminded", False)))
    return alerts


def add_pending_alert(device: Device):
    """Add a device to pending alerts."""
    data = _load_devices_data()
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
    _save_devices_data(data)


# --- Network scanning ---

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
            for line in result.stdout.strip().split('\n'):
                match = re.match(r'(\d+\.\d+\.\d+\.\d+)\s+([0-9a-fA-F:]{17})', line)
                if match:
                    ip, mac = match.groups()
                    devices.append(Device(mac.lower(), ip))
            if devices:
                return devices
    except FileNotFoundError:
        logger.debug("arp-scan not found, trying fallback")
    except subprocess.TimeoutExpired:
        logger.warning("arp-scan timed out")
    except Exception as e:
        logger.warning(f"arp-scan failed: {e}")

    # Fallback to ip neigh
    try:
        result = subprocess.run(
            ["ip", "neigh", "show"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                match = re.match(r'(\d+\.\d+\.\d+\.\d+)\s+.*lladdr\s+([0-9a-fA-F:]{17})', line)
                if match:
                    ip, mac = match.groups()
                    if 'REACHABLE' in line or 'STALE' in line or 'DELAY' in line:
                        devices.append(Device(mac.lower(), ip))
    except Exception as e:
        logger.warning(f"ip neigh failed: {e}")

    return devices


# --- Command classes ---

class ListKnownDevicesCommand(Command):
    name = "list_known_devices"
    description = "List all known/whitelisted devices on the network"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, **_) -> str:
        devices = get_known_devices()
        if not devices:
            return "No known devices configured"

        lines = ["Known devices:"]
        for d in devices:
            lines.append(f"- {d.name or 'unnamed'}: {d.ip} ({d.mac})")
        return "\n".join(lines)


class ScanNetworkCommand(Command):
    name = "scan_network"
    description = "Scan the network now and report any unknown devices"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, **_) -> str:
        devices = scan_network()
        if not devices:
            return "No devices found on network (scan may have failed)"

        known_count = 0
        unknown = []

        for d in devices:
            if is_known_device(d.mac, d.ip):
                known_count += 1
            else:
                unknown.append(d)
                add_pending_alert(d)

        if unknown:
            lines = [f"Found {len(devices)} devices. {len(unknown)} unknown:"]
            for d in unknown:
                lines.append(f"- {d.ip} ({d.mac})")
            return "\n".join(lines)
        else:
            return f"Found {len(devices)} devices, all known"


class PendingNetworkAlertsCommand(Command):
    name = "pending_network_alerts"
    description = "List unacknowledged unknown devices that were detected"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, **_) -> str:
        alerts = get_pending_alerts()
        if not alerts:
            return "No pending network alerts"

        lines = ["Pending unknown devices:"]
        for alert in alerts:
            status = " (reminded)" if alert.reminded else ""
            lines.append(f"- {alert.device.ip} ({alert.device.mac}){status}")
        return "\n".join(lines)


class AddKnownDeviceCommand(Command):
    name = "add_known_device"
    description = "Add a device to the known/whitelist by MAC address"

    @property
    def parameters(self) -> dict:
        return {
            "mac": {
                "type": "string",
                "description": "MAC address of the device (e.g., 'aa:bb:cc:dd:ee:ff')"
            },
            "ip": {
                "type": "string",
                "description": "IP address of the device (e.g., '192.168.0.50')"
            },
            "name": {
                "type": "string",
                "description": "Friendly name for the device (e.g., 'Guest Phone')"
            }
        }

    def execute(self, mac: str, ip: str, name: str) -> str:
        mac = mac.strip().lower()
        ip = ip.strip()
        name = name.strip()

        if not mac or not ip or not name:
            return "MAC address, IP, and name are all required"

        if add_known_device(mac, ip, name):
            return f"Added '{name}' ({mac}) to known devices"
        else:
            return "Failed to add device"


class RemoveKnownDeviceCommand(Command):
    name = "remove_known_device"
    description = "Remove a device from the known/whitelist"

    @property
    def parameters(self) -> dict:
        return {
            "identifier": {
                "type": "string",
                "description": "MAC address, IP address, or name of the device to remove"
            }
        }

    def execute(self, identifier: str) -> str:
        identifier = identifier.strip()
        if not identifier:
            return "Please specify a MAC, IP, or device name"

        if remove_known_device(identifier):
            return f"Removed '{identifier}' from known devices"
        else:
            return f"Device '{identifier}' not found in known devices"


class AcknowledgeDeviceCommand(Command):
    name = "acknowledge_device"
    description = "Acknowledge an unknown device alert and optionally add it to known devices"

    @property
    def parameters(self) -> dict:
        return {
            "mac": {
                "type": "string",
                "description": "MAC address of the device to acknowledge"
            },
            "name": {
                "type": "string",
                "description": "Optional friendly name to save the device (if blank, just dismisses the alert)"
            }
        }

    def execute(self, mac: str, name: str = "") -> str:
        mac = mac.strip().lower()
        name = name.strip()

        if not mac:
            return "MAC address is required"

        # Find the device in pending alerts to get its IP
        alerts = get_pending_alerts()
        device_ip = None
        for alert in alerts:
            if alert.device.mac.lower() == mac:
                device_ip = alert.device.ip
                break

        if name and device_ip:
            # Add to known devices
            add_known_device(mac, device_ip, name)
            return f"Added '{name}' to known devices"
        else:
            # Just dismiss
            if dismiss_alert(mac):
                return f"Dismissed alert for {mac}"
            else:
                return f"No pending alert found for {mac}"


class DismissAlertCommand(Command):
    name = "dismiss_network_alert"
    description = "Dismiss an unknown device alert without adding to known devices"

    @property
    def parameters(self) -> dict:
        return {
            "mac": {
                "type": "string",
                "description": "MAC address of the device to dismiss"
            }
        }

    def execute(self, mac: str) -> str:
        mac = mac.strip().lower()
        if not mac:
            return "MAC address is required"

        if dismiss_alert(mac):
            return f"Dismissed alert for {mac}"
        else:
            return f"No pending alert found for {mac}"
