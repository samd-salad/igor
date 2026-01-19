"""Network monitoring commands."""
from .base import Command
from network_monitor import (
    get_known_devices,
    get_pending_alerts,
    add_known_device,
    remove_known_device,
    dismiss_alert,
    scan_network,
    is_known_device,
    add_pending_alert,
)


class ListKnownDevicesCommand(Command):
    name = "list_known_devices"
    description = "List all known/whitelisted devices on the network"

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self) -> str:
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

    def execute(self) -> str:
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

    def execute(self) -> str:
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
