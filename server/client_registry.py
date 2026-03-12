"""ClientRegistry — thread-safe dynamic client registration.

Replaces static ALLOWED_CLIENT_IPS. Each client registers at startup with
its client_id, room_id, client_type, and callback_url. The registry is used
for IP allowlisting, timer alert delivery, and hardware command routing.
"""
import logging
import threading
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class RegisteredClient:
    """A registered client."""
    client_id: str
    room_id: str
    client_type: str  # "audio" | "text"
    callback_url: Optional[str]  # e.g. "http://192.168.0.3:8080"
    ip: str  # extracted from callback_url or request IP


class ClientRegistry:
    """Thread-safe registry of active clients."""

    def __init__(self, trusted_ips: set[str] | None = None):
        self._clients: dict[str, RegisteredClient] = {}  # keyed by client_id
        self._lock = threading.Lock()
        self._trusted_ips = trusted_ips or set()

    def register(self, client_id: str, room_id: str, client_type: str,
                 callback_url: Optional[str], ip: str) -> None:
        """Register or re-register a client."""
        client = RegisteredClient(
            client_id=client_id,
            room_id=room_id,
            client_type=client_type,
            callback_url=callback_url,
            ip=ip,
        )
        with self._lock:
            self._clients[client_id] = client
        logger.info(f"Registered client '{client_id}' (room={room_id}, type={client_type}, ip={ip})")

    def unregister(self, client_id: str) -> bool:
        """Unregister a client. Returns True if found."""
        with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
                logger.info(f"Unregistered client '{client_id}'")
                return True
        return False

    def get(self, client_id: str) -> Optional[RegisteredClient]:
        """Get a registered client by ID."""
        with self._lock:
            return self._clients.get(client_id)

    def get_by_room(self, room_id: str, client_type: str | None = None) -> list[RegisteredClient]:
        """Get all clients in a room, optionally filtered by type."""
        with self._lock:
            clients = [c for c in self._clients.values() if c.room_id == room_id]
            if client_type:
                clients = [c for c in clients if c.client_type == client_type]
            return clients

    def all_client_ips(self) -> set[str]:
        """Return all registered client IPs plus trusted IPs."""
        with self._lock:
            ips = {c.ip for c in self._clients.values()}
        return ips | self._trusted_ips

    def is_allowed_ip(self, ip: str) -> bool:
        """Check if an IP is from a registered client or trusted."""
        return ip in self.all_client_ips()

    def get_audio_client_for_room(self, room_id: str) -> Optional[RegisteredClient]:
        """Get the first audio client in a room (for timer delivery, etc.)."""
        clients = self.get_by_room(room_id, client_type="audio")
        return clients[0] if clients else None

    def get_callback_url_for_room(self, room_id: str) -> Optional[str]:
        """Get the callback URL for the audio client in a room."""
        client = self.get_audio_client_for_room(room_id)
        return client.callback_url if client else None

    def list_all(self) -> list[RegisteredClient]:
        """Return all registered clients."""
        with self._lock:
            return list(self._clients.values())
