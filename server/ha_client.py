"""Home Assistant REST client for Igor's HA-backed device commands.

All device actions (lights, media players, TVs, todos, notifications) call
through this module. HA's WebSocket API is more powerful but the REST API
is sufficient for our needs and avoids holding a long-lived connection
inside the Igor container.

Configuration:
  HA_URL    — base URL, default http://10.0.40.5:8123
  HA_TOKEN  — long-lived access token (required)

Caching:
  Entity registry and area mappings are cached for ENTITY_CACHE_TTL seconds
  (default 300). Call invalidate_cache() to force a refresh on the next read.

Thread safety:
  The session and cache are protected by a lock. Safe to call from multiple
  command threads concurrently.
"""
import logging
import os
import threading
import time
from collections import defaultdict
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


ENTITY_CACHE_TTL = 300  # seconds; entities and area mappings refresh after this
REQUEST_TIMEOUT = 10  # seconds


class HAError(Exception):
    """Raised when a Home Assistant call fails."""


class HAClient:
    """Thin REST client for Home Assistant.

    Reads HA_URL and HA_TOKEN from the environment at construction time.
    Use the module-level `get_client()` for a process-wide singleton.
    """

    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None):
        self.base_url = (base_url or os.environ.get("HA_URL", "http://10.0.40.5:8123")).rstrip("/")
        self.token = token or os.environ.get("HA_TOKEN", "")
        if not self.token:
            logger.warning("HA_TOKEN not set — Home Assistant calls will fail")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        })
        self._lock = threading.Lock()
        self._cache_time: float = 0.0
        self._states_cache: list[dict] = []
        self._entities_by_area: dict[str, list[str]] = {}
        self._entity_id_to_area: dict[str, str] = {}
        self._areas: list[str] = []

    # -- raw HTTP --

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        try:
            r = self._session.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            raise HAError(f"HA GET {path} failed: {e}") from e
        except requests.RequestException as e:
            raise HAError(f"HA GET {path} unreachable: {e}") from e

    def _post(self, path: str, data: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            r = self._session.post(url, json=data or {}, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            # Some POST endpoints return text; tolerate that
            try:
                return r.json()
            except ValueError:
                return r.text
        except requests.HTTPError as e:
            raise HAError(f"HA POST {path} failed: {e}") from e
        except requests.RequestException as e:
            raise HAError(f"HA POST {path} unreachable: {e}") from e

    # -- service calls --

    def call_service(
        self,
        domain: str,
        service: str,
        data: dict | None = None,
        target: dict | None = None,
        return_response: bool = False,
    ) -> list[dict] | dict:
        """Call a HA service.

        Default returns the list of state changes HA reports. When
        return_response=True, returns the dict {"changed_states": [...],
        "service_response": {...}} — required for services like todo.get_items
        that produce a response payload.

        Examples:
            ha.call_service("light", "turn_on", {"entity_id": "light.office_lamp"})
            ha.call_service("todo", "get_items", {"entity_id": "todo.shopping_list"},
                            return_response=True)
        """
        payload: dict = {}
        if data:
            payload.update(data)
        if target:
            payload["target"] = target
        path = f"/api/services/{domain}/{service}"
        if return_response:
            path += "?return_response"
        return self._post(path, payload)

    # -- state reads --

    def get_states(self, force_refresh: bool = False) -> list[dict]:
        """Return all entity states, cached for ENTITY_CACHE_TTL seconds."""
        with self._lock:
            now = time.monotonic()
            if not force_refresh and self._states_cache and (now - self._cache_time) < ENTITY_CACHE_TTL:
                return list(self._states_cache)
            states = self._get("/api/states")
            if not isinstance(states, list):
                raise HAError("Unexpected /api/states response shape")
            self._states_cache = states
            self._cache_time = now
            self._rebuild_area_index()
            return list(states)

    def get_state(self, entity_id: str) -> dict:
        """Return a single entity's state. Always live (no cache) for freshness."""
        return self._get(f"/api/states/{entity_id}")

    def states_in_domain(self, domain: str, force_refresh: bool = False) -> list[dict]:
        return [s for s in self.get_states(force_refresh) if s["entity_id"].startswith(f"{domain}.")]

    # -- area / template helpers --

    def render_template(self, template: str) -> str:
        """Render a Jinja template against HA state. Returns the rendered string."""
        result = self._post("/api/template", {"template": template})
        return result if isinstance(result, str) else str(result)

    def _rebuild_area_index(self) -> None:
        """Populate _entities_by_area and _entity_id_to_area using the templating API.

        HA exposes `area_id(entity_id)` and `area_name(entity_id)` in templates.
        Build a single template that emits "entity_id|area_name" lines for all entities,
        then parse. One round trip instead of N.
        """
        try:
            template = (
                "{% for s in states %}"
                "{{ s.entity_id }}|{{ area_name(s.entity_id) or '' }}\n"
                "{% endfor %}"
            )
            rendered = self.render_template(template)
        except HAError as e:
            logger.warning(f"Area index rebuild failed: {e}")
            return

        by_area: dict[str, list[str]] = defaultdict(list)
        by_entity: dict[str, str] = {}
        for line in rendered.splitlines():
            if "|" not in line:
                continue
            entity_id, area = line.split("|", 1)
            entity_id = entity_id.strip()
            area = area.strip()
            if not entity_id:
                continue
            if area:
                by_area[area].append(entity_id)
                by_entity[entity_id] = area
        self._entities_by_area = dict(by_area)
        self._entity_id_to_area = by_entity
        self._areas = sorted(by_area.keys())

    def get_areas(self) -> list[str]:
        """Return all area names known to HA."""
        self.get_states()  # ensures index is fresh
        return list(self._areas)

    def entities_in_area(self, area_name: str, domain: Optional[str] = None) -> list[str]:
        """Return entity_ids in an area, optionally filtered by domain.

        Area name match is case-insensitive — HA UI displays areas with whatever
        casing the user picked, but we want to be forgiving on inbound names.
        """
        self.get_states()  # ensures index is fresh
        target = area_name.strip().lower()
        for area, entities in self._entities_by_area.items():
            if area.lower() == target:
                if domain:
                    prefix = f"{domain}."
                    return [e for e in entities if e.startswith(prefix)]
                return list(entities)
        return []

    def area_of(self, entity_id: str) -> str:
        """Return the area name for an entity, or empty string if none."""
        self.get_states()
        return self._entity_id_to_area.get(entity_id, "")

    def area_of_device(self, device_id: str) -> str:
        """Return the area name for a HA device_id, or empty string.

        Used by the conversation agent to map satellite device_id (sent by
        HA when a user speaks via a voice satellite) to the room context.
        """
        if not device_id:
            return ""
        try:
            rendered = self.render_template(f"{{{{ area_name('{device_id}') or '' }}}}")
            return (rendered or "").strip()
        except HAError as e:
            logger.warning(f"area_of_device({device_id}) failed: {e}")
            return ""

    def invalidate_cache(self) -> None:
        with self._lock:
            self._cache_time = 0.0


# -- module singleton --

_client: Optional[HAClient] = None
_client_lock = threading.Lock()


def get_client() -> HAClient:
    """Return the process-wide HAClient, building it on first call."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = HAClient()
    return _client
