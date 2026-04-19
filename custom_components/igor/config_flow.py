"""Config flow for the Igor conversation agent integration."""
from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigFlow, ConfigFlowResult
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import CONF_API_KEY, CONF_URL, DEFAULT_URL, DOMAIN, REQUEST_TIMEOUT_SECONDS


class IgorConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Igor."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            url = user_input[CONF_URL].rstrip("/")
            api_key = user_input.get(CONF_API_KEY, "") or ""

            # Probe the server's /api/health (best-effort) so the user gets
            # immediate feedback if the URL is wrong.
            try:
                session = async_get_clientsession(self.hass)
                async with session.get(f"{url}/api/health", timeout=REQUEST_TIMEOUT_SECONDS) as resp:
                    if resp.status >= 400:
                        errors["base"] = "cannot_connect"
            except Exception:  # noqa: BLE001
                errors["base"] = "cannot_connect"

            if not errors:
                await self.async_set_unique_id(url)
                self._abort_if_unique_id_configured()
                return self.async_create_entry(
                    title="Igor",
                    data={CONF_URL: url, CONF_API_KEY: api_key},
                )

        schema = vol.Schema({
            vol.Required(CONF_URL, default=DEFAULT_URL): str,
            vol.Optional(CONF_API_KEY, default=""): str,
        })
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)
