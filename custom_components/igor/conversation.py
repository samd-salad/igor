"""Igor conversation entity. Bridges HA's voice pipeline to Igor's
/conversation/process endpoint.

HA contract reference (HA 2026.x):
  - ConversationEntity from homeassistant.components.conversation.entity
  - ConversationInput fields: text, context, conversation_id, device_id,
    satellite_id, language, agent_id, extra_system_prompt
  - ConversationResult fields: response (IntentResponse), conversation_id,
    continue_conversation (bool — note: NOT end_conversation. Inverted
    semantics from Igor's API; we flip it in async_process).
"""
from __future__ import annotations

import logging

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_API_KEY, CONF_URL, DOMAIN, REQUEST_TIMEOUT_SECONDS

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the Igor conversation entity from a config entry."""
    async_add_entities([IgorConversationEntity(entry)])


class IgorConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    """Forwards HA conversation requests to the Igor brain server."""

    _attr_has_entity_name = True
    _attr_name = "Igor"

    def __init__(self, entry: ConfigEntry) -> None:
        self._entry = entry
        self._attr_unique_id = entry.entry_id

    @property
    def supported_languages(self) -> list[str] | str:
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self._entry, self)

    async def async_will_remove_from_hass(self) -> None:
        conversation.async_unset_agent(self.hass, self._entry)
        await super().async_will_remove_from_hass()

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        url = self._entry.data[CONF_URL].rstrip("/")
        api_key = self._entry.data.get(CONF_API_KEY) or ""

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-Igor-Token"] = api_key

        payload = {
            "text": user_input.text,
            "conversation_id": user_input.conversation_id,
            "device_id": user_input.device_id,
            "language": user_input.language,
        }

        speech: str
        end_conversation = True
        conv_id: str | None = user_input.conversation_id

        try:
            session = async_get_clientsession(self.hass)
            async with session.post(
                f"{url}/conversation/process",
                json=payload,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as resp:
                if resp.status != 200:
                    _LOGGER.warning(
                        "Igor returned %s: %s", resp.status, await resp.text()
                    )
                    speech = "Sorry, Igor is unreachable right now."
                else:
                    data = await resp.json()
                    speech = data.get("response") or "(no response)"
                    end_conversation = bool(data.get("end_conversation", True))
                    conv_id = data.get("conversation_id") or conv_id
        except Exception as err:  # noqa: BLE001
            _LOGGER.warning("Igor request failed: %s", err)
            speech = "Sorry, Igor is unreachable right now."

        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(speech)
        return conversation.ConversationResult(
            response=response,
            conversation_id=conv_id,
            continue_conversation=not end_conversation,
        )
