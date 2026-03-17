"""Wake word sample labeling command.

Lets the user relabel a false-positive wake word detection ("that wasn't me",
"bad wake word") so the auto-saved audio moves from positive/ to negative/,
improving future retraining accuracy.
"""
import logging

from server.commands.base import Command

logger = logging.getLogger(__name__)


class LabelWakewordCommand(Command):
    name = "label_wakeword"
    description = (
        "Relabel the most recent wake word detection as a false positive. "
        "Call when the user says something like 'that wasn't me', 'bad wake word', "
        "'false alarm', or 'I didn't say that'. This moves the auto-saved audio "
        "from the positive training folder to the negative folder."
    )

    @property
    def parameters(self) -> dict:
        return {}

    def execute(self, _ctx=None, **_) -> str:
        # Find the right Pi callback client for this room
        pi_client = None
        if _ctx and hasattr(self, '_registry'):
            client_entry = self._registry.get(_ctx.client_id)
            if client_entry and client_entry.callback_url:
                from server.pi_callback import PiCallbackClient
                pi_client = PiCallbackClient(client_entry.callback_url)

        # Fall back to legacy pi_client (single-client setups)
        if pi_client is None and hasattr(self, 'pi_client'):
            pi_client = self.pi_client

        if pi_client is None:
            return "No Pi client available to relabel the sample."

        result = pi_client.relabel_wakeword_sample()
        if result is None:
            return "Could not reach the Pi to relabel the sample."
        return result
