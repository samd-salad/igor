"""LLM integration with Ollama for conversational AI."""
import logging
import re
from typing import Optional, List, Dict
import requests

from server.config import OLLAMA_URL, OLLAMA_MODEL, MAX_CONVERSATION_HISTORY
from prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLM:
    """Handles LLM interactions with Ollama."""

    def __init__(self, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL, max_history: int = MAX_CONVERSATION_HISTORY):
        self.url = url
        self.model = model
        self.max_history = max_history
        self.conversation_history: List[Dict] = []
        logger.info(f"LLM initialized: {url} ({model})")

    def _get_system_prompt(self, persistent_memory: str = "") -> str:
        """Build system prompt with current persistent memory."""
        return SYSTEM_PROMPT.format(persistent_memory=persistent_memory)

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """Convert Anthropic tool format to Ollama/OpenAI format."""
        return [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"]
            }
        } for t in tools]

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove Qwen3 thinking tags from response."""
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

    def chat(
        self,
        user_text: str,
        tools: List[Dict],
        tool_executor,
        persistent_memory: str = ""
    ) -> Optional[str]:
        """
        Send message to LLM and get response.

        Args:
            user_text: User's message
            tools: List of available tools in Anthropic format
            tool_executor: Callback function to execute tools (name, **kwargs) -> result
            persistent_memory: Persistent memory to include in system prompt

        Returns:
            LLM's reply or None on failure
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_text})

        # Trim history if needed
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        # Build messages with system prompt
        messages = [
            {"role": "system", "content": self._get_system_prompt(persistent_memory)}
        ] + self.conversation_history

        # Convert tools to Ollama format
        ollama_tools = self._convert_tools(tools)

        # First LLM call
        try:
            # Truncate user text for logging to avoid leaking sensitive data
            log_text = user_text[:50] + "..." if len(user_text) > 50 else user_text
            logger.debug(f"Sending message to LLM: '{log_text}'")
            response = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": ollama_tools,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            message = response.json().get("message", {})
        except requests.Timeout:
            logger.error("LLM request timed out")
            return None
        except requests.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return None

        # Handle tool calls
        if message.get("tool_calls"):
            logger.info(f"LLM requested {len(message['tool_calls'])} tool call(s)")
            tool_results = []

            for tool_call in message["tool_calls"]:
                func = tool_call["function"]
                tool_name = func["name"]
                tool_args = func["arguments"]

                logger.info(f"Executing tool: {tool_name}({tool_args})")
                try:
                    result = tool_executor(tool_name, **tool_args)
                    logger.info(f"Tool '{tool_name}' returned: {result}")
                except Exception as e:
                    logger.error(f"Tool '{tool_name}' failed: {e}")
                    result = f"Error executing {tool_name}: {e}"

                tool_results.append({"role": "tool", "content": result})

            # Add tool call message and results to history
            self.conversation_history.append(message)
            self.conversation_history.extend(tool_results)

            # Rebuild messages with tool results
            messages = [
                {"role": "system", "content": self._get_system_prompt(persistent_memory)}
            ] + self.conversation_history

            # Follow-up LLM call
            try:
                logger.debug("Sending follow-up message to LLM with tool results")
                response = requests.post(
                    f"{self.url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "tools": ollama_tools,
                        "stream": False
                    },
                    timeout=60
                )
                response.raise_for_status()
                message = response.json().get("message", {})
            except Exception as e:
                logger.error(f"LLM follow-up failed: {e}")
                return None

        # Extract and clean response
        reply = self._strip_thinking(message.get("content", ""))

        if not reply:
            logger.warning("LLM returned empty response")
            return None

        # Truncate reply for logging
        log_reply = reply[:100] + "..." if len(reply) > 100 else reply
        logger.info(f"LLM replied: '{log_reply}'")

        # Add assistant message to history
        self.conversation_history.append({"role": "assistant", "content": reply})

        # Trim history again
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        return reply

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict]:
        """Get current conversation history."""
        return self.conversation_history.copy()

    def set_history(self, history: List[Dict]):
        """Set conversation history (for restoring from saved state)."""
        self.conversation_history = history
        logger.info(f"Conversation history restored ({len(history)} messages)")
