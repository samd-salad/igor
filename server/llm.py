"""LLM integration with Claude API for conversational AI."""
import logging
import re
from typing import Optional, List, Dict, Tuple
import anthropic

from datetime import datetime
from server.config import CLAUDE_API_KEY, CLAUDE_MODEL, MAX_CONVERSATION_HISTORY
from prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLM:
    """Handles LLM interactions with the Claude API."""

    def __init__(self, api_key: str = CLAUDE_API_KEY, model: str = CLAUDE_MODEL, max_history: int = MAX_CONVERSATION_HISTORY):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_history = max_history
        self.conversation_history: List[Dict] = []
        self.last_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        logger.info(f"LLM initialized: Claude API ({model})")

    def _get_system_prompt(self, persistent_memory: str = "", speaker: str = None) -> str:
        """Build system prompt with current context injected."""
        now = datetime.now()
        time_context = f"Current: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"

        if speaker:
            time_context += f" | Speaking: {speaker}"

        prompt = SYSTEM_PROMPT.format(persistent_memory=persistent_memory)
        prompt += f"\n<current_context>\n{time_context}\n</current_context>"
        return prompt

    @staticmethod
    def _strip_await_marker(text: str) -> Tuple[str, bool]:
        """
        Detect and strip [AWAIT] marker from response.

        Returns:
            Tuple of (cleaned_text, await_followup)
        """
        await_pattern = r'\s*\[AWAIT\]\s*$'
        if re.search(await_pattern, text, re.IGNORECASE):
            cleaned = re.sub(await_pattern, '', text, flags=re.IGNORECASE).strip()
            return (cleaned, True)
        return (text, False)

    @staticmethod
    def _serialize_content(content) -> list:
        """Convert Anthropic SDK content block objects to plain dicts for history storage."""
        result = []
        for block in content:
            if block.type == "text":
                result.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                result.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return result

    def _trim_history(self):
        """Trim history and ensure it starts with a plain-text user message.

        A tool_result user message without its preceding tool_use assistant message
        causes a Claude API validation error, so we walk past any such orphaned messages.
        """
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        while self.conversation_history:
            first = self.conversation_history[0]
            if first["role"] == "user" and isinstance(first["content"], str):
                break
            self.conversation_history.pop(0)

    def chat(
        self,
        user_text: str,
        tools: List[Dict],
        tool_executor,
        persistent_memory: str = "",
        speaker: str = None
    ) -> Optional[Tuple[str, bool]]:
        """
        Send message to LLM and get response.

        Args:
            user_text: User's message
            tools: List of available tools in Anthropic format
            tool_executor: Callback function to execute tools (name, **kwargs) -> result
            persistent_memory: Persistent memory to include in system prompt
            speaker: Identified speaker name (if known)

        Returns:
            Tuple of (reply_text, await_followup) or None on failure
        """
        self.conversation_history.append({"role": "user", "content": user_text})
        self._trim_history()

        system_prompt = self._get_system_prompt(persistent_memory, speaker)

        # Reset usage tracking for this interaction
        self.last_usage = {"input_tokens": 0, "output_tokens": 0}

        # First LLM call
        try:
            logger.debug(f"Sending message to LLM: '{user_text}'")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                timeout=30.0,
                system=system_prompt,
                tools=tools,
                messages=self.conversation_history,
            )
            self.last_usage["input_tokens"] += response.usage.input_tokens
            self.last_usage["output_tokens"] += response.usage.output_tokens
        except anthropic.APIStatusError as e:
            logger.error(f"LLM request failed ({e.status_code}): {e.message}")
            return None
        except anthropic.APIConnectionError as e:
            logger.error(f"LLM connection error: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return None

        # Handle tool calls
        if response.stop_reason == "tool_use":
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            logger.info(f"LLM requested {len(tool_use_blocks)} tool call(s)")

            # Store assistant turn with full content (text + tool_use blocks)
            self.conversation_history.append({
                "role": "assistant",
                "content": self._serialize_content(response.content),
            })

            # Execute each tool and collect results
            tool_results = []
            for tool_use in tool_use_blocks:
                tool_name = tool_use.name
                tool_args = tool_use.input
                logger.info(f"Executing tool: {tool_name}({tool_args})")
                try:
                    result = tool_executor(tool_name, **tool_args)
                    logger.info(f"Tool '{tool_name}' returned: {result}")
                except Exception as e:
                    logger.error(f"Tool '{tool_name}' failed: {e}")
                    result = f"Error executing {tool_name}: {e}"

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": str(result),
                })

            # Tool results go back as a user message (Anthropic format)
            self.conversation_history.append({
                "role": "user",
                "content": tool_results,
            })

            # Follow-up LLM call
            try:
                logger.debug("Sending follow-up message to LLM with tool results")
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    timeout=30.0,
                    system=system_prompt,
                    tools=tools,
                    messages=self.conversation_history,
                )
                self.last_usage["input_tokens"] += response.usage.input_tokens
                self.last_usage["output_tokens"] += response.usage.output_tokens
            except Exception as e:
                logger.error(f"LLM follow-up failed: {e}")
                return None

        # Extract text from final response
        reply = "".join(block.text for block in response.content if block.type == "text").strip()

        if not reply:
            logger.warning("LLM returned empty response")
            return None

        reply, await_followup = self._strip_await_marker(reply)

        if await_followup:
            logger.info("LLM is awaiting follow-up response")

        logger.info(f"LLM replied: '{reply}'")

        # Store final assistant turn
        self.conversation_history.append({"role": "assistant", "content": reply})
        self._trim_history()

        return (reply, await_followup)

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
