"""
Transcript Parser - Parse Claude Code transcript files.

Claude Code stores transcripts as JSONL files following the Anthropic API
message format. This parser extracts structured data for ATIF conversion.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any


@dataclass
class TextBlock:
    """A text content block from the transcript."""
    text: str


@dataclass
class ThinkingBlock:
    """A thinking/reasoning content block from the transcript."""
    thinking: str


@dataclass
class ToolUseBlock:
    """A tool use content block from the transcript."""
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultBlock:
    """A tool result content block from the transcript."""
    tool_use_id: str
    content: str
    is_error: bool = False


@dataclass
class ParsedMessage:
    """A parsed message from the transcript."""
    role: str  # "user" or "assistant"
    text_blocks: list[TextBlock] = field(default_factory=list)
    thinking_blocks: list[ThinkingBlock] = field(default_factory=list)
    tool_use_blocks: list[ToolUseBlock] = field(default_factory=list)
    tool_result_blocks: list[ToolResultBlock] = field(default_factory=list)
    raw_content: list[dict] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Get combined text from all text blocks."""
        return "\n".join(block.text for block in self.text_blocks)

    @property
    def thinking(self) -> str:
        """Get combined thinking from all thinking blocks."""
        return "\n".join(block.thinking for block in self.thinking_blocks)

    @property
    def has_tool_calls(self) -> bool:
        """Check if this message contains tool calls."""
        return len(self.tool_use_blocks) > 0

    @property
    def has_tool_results(self) -> bool:
        """Check if this message contains tool results."""
        return len(self.tool_result_blocks) > 0


@dataclass
class TranscriptTurn:
    """
    A turn in the conversation, potentially containing multiple messages.

    A turn represents a complete exchange: user message → agent response(s).
    """
    user_message: Optional[ParsedMessage] = None
    agent_messages: list[ParsedMessage] = field(default_factory=list)
    tool_results: list[ToolResultBlock] = field(default_factory=list)


class TranscriptParser:
    """
    Parser for Claude Code transcript files.

    Handles the JSONL format used by Claude Code and extracts structured
    data for ATIF conversion.
    """

    def __init__(self, transcript_path: str | Path):
        """
        Initialize the parser with a transcript file path.

        Args:
            transcript_path: Path to the Claude Code transcript JSONL file
        """
        self.transcript_path = Path(transcript_path)
        self._messages: list[ParsedMessage] = []
        self._parsed = False

    def parse(self) -> list[ParsedMessage]:
        """
        Parse the transcript file and return structured messages.

        Returns:
            List of ParsedMessage objects
        """
        if self._parsed:
            return self._messages

        if not self.transcript_path.exists():
            return []

        self._messages = []

        with open(self.transcript_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    message = self._parse_message(data)
                    if message:
                        self._messages.append(message)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue

        self._parsed = True
        return self._messages

    def _parse_message(self, data: dict) -> Optional[ParsedMessage]:
        """
        Parse a single message from the transcript.

        Args:
            data: Raw message dictionary from transcript

        Returns:
            ParsedMessage or None if not a valid message
        """
        # Handle different transcript formats
        # Format 1: Direct message with role and content
        if "role" in data and "content" in data:
            return self._parse_anthropic_message(data)

        # Format 2: Wrapped message (e.g., {"message": {...}})
        if "message" in data:
            return self._parse_anthropic_message(data["message"])

        return None

    def _parse_anthropic_message(self, data: dict) -> Optional[ParsedMessage]:
        """
        Parse an Anthropic API format message.

        Args:
            data: Message dictionary with role and content

        Returns:
            ParsedMessage or None
        """
        role = data.get("role")
        content = data.get("content", [])

        if role not in ("user", "assistant"):
            return None

        # Handle string content (simple messages)
        if isinstance(content, str):
            return ParsedMessage(
                role=role,
                text_blocks=[TextBlock(text=content)],
                raw_content=[{"type": "text", "text": content}]
            )

        # Handle array content (structured messages)
        if not isinstance(content, list):
            return None

        message = ParsedMessage(role=role, raw_content=content)

        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")

            if block_type == "text":
                message.text_blocks.append(
                    TextBlock(text=block.get("text", ""))
                )
            elif block_type == "thinking":
                message.thinking_blocks.append(
                    ThinkingBlock(thinking=block.get("thinking", ""))
                )
            elif block_type == "tool_use":
                message.tool_use_blocks.append(
                    ToolUseBlock(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        input=block.get("input", {})
                    )
                )
            elif block_type == "tool_result":
                # Tool results can have string or structured content
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    # Extract text from content blocks
                    result_content = "\n".join(
                        item.get("text", str(item))
                        for item in result_content
                        if isinstance(item, dict)
                    )
                elif not isinstance(result_content, str):
                    result_content = str(result_content)

                message.tool_result_blocks.append(
                    ToolResultBlock(
                        tool_use_id=block.get("tool_use_id", ""),
                        content=result_content,
                        is_error=block.get("is_error", False)
                    )
                )

        return message

    def get_last_assistant_message(self) -> Optional[ParsedMessage]:
        """
        Get the last assistant message from the transcript.

        Returns:
            Last ParsedMessage with role="assistant" or None
        """
        messages = self.parse()
        for message in reversed(messages):
            if message.role == "assistant":
                return message
        return None

    def get_last_user_message(self) -> Optional[ParsedMessage]:
        """
        Get the last user message from the transcript.

        Returns:
            Last ParsedMessage with role="user" or None
        """
        messages = self.parse()
        for message in reversed(messages):
            if message.role == "user":
                return message
        return None

    def get_tool_calls_since(self, after_index: int = 0) -> list[ToolUseBlock]:
        """
        Get all tool calls after a given message index.

        Args:
            after_index: Start looking after this message index

        Returns:
            List of ToolUseBlock objects
        """
        messages = self.parse()
        tool_calls = []

        for message in messages[after_index:]:
            if message.role == "assistant":
                tool_calls.extend(message.tool_use_blocks)

        return tool_calls

    def get_tool_results_for(self, tool_call_ids: list[str]) -> dict[str, ToolResultBlock]:
        """
        Get tool results for specific tool call IDs.

        Args:
            tool_call_ids: List of tool call IDs to find results for

        Returns:
            Dictionary mapping tool_call_id to ToolResultBlock
        """
        messages = self.parse()
        results = {}

        for message in messages:
            for result in message.tool_result_blocks:
                if result.tool_use_id in tool_call_ids:
                    results[result.tool_use_id] = result

        return results

    def get_thinking_content(self) -> list[str]:
        """
        Get all thinking content from the transcript.

        Returns:
            List of thinking text strings
        """
        messages = self.parse()
        thinking = []

        for message in messages:
            if message.role == "assistant":
                for block in message.thinking_blocks:
                    thinking.append(block.thinking)

        return thinking

    def get_assistant_turns_with_tools(self) -> list[tuple[ParsedMessage, list[ToolResultBlock]]]:
        """
        Get assistant messages paired with their tool results.

        Returns:
            List of (assistant_message, tool_results) tuples
        """
        messages = self.parse()
        turns = []

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.role == "assistant" and msg.has_tool_calls:
                # Find the corresponding tool results
                tool_results = []
                tool_call_ids = {tc.id for tc in msg.tool_use_blocks}

                # Look for tool results in subsequent user messages
                j = i + 1
                while j < len(messages):
                    next_msg = messages[j]
                    if next_msg.role == "user" and next_msg.has_tool_results:
                        for result in next_msg.tool_result_blocks:
                            if result.tool_use_id in tool_call_ids:
                                tool_results.append(result)
                        break
                    elif next_msg.role == "assistant":
                        # Hit another assistant message without finding results
                        break
                    j += 1

                turns.append((msg, tool_results))

            i += 1

        return turns

    def count_messages(self) -> dict[str, int]:
        """
        Count messages by role.

        Returns:
            Dictionary with counts for 'user', 'assistant', 'total'
        """
        messages = self.parse()
        counts = {"user": 0, "assistant": 0, "total": len(messages)}

        for message in messages:
            if message.role in counts:
                counts[message.role] += 1

        return counts

    def to_conversation_text(self) -> str:
        """
        Convert transcript to a readable conversation format.

        Returns:
            Human-readable conversation string
        """
        messages = self.parse()
        lines = []

        for message in messages:
            role_label = "User" if message.role == "user" else "Assistant"

            if message.thinking:
                lines.append(f"[{role_label} Thinking]: {message.thinking[:200]}...")

            if message.text:
                lines.append(f"{role_label}: {message.text}")

            for tc in message.tool_use_blocks:
                lines.append(f"[Tool Call: {tc.name}] {json.dumps(tc.input)[:100]}...")

            for tr in message.tool_result_blocks:
                status = "Error" if tr.is_error else "Result"
                lines.append(f"[Tool {status}]: {tr.content[:100]}...")

        return "\n".join(lines)
