"""
ATIF Writer - Creates ATIF-compliant trajectory files.

Based on the Agent Trajectory Interchange Format (ATIF) v1.4 specification.
https://harbor.dev/docs/atif
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any


@dataclass
class ObservationResult:
    """Result from a tool execution."""
    source_call_id: str
    content: str
    is_error: bool = False
    extra: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        result = {
            "source_call_id": self.source_call_id,
            "content": self.content,
        }
        if self.is_error:
            result["is_error"] = self.is_error
        if self.extra:
            result["extra"] = self.extra
        return result


@dataclass
class Observation:
    """Observation containing tool execution results."""
    results: list[ObservationResult] = field(default_factory=list)
    extra: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        result = {
            "results": [r.to_dict() for r in self.results],
        }
        if self.extra:
            result["extra"] = self.extra
        return result


@dataclass
class ToolCall:
    """A tool/function invocation."""
    tool_call_id: str
    function_name: str
    arguments: dict[str, Any]
    extra: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        result = {
            "tool_call_id": self.tool_call_id,
            "function_name": self.function_name,
            "arguments": self.arguments,
        }
        if self.extra:
            result["extra"] = self.extra
        return result


@dataclass
class Metrics:
    """LLM operational metrics for a step."""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    logprobs: Optional[list[float]] = None
    completion_token_ids: Optional[list[int]] = None
    extra: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        result = {}
        if self.prompt_tokens is not None:
            result["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens is not None:
            result["completion_tokens"] = self.completion_tokens
        if self.cached_tokens is not None:
            result["cached_tokens"] = self.cached_tokens
        if self.cost_usd is not None:
            result["cost_usd"] = self.cost_usd
        if self.logprobs is not None:
            result["logprobs"] = self.logprobs
        if self.completion_token_ids is not None:
            result["completion_token_ids"] = self.completion_token_ids
        if self.extra:
            result["extra"] = self.extra
        return result


@dataclass
class FinalMetrics:
    """Aggregate metrics for the entire trajectory."""
    total_prompt_tokens: Optional[int] = None
    total_completion_tokens: Optional[int] = None
    total_cached_tokens: Optional[int] = None
    total_cost_usd: Optional[float] = None
    total_steps: Optional[int] = None
    extra: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        result = {}
        if self.total_prompt_tokens is not None:
            result["total_prompt_tokens"] = self.total_prompt_tokens
        if self.total_completion_tokens is not None:
            result["total_completion_tokens"] = self.total_completion_tokens
        if self.total_cached_tokens is not None:
            result["total_cached_tokens"] = self.total_cached_tokens
        if self.total_cost_usd is not None:
            result["total_cost_usd"] = self.total_cost_usd
        if self.total_steps is not None:
            result["total_steps"] = self.total_steps
        if self.extra:
            result["extra"] = self.extra
        return result


@dataclass
class Step:
    """An individual interaction step in the trajectory."""
    step_id: int
    timestamp: str
    source: str  # "user", "agent", or "system"
    message: Optional[str] = None
    model_name: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    observation: Optional[Observation] = None
    metrics: Optional[Metrics] = None
    extra: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        result = {
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "source": self.source,
        }
        if self.message is not None:
            result["message"] = self.message
        if self.model_name is not None:
            result["model_name"] = self.model_name
        if self.reasoning_content is not None:
            result["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.observation:
            result["observation"] = self.observation.to_dict()
        if self.metrics:
            metrics_dict = self.metrics.to_dict()
            if metrics_dict:  # Only include if non-empty
                result["metrics"] = metrics_dict
        if self.extra:
            result["extra"] = self.extra
        return result


@dataclass
class Agent:
    """Agent configuration."""
    name: str
    version: str
    model_name: Optional[str] = None
    extra: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "version": self.version,
        }
        if self.model_name:
            result["model_name"] = self.model_name
        if self.extra:
            result["extra"] = self.extra
        return result


@dataclass
class Trajectory:
    """Root-level trajectory object."""
    schema_version: str
    session_id: str
    agent: Agent
    steps: list[Step] = field(default_factory=list)
    final_metrics: Optional[FinalMetrics] = None
    extra: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        result = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "agent": self.agent.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
        }
        if self.final_metrics:
            result["final_metrics"] = self.final_metrics.to_dict()
        if self.extra:
            result["extra"] = self.extra
        return result


class ATIFWriter:
    """
    Writer for ATIF-compliant trajectory files.

    Supports both incremental JSONL writing and final JSON export.
    """

    SCHEMA_VERSION = "ATIF-v1.4"

    def __init__(self, output_dir: Path, session_id: str, agent_name: str = "claude-code",
                 agent_version: str = "1.0.0", model_name: Optional[str] = None):
        """
        Initialize the ATIF writer.

        Args:
            output_dir: Base directory for trajectory output
            session_id: Unique session identifier
            agent_name: Name of the agent
            agent_version: Version of the agent
            model_name: Default model name for the agent
        """
        self.output_dir = output_dir
        self.session_id = session_id
        self.session_dir = output_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.agent = Agent(
            name=agent_name,
            version=agent_version,
            model_name=model_name
        )

        self.jsonl_path = self.session_dir / "trajectory.jsonl"
        self.json_path = self.session_dir / "trajectory.json"

        # Initialize header written flag
        self._header_written = False

    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO 8601 format."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def write_header(self) -> None:
        """Write the trajectory header to JSONL (schema and agent info)."""
        if self._header_written:
            return

        header = {
            "_type": "header",
            "schema_version": self.SCHEMA_VERSION,
            "session_id": self.session_id,
            "agent": self.agent.to_dict(),
            "started_at": self._get_timestamp(),
        }

        with open(self.jsonl_path, "w") as f:
            f.write(json.dumps(header) + "\n")
            f.flush()

        self._header_written = True

    def write_step(self, step: Step) -> None:
        """
        Append a step to the trajectory JSONL file.

        Args:
            step: The Step to write
        """
        if not self._header_written:
            self.write_header()

        step_dict = step.to_dict()
        step_dict["_type"] = "step"

        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(step_dict) + "\n")
            f.flush()

    def write_user_step(self, step_id: int, message: str,
                        extra: Optional[dict] = None) -> Step:
        """
        Write a user message step.

        Args:
            step_id: Sequential step ID
            message: User's message
            extra: Optional extra metadata

        Returns:
            The created Step
        """
        step = Step(
            step_id=step_id,
            timestamp=self._get_timestamp(),
            source="user",
            message=message,
            extra=extra
        )
        self.write_step(step)
        return step

    def write_agent_step(
        self,
        step_id: int,
        message: Optional[str] = None,
        reasoning_content: Optional[str] = None,
        tool_calls: Optional[list[ToolCall]] = None,
        observation: Optional[Observation] = None,
        metrics: Optional[Metrics] = None,
        model_name: Optional[str] = None,
        extra: Optional[dict] = None
    ) -> Step:
        """
        Write an agent response step.

        Args:
            step_id: Sequential step ID
            message: Agent's text response
            reasoning_content: Agent's thinking/reasoning
            tool_calls: List of tool calls made
            observation: Observation with tool results
            metrics: Token/cost metrics
            model_name: Model used for this step
            extra: Optional extra metadata

        Returns:
            The created Step
        """
        step = Step(
            step_id=step_id,
            timestamp=self._get_timestamp(),
            source="agent",
            message=message,
            model_name=model_name or self.agent.model_name,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            observation=observation,
            metrics=metrics,
            extra=extra
        )
        self.write_step(step)
        return step

    def write_system_step(self, step_id: int, message: str,
                          extra: Optional[dict] = None) -> Step:
        """
        Write a system message step.

        Args:
            step_id: Sequential step ID
            message: System message
            extra: Optional extra metadata

        Returns:
            The created Step
        """
        step = Step(
            step_id=step_id,
            timestamp=self._get_timestamp(),
            source="system",
            message=message,
            extra=extra
        )
        self.write_step(step)
        return step

    def finalize(self, final_metrics: Optional[FinalMetrics] = None) -> Trajectory:
        """
        Finalize the trajectory by assembling JSONL into JSON.

        Args:
            final_metrics: Aggregate metrics for the trajectory

        Returns:
            The complete Trajectory object
        """
        # Read JSONL and reconstruct trajectory
        steps = []
        header_data = None

        if self.jsonl_path.exists():
            with open(self.jsonl_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "header":
                        header_data = data
                    elif data.get("_type") == "step":
                        # Remove internal _type field
                        data.pop("_type", None)
                        steps.append(self._dict_to_step(data))

        trajectory = Trajectory(
            schema_version=self.SCHEMA_VERSION,
            session_id=self.session_id,
            agent=self.agent,
            steps=steps,
            final_metrics=final_metrics
        )

        # Write final JSON
        with open(self.json_path, "w") as f:
            json.dump(trajectory.to_dict(), f, indent=2)

        return trajectory

    def _dict_to_step(self, data: dict) -> Step:
        """Convert a dictionary to a Step object."""
        tool_calls = None
        if data.get("tool_calls"):
            tool_calls = [
                ToolCall(
                    tool_call_id=tc["tool_call_id"],
                    function_name=tc["function_name"],
                    arguments=tc["arguments"],
                    extra=tc.get("extra")
                )
                for tc in data["tool_calls"]
            ]

        observation = None
        if data.get("observation"):
            obs_data = data["observation"]
            observation = Observation(
                results=[
                    ObservationResult(
                        source_call_id=r["source_call_id"],
                        content=r["content"],
                        is_error=r.get("is_error", False),
                        extra=r.get("extra")
                    )
                    for r in obs_data.get("results", [])
                ],
                extra=obs_data.get("extra")
            )

        metrics = None
        if data.get("metrics"):
            m = data["metrics"]
            metrics = Metrics(
                prompt_tokens=m.get("prompt_tokens"),
                completion_tokens=m.get("completion_tokens"),
                cached_tokens=m.get("cached_tokens"),
                cost_usd=m.get("cost_usd"),
                logprobs=m.get("logprobs"),
                completion_token_ids=m.get("completion_token_ids"),
                extra=m.get("extra")
            )

        return Step(
            step_id=data["step_id"],
            timestamp=data["timestamp"],
            source=data["source"],
            message=data.get("message"),
            model_name=data.get("model_name"),
            reasoning_content=data.get("reasoning_content"),
            tool_calls=tool_calls,
            observation=observation,
            metrics=metrics,
            extra=data.get("extra")
        )


def create_tool_call(tool_call_id: str, function_name: str,
                     arguments: dict[str, Any]) -> ToolCall:
    """Helper to create a ToolCall."""
    return ToolCall(
        tool_call_id=tool_call_id,
        function_name=function_name,
        arguments=arguments
    )


def create_observation(results: list[tuple[str, str, bool]]) -> Observation:
    """
    Helper to create an Observation from a list of results.

    Args:
        results: List of (source_call_id, content, is_error) tuples

    Returns:
        Observation object
    """
    return Observation(
        results=[
            ObservationResult(
                source_call_id=source_id,
                content=content,
                is_error=is_error
            )
            for source_id, content, is_error in results
        ]
    )
