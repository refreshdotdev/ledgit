#!/usr/bin/env python3
"""
SessionStart Hook - Initialize ATIF trajectory when a Claude Code session starts.

This hook creates the session directory with proper naming:
  {timestamp}_{project-name}_{session-id}/

And initializes:
  - trajectory.jsonl (incremental events)
  - metadata.json (session metadata)
  - state.json (internal state)
  - index.json (global session index)
"""

import json
import os
import sys
from pathlib import Path

# Add lib directory to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

from atif_writer import ATIFWriter
from state_manager import StateManager, get_trajectories_dir


def main():
    """Handle SessionStart event."""
    try:
        # Read hook input from stdin
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    session_id = input_data.get("session_id", "unknown")
    model_name = input_data.get("model", None)
    source = input_data.get("source", "startup")  # startup, resume, clear, compact
    cwd = input_data.get("cwd", os.getcwd())

    # Get trajectories directory
    trajectories_dir = get_trajectories_dir()

    # Initialize state manager with project path
    state_manager = StateManager(
        trajectories_dir=trajectories_dir,
        session_id=session_id,
        project_path=cwd
    )

    # Initialize session (creates folder, metadata, index entry)
    metadata = state_manager.initialize_session(model_name=model_name)

    # Store extra session info
    state_manager.set_extra("source", source)
    state_manager.set_extra("cwd", cwd)

    # Copy transcript path for reference
    transcript_path = input_data.get("transcript_path")
    if transcript_path:
        state_manager.set_extra("transcript_path", transcript_path)

    # Initialize ATIF writer and write header
    writer = ATIFWriter(
        output_dir=trajectories_dir,
        session_id=session_id,
        agent_name="claude-code",
        agent_version="1.0.0",
        model_name=model_name
    )
    # Point writer to the correct session directory
    writer.session_dir = state_manager.session_dir
    writer.jsonl_path = state_manager.session_dir / "trajectory.jsonl"
    writer.json_path = state_manager.session_dir / "trajectory.json"
    writer.write_header()

    # Output info (shown in verbose mode)
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": f"ATIF trajectory: {metadata.folder_name}"
        }
    }
    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
