"""
Checkpoint manager for resume support.

Tracks per-step completion status and progress in a checkpoint.json file.
Uses atomic writes (tmp + rename) to avoid corruption.
"""

import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional


class CheckpointManager:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.ckpt_path = os.path.join(output_dir, "checkpoint.json")
        self._state: Optional[Dict[str, Any]] = None

    def load(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.ckpt_path):
            return None
        try:
            with open(self.ckpt_path, "r") as f:
                self._state = json.load(f)
            return self._state
        except Exception:
            return None

    def save(self, state: Dict[str, Any]) -> None:
        """Atomic write to checkpoint.json."""
        os.makedirs(self.output_dir, exist_ok=True)
        state["updated_at"] = datetime.now().isoformat()
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self.output_dir, suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.ckpt_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise
        self._state = state

    def _ensure_state(self) -> Dict[str, Any]:
        if self._state is None:
            loaded = self.load()
            if loaded is None:
                self._state = {
                    "version": "1.0",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "progress": {
                        "total_samples": 0,
                        "current_step": None,
                        "completed_steps": [],
                        "step_progress": {},
                    },
                }
        return self._state

    def is_step_complete(self, step: str) -> bool:
        state = self._ensure_state()
        return step in state["progress"]["completed_steps"]

    def get_step_output(self, step: str) -> Optional[str]:
        state = self._ensure_state()
        sp = state["progress"]["step_progress"].get(step, {})
        return sp.get("output_file")

    def get_step_completed_count(self, step: str) -> int:
        state = self._ensure_state()
        sp = state["progress"]["step_progress"].get(step, {})
        return sp.get("completed", 0)

    def mark_step_complete(self, step: str, output_file: str, total: int) -> None:
        state = self._ensure_state()
        prog = state["progress"]
        prog["step_progress"].setdefault(step, {})
        prog["step_progress"][step]["completed"] = total
        prog["step_progress"][step]["output_file"] = output_file
        if step not in prog["completed_steps"]:
            prog["completed_steps"].append(step)
        prog["total_samples"] = total
        prog["current_step"] = step
        self.save(state)

    def update_step_progress(self, step: str, completed: int, output_file: Optional[str] = None) -> None:
        state = self._ensure_state()
        prog = state["progress"]
        prog["step_progress"].setdefault(step, {})
        prog["step_progress"][step]["completed"] = completed
        if output_file:
            prog["step_progress"][step]["output_file"] = output_file
        prog["current_step"] = step
        self.save(state)

    def get_step_param(self, step: str, key: str, default=None):
        state = self._ensure_state()
        sp = state["progress"]["step_progress"].get(step, {})
        return sp.get(key, default)

    def set_step_param(self, step: str, key: str, value) -> None:
        state = self._ensure_state()
        state["progress"]["step_progress"].setdefault(step, {})
        state["progress"]["step_progress"][step][key] = value
        self.save(state)

    def init_state(self, config_summary: Dict[str, Any], total_samples: int) -> None:
        state = self._ensure_state()
        state.setdefault("config", config_summary)
        state["progress"]["total_samples"] = total_samples
        self.save(state)
