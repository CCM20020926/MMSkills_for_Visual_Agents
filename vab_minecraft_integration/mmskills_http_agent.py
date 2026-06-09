"""MMSkills-aware HTTPAgent adapter for VAB-Minecraft.

Copy this file into VisualAgentBench as:
`src/client/agents/mmskills_http_agent.py`.

The adapter keeps VAB's normal HTTPAgent request path and adds a lightweight
text branch for MMSkills packages. It intentionally does not copy VAB or
Minecraft runtime code into this repository.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.client.agents.http_agent import HTTPAgent


LOAD_SKILL_RE = re.compile(r"LOAD_SKILL\(\s*['\"]([^'\"]+)['\"]\s*\)", re.IGNORECASE)


class MMSkillsHTTPAgent(HTTPAgent):
    """HTTPAgent-compatible MMSkills adapter for VAB-Minecraft."""

    def __init__(
        self,
        *args,
        skills_root: Optional[str] = None,
        skill_names: Optional[str | List[str]] = None,
        max_skills: int = 6,
        max_skill_chars: int = 9000,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.skills_root = self._resolve_skills_root(skills_root)
        self.max_skills = max(1, int(max_skills))
        self.max_skill_chars = max(1000, int(max_skill_chars))
        self.skill_name_filter = self._normalize_skill_names(skill_names)
        self.skill_bundles = self._load_skill_bundles()

    @staticmethod
    def _normalize_skill_names(skill_names: Optional[str | List[str]]) -> Optional[set[str]]:
        raw = skill_names or os.getenv("VAB_MMSKILLS_SKILL_NAMES")
        if not raw:
            return None
        if isinstance(raw, str):
            names = [part.strip() for part in raw.split(",")]
        else:
            names = [str(part).strip() for part in raw]
        return {name for name in names if name} or None

    @staticmethod
    def _resolve_skills_root(skills_root: Optional[str]) -> Optional[Path]:
        raw = skills_root or os.getenv("VAB_MMSKILLS_SKILLS_ROOT") or os.getenv("MMSKILLS_SKILLS_ROOT")
        if not raw:
            return None
        path = Path(raw).expanduser()
        return path if path.exists() else None

    def _iter_skill_dirs(self) -> Iterable[Path]:
        if self.skills_root is None:
            return []
        if (self.skills_root / "SKILL.md").exists():
            return [self.skills_root]
        return sorted(path.parent for path in self.skills_root.rglob("SKILL.md"))

    def _load_skill_bundles(self) -> Dict[str, Dict[str, Any]]:
        bundles: Dict[str, Dict[str, Any]] = {}
        for skill_dir in self._iter_skill_dirs():
            runtime_path = skill_dir / "runtime_state_cards.json"
            skill_path = skill_dir / "SKILL.md"
            if not runtime_path.exists() or not skill_path.exists():
                continue
            try:
                runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            except Exception:
                runtime = {}
            domain = str(runtime.get("domain") or skill_dir.parent.name or "").lower()
            if domain and all(token not in domain for token in ("minecraft", "vab")):
                continue
            name = str(runtime.get("skill_name") or skill_dir.name)
            if self.skill_name_filter and name not in self.skill_name_filter and skill_dir.name not in self.skill_name_filter:
                continue
            bundles[name] = {
                "name": name,
                "directory_name": skill_dir.name,
                "title": str(runtime.get("title") or name),
                "purpose": str(runtime.get("purpose") or ""),
                "runtime": runtime,
                "skill_md": skill_path.read_text(encoding="utf-8", errors="replace"),
            }
            if len(bundles) >= self.max_skills:
                break
        return bundles

    def inference(self, history: List[dict]) -> str:
        if not self.skill_bundles:
            return super().inference(history)

        main_history = self._append_to_latest_user(
            history,
            "\n\n" + self._main_skill_hint_text(),
        )
        main_response = super().inference(main_history)
        skill_name = self._extract_load_skill(main_response)
        if not skill_name:
            return main_response

        bundle = self._resolve_requested_skill(skill_name)
        if bundle is None:
            return super().inference(
                self._append_to_latest_user(
                    history,
                    "\n\nThe requested MMSkill was not available. Return the next valid VAB-Minecraft ACTION without LOAD_SKILL.",
                )
            )

        planner_response = super().inference(
            self._append_to_latest_user(history, "\n\n" + self._branch_prompt(bundle))
        )
        planner_text = self._extract_planner_text(planner_response)
        return super().inference(
            self._append_to_latest_user(history, "\n\n" + self._final_action_prompt(bundle["name"], planner_text))
        )

    def _main_skill_hint_text(self) -> str:
        lines = [
            "# Optional MMSkills for VAB-Minecraft",
            "Use these only when the current image and task need extra procedural guidance.",
            "To consult one skill, return `LOAD_SKILL(\"<exact_skill_name>\")` inside the ACTION field instead of a game action.",
            "Do not copy coordinates from skills; the current game image is authoritative.",
        ]
        for bundle in self.skill_bundles.values():
            lines.append(f"- {bundle['name']}: {bundle['purpose'] or bundle['title']}")
            for state_line in self._runtime_state_preview(bundle["runtime"]):
                lines.append(f"  - {state_line}")
        return "\n".join(lines)

    def _branch_prompt(self, bundle: Dict[str, Any]) -> str:
        skill_text = bundle["skill_md"][: self.max_skill_chars]
        runtime_text = "\n".join(self._runtime_state_preview(bundle["runtime"], max_states=8))
        return f"""
# MMSkills Planner Branch
You are not returning the final VAB-Minecraft action yet.
Read the current task/image, the skill text, and the runtime state cues. Return one JSON object with:
- skill_applicability: effective, ineffective, or uncertain
- subgoal: the next local game milestone
- plan: concise guidance for the next 2-4 checks/actions
- expected_state: visible cue to verify

Skill: {bundle['name']}

Runtime cues:
{runtime_text or 'None'}

SKILL.md:
{skill_text}
""".strip()

    @staticmethod
    def _final_action_prompt(skill_name: str, planner_text: str) -> str:
        return f"""
# MMSkills Planner Memo
Consulted skill: {skill_name}
Planner memo:
{planner_text}

Now return the actual next VAB-Minecraft response in the benchmark's required format.
Do not return LOAD_SKILL. Do not return JSON. Use the current image as ground truth.
""".strip()

    @staticmethod
    def _runtime_state_preview(runtime: Dict[str, Any], max_states: int = 4) -> List[str]:
        states = runtime.get("states")
        if not isinstance(states, list):
            states = runtime.get("state_cards")
        if not isinstance(states, list):
            return []
        lines = []
        for state in states[:max_states]:
            if not isinstance(state, dict):
                continue
            state_name = state.get("state_name") or state.get("state_id") or "state"
            when = state.get("when_to_use") or state.get("use_when") or state.get("description") or ""
            line = f"{state_name}: {str(when).strip()}"
            lines.append(line[:240])
        return lines

    @staticmethod
    def _extract_load_skill(response: str) -> Optional[str]:
        match = LOAD_SKILL_RE.search(response or "")
        return match.group(1).strip() if match else None

    def _resolve_requested_skill(self, requested: str) -> Optional[Dict[str, Any]]:
        if requested in self.skill_bundles:
            return self.skill_bundles[requested]
        for bundle in self.skill_bundles.values():
            if requested == bundle["directory_name"]:
                return bundle
        return None

    @staticmethod
    def _extract_planner_text(response: str) -> str:
        match = re.search(r"```(?:json)?\s*(.*?)```", response or "", re.DOTALL | re.IGNORECASE)
        return (match.group(1) if match else response or "").strip()

    @staticmethod
    def _append_to_latest_user(history: List[dict], extra_text: str) -> List[dict]:
        updated = copy.deepcopy(history)
        for item in reversed(updated):
            if item.get("role") != "user":
                continue
            content = item.get("content")
            if isinstance(content, str):
                item["content"] = content + extra_text
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"] = str(part.get("text", "")) + extra_text
                        break
                else:
                    content.insert(0, {"type": "text", "text": extra_text.strip()})
            else:
                item["content"] = str(content or "") + extra_text
            break
        return updated
