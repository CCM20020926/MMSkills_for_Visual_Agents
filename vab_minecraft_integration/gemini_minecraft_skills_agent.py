"""Gemini VAB-Minecraft agent with internal MMSkills consultation.

Copy this file and `minecraft_skill_loader.py` into the VisualAgentBench agent
package that contains `gemini_agent.py`. The public VAB task still receives only
normal Minecraft `OBSERVATION`, `THOUGHT`, and `ACTION` responses. Internal
`LOAD_SKILL(...)` and `LOAD_STATE_VIEWS(...)` calls are consumed by this host
agent and are never sent to the Minecraft container.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .gemini_agent import GeminiAgent
from .minecraft_skill_loader import (
    MinecraftSkillLoader,
    ResolvedSkillStateSelection,
    SkillMetadata,
)


ARCHITECTURE_VERSION = "vab_minecraft_gemini_multimodal_skill_agent_v2_gated_planner"
LOAD_SKILL_PATTERN = re.compile(r"LOAD_SKILL\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")
LOAD_STATE_VIEWS_PATTERN = re.compile(r"LOAD_STATE_VIEWS\s*\((.*)\)\s*$", re.DOTALL)
MAIN_STATE_HINT_LIMIT = 3
STAGE1_EVIDENCE_GOALS = {
    "inventory_or_hotbar_check",
    "action_target_context",
    "progress_verification",
    "failure_recovery",
}


class GeminiMinecraftSkillsAgent(GeminiAgent):
    """GeminiAgent wrapper that branch-loads VAB-Minecraft MMSkills."""

    def __init__(
        self,
        *args,
        skills_library_dir: Optional[str] = None,
        task_skill_mapping_path: Optional[str] = None,
        skill_mode: str = "multimodal",
        max_skill_chars: int = 10000,
        max_skill_consults_per_skill: int = 2,
        max_skill_consults_per_step: int = 1,
        max_state_view_selection_rounds: int = 2,
        max_stage1_selected_states: int = 2,
        max_stage1_selected_views: int = 4,
        max_planner_rounds: int = 3,
        active_memo_ttl_steps: int = 5,
        save_skill_logs: bool = True,
        min_task_skills: int = 1,
        max_task_skills: int = 5,
        fallback_skill_names: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._init_minecraft_skill_consultation(
            skills_library_dir=(
                skills_library_dir
                or os.getenv("VAB_MMSKILLS_SKILLS_ROOT")
                or os.getenv("MMSKILLS_SKILLS_ROOT")
                or "skills_library/vab_minecraft"
            ),
            task_skill_mapping_path=(
                task_skill_mapping_path
                or os.getenv("VAB_MMSKILLS_TASK_SKILL_MAPPING")
                or "configs/skills/vab_minecraft_task_skill_mapping_top5.json"
            ),
            skill_mode=skill_mode,
            max_skill_chars=max_skill_chars,
            max_skill_consults_per_skill=max_skill_consults_per_skill,
            max_skill_consults_per_step=max_skill_consults_per_step,
            max_state_view_selection_rounds=max_state_view_selection_rounds,
            max_stage1_selected_states=max_stage1_selected_states,
            max_stage1_selected_views=max_stage1_selected_views,
            max_planner_rounds=max_planner_rounds,
            active_memo_ttl_steps=active_memo_ttl_steps,
            save_skill_logs=save_skill_logs,
            min_task_skills=min_task_skills,
            max_task_skills=max_task_skills,
            fallback_skill_names=fallback_skill_names,
        )

    def _init_minecraft_skill_consultation(
        self,
        *,
        skills_library_dir: str,
        task_skill_mapping_path: str,
        skill_mode: str,
        max_skill_chars: int,
        max_skill_consults_per_skill: int,
        max_skill_consults_per_step: int,
        max_state_view_selection_rounds: int,
        max_stage1_selected_states: int,
        max_stage1_selected_views: int,
        max_planner_rounds: int,
        active_memo_ttl_steps: int,
        save_skill_logs: bool,
        min_task_skills: int,
        max_task_skills: int,
        fallback_skill_names: Optional[List[str]],
    ) -> None:
        if skill_mode not in {"text_only", "multimodal"}:
            raise ValueError("skill_mode must be 'text_only' or 'multimodal'")
        self.skill_mode = skill_mode
        self.skills_library_dir = skills_library_dir
        self.task_skill_mapping_path = task_skill_mapping_path
        self.skill_loader = MinecraftSkillLoader(
            skills_library_dir=skills_library_dir,
            max_skill_chars=max_skill_chars,
        )
        self.min_task_skills = min(5, max(1, int(min_task_skills)))
        self.max_task_skills = min(5, max(self.min_task_skills, int(max_task_skills)))
        self.fallback_skill_names = fallback_skill_names or [
            "MINECRAFT_Resolve_Recipes_Tags_And_Dependency_Order"
        ]
        self.max_skill_consults_per_skill = max(1, int(max_skill_consults_per_skill))
        self.max_skill_consults_per_step = max(0, int(max_skill_consults_per_step))
        self.max_state_view_selection_rounds = max(1, int(max_state_view_selection_rounds))
        self.max_stage1_selected_states = max(0, int(max_stage1_selected_states))
        self.max_stage1_selected_views = max(0, int(max_stage1_selected_views))
        self.max_planner_rounds = max(1, int(max_planner_rounds))
        self.active_memo_ttl_steps = max(1, int(active_memo_ttl_steps))
        self.save_skill_logs = save_skill_logs
        self._thread_state = threading.local()
        self._metadata_cache: Optional[List[SkillMetadata]] = None
        self._task_skill_mapping_cache: Optional[Dict[str, Dict[str, Any]]] = None

    def _llm_inference(self, history: List[dict]) -> str:
        return super().inference(history)

    def inference(self, history: List[dict]) -> str:
        state = self._get_or_reset_state(history)
        step_idx = self._current_step_index(history)
        if state.get("current_step") != step_idx:
            state["current_step"] = step_idx
            state["current_step_planner_summaries"] = []
            state["current_step_consults"] = 0

        feedback: List[str] = []
        max_rounds = self.max_skill_consults_per_step + 2
        for round_idx in range(max_rounds):
            force_action = (
                bool(state["current_step_planner_summaries"])
                or state.get("current_step_consults", 0) >= self.max_skill_consults_per_step
                or round_idx > 0
            )
            augmented_history = self._build_main_history(
                history,
                state=state,
                force_action=force_action,
                feedback=feedback,
            )
            response = self._llm_inference(augmented_history)
            skill_name = self._extract_load_skill_request(response)
            if not skill_name:
                self._save_skill_logs(state)
                return response

            if force_action:
                feedback.append("A skill branch has already been handled. Return OBSERVATION, THOUGHT, and ACTION only.")
                continue

            allowed, reason = self._can_consult_skill(state, skill_name)
            if not allowed:
                feedback.append(reason)
                continue

            branch_result = self._run_skill_branch(
                history=history,
                state=state,
                skill_name=skill_name,
                main_trigger_response=response,
                step_idx=step_idx,
            )
            state["current_step_consults"] = state.get("current_step_consults", 0) + 1
            state["skill_invocations"].append(branch_result["log"])
            if branch_result.get("success") and branch_result.get("summary"):
                summary_record = self._planner_summary_to_record(skill_name, branch_result["summary"], state)
                accepted, guidance = self._handle_planner_summary(state, summary_record)
                feedback.append(
                    f"Planner guidance from {skill_name} is available. Return the real Minecraft ACTION now."
                    if accepted
                    else guidance
                )
            else:
                feedback.append(branch_result.get("feedback") or "Skill branch failed. Return a grounded action.")
            self._save_skill_logs(state)

        corrective_history = self._build_main_history(
            history,
            state=state,
            force_action=True,
            feedback=feedback + ["Final attempt: output only OBSERVATION, THOUGHT, and ACTION."],
        )
        final_response = self._llm_inference(corrective_history)
        self._save_skill_logs(state)
        return final_response

    def _get_or_reset_state(self, history: List[dict]) -> Dict[str, Any]:
        key = self._trajectory_key(history)
        state = getattr(self._thread_state, "state", None)
        if not isinstance(state, dict) or state.get("key") != key:
            task_goal = self._extract_task_goal(history) or "unknown_task"
            skill_assignment = self._resolve_task_skill_assignment(task_goal)
            state = {
                "key": key,
                "created_at": time.time(),
                "task_goal": task_goal,
                "assigned_skill_names": list(skill_assignment["skills"]),
                "skill_assignment_source": skill_assignment["source"],
                "skill_assignment_group": skill_assignment.get("group", ""),
                "skill_consult_counts": {},
                "consulted_skills": set(),
                "current_step": 0,
                "current_step_consults": 0,
                "current_step_planner_summaries": [],
                "active_planner_memo": None,
                "skill_invocations": [],
                "case_dir": self._case_dir_from_history(history),
                "last_history_len": len(history),
            }
            self._thread_state.state = state
        else:
            state["last_history_len"] = len(history)
            case_dir = self._case_dir_from_history(history)
            if case_dir:
                state["case_dir"] = case_dir
        return state

    def _trajectory_key(self, history: List[dict]) -> str:
        task_goal = self._extract_task_goal(history) or "unknown_task"
        case_dir = self._case_dir_from_history(history) or "unknown_case"
        return f"{threading.get_ident()}::{case_dir}::{task_goal}"

    @staticmethod
    def _current_step_index(history: List[dict]) -> int:
        return sum(1 for item in history if item.get("role") == "agent") + 1

    def _build_main_history(
        self,
        history: List[dict],
        *,
        state: Dict[str, Any],
        force_action: bool,
        feedback: Optional[List[str]] = None,
    ) -> List[dict]:
        cloned = copy.deepcopy(history)
        if not cloned:
            return cloned

        system_idx = next((idx for idx, item in enumerate(cloned) if item.get("role") == "system"), None)
        addendum = self._main_system_addendum(force_action=force_action)
        if system_idx is not None:
            cloned[system_idx]["content"] = str(cloned[system_idx].get("content", "")) + "\n\n" + addendum
        else:
            cloned.insert(0, {"role": "system", "content": addendum})

        latest_user_idx = self._latest_user_index(cloned)
        if latest_user_idx is not None:
            self._append_text_to_message(
                cloned[latest_user_idx],
                self._main_user_addendum(state, force_action=force_action, feedback=feedback),
            )
        return cloned

    def _main_system_addendum(self, *, force_action: bool) -> str:
        if force_action:
            mode_text = (
                "A skill planner has already been consulted or skill consultation is disabled for this step. "
                "Do NOT return LOAD_SKILL. Return the normal Minecraft response with OBSERVATION, THOUGHT, and ACTION only."
            )
        else:
            mode_text = (
                "You may request one temporary skill planner branch by returning exactly "
                "`LOAD_SKILL(\"<exact_skill_dir_name>\")` when procedural Minecraft guidance would likely help. "
                "If you consult a skill, your whole response must be only that LOAD_SKILL call."
            )
        return f"""
# Multimodal Minecraft Skills
{mode_text}

Skills are optional references, not executable actions.
- Use only the non-exhausted task-specific skills listed in the latest user message.
- Detailed skill text, state cards, and images are loaded only inside the temporary branch.
- Skill images are examples from other trajectories: coordinates, terrain layout, visible item counts, and exact screenshots are not transferable.
- The Minecraft environment can execute only one valid function in ACTION: craft, smelt, equip, teleport_to_spawn, look_up, or execute.
- Never output LOAD_STATE_VIEWS; that is only for internal planner branches.

Output rules:
- For real Minecraft environment interaction, keep the original VAB format: OBSERVATION, THOUGHT, ACTION.
- For skill consultation, output exactly LOAD_SKILL("<exact_skill_dir_name>") as the entire response.
- Never write ACTION: LOAD_SKILL(...). LOAD_SKILL is an internal branch request, not an environment action.
""".strip()

    def _main_user_addendum(
        self,
        state: Dict[str, Any],
        *,
        force_action: bool,
        feedback: Optional[List[str]],
    ) -> str:
        rules = [
            "- Ground the final ACTION in the current screenshot, inventory, equipment, and feedback.",
            "- Prefer `look_up` when the recipe/dependency is unknown.",
            "- Prefer smaller concrete `execute` prompts when the executor stalls.",
        ]
        if force_action:
            rules.append("- Return final OBSERVATION, THOUGHT, and ACTION only. Do not return LOAD_SKILL.")
        else:
            rules.append("- If a listed skill is useful now, return only LOAD_SKILL(\"<exact_skill_dir_name>\").")

        sections = [
            "\n\n# Available Non-Exhausted Multimodal Minecraft Skills For This Task",
            self._skills_with_counts_text(state),
            "\n# Active Skill Planner Memo",
            self._active_memo_text(state),
            "\n# Planner Notes Returned In This Step",
            self._planner_summaries_text(state.get("current_step_planner_summaries") or []),
            "\n# Skill Use Rules For This Turn",
            "\n".join(rules),
        ]
        if feedback:
            sections.extend(["\n# Feedback From Internal Skill Controller", "\n".join(f"- {item}" for item in feedback if item)])
        return "\n".join(sections)

    def _skills_with_counts_text(self, state: Dict[str, Any]) -> str:
        assigned_skill_names = self._assigned_skill_names(state)
        if not assigned_skill_names:
            return "None"
        metadata_by_name = self._skill_metadata_by_name()
        lines: List[str] = []
        for skill_name in assigned_skill_names:
            if self._is_skill_exhausted(state, skill_name):
                continue
            meta = metadata_by_name.get(skill_name)
            description = meta.description if meta else "(metadata unavailable)"
            consult_count = int((state.get("skill_consult_counts") or {}).get(skill_name, 0))
            lines.append(
                f"- {skill_name}: {description or '(no description)'} "
                f"[consulted {consult_count}/{self.max_skill_consults_per_skill}]"
            )
            lines.append(self._minimal_state_hint_preview(skill_name))
        return "\n".join(lines) if lines else "None (all assigned skills are exhausted)"

    def _minimal_state_hint_preview(self, skill_name: str) -> str:
        bundles = self.skill_loader.load_state_bundles(skill_name)
        if not bundles or not bundles.bundles:
            return "  - no runtime state hints"
        lines = []
        for bundle in bundles.bundles[:MAIN_STATE_HINT_LIMIT]:
            state_name = bundle.state_name or bundle.state_id or "(unnamed state)"
            when_to_use = self._compact(bundle.when_to_use, 130)
            lines.append(f"  - {state_name}: when to use: {when_to_use}.")
        remaining = len(bundles.bundles) - MAIN_STATE_HINT_LIMIT
        if remaining > 0:
            lines.append(f"  - ... {remaining} more states")
        return "\n".join(lines)

    def _skill_metadata(self) -> List[SkillMetadata]:
        if self._metadata_cache is None:
            self._metadata_cache = self.skill_loader.discover_skills()
        return list(self._metadata_cache)

    def _skill_metadata_by_name(self) -> Dict[str, SkillMetadata]:
        return {Path(meta.directory).name: meta for meta in self._skill_metadata()}

    def _can_consult_skill(self, state: Dict[str, Any], skill_name: str) -> Tuple[bool, str]:
        available = {Path(meta.directory).name for meta in self._skill_metadata()}
        if skill_name not in available:
            return False, f"Unknown skill '{skill_name}'. Use an exact skill directory name from the available list."
        assigned = set(self._assigned_skill_names(state))
        if skill_name not in assigned:
            allowed = ", ".join(self._assigned_skill_names(state)) or "(none)"
            return False, f"Skill '{skill_name}' is not assigned for this task. Use one of: {allowed}."
        counts = state.setdefault("skill_consult_counts", {})
        if self._is_skill_exhausted(state, skill_name):
            return False, f"Skill '{skill_name}' has reached its consult limit. Return a grounded ACTION."
        if int(state.get("current_step_consults", 0)) >= self.max_skill_consults_per_step:
            return False, "This step has reached its skill consult limit. Return a final ACTION."
        counts[skill_name] = int(counts.get(skill_name, 0)) + 1
        state.setdefault("consulted_skills", set()).add(skill_name)
        return True, ""

    def _is_skill_exhausted(self, state: Dict[str, Any], skill_name: str) -> bool:
        counts = state.get("skill_consult_counts") or {}
        return int(counts.get(skill_name, 0)) >= self.max_skill_consults_per_skill

    def _assigned_skill_names(self, state: Dict[str, Any]) -> List[str]:
        names = state.get("assigned_skill_names")
        return [str(name) for name in names if str(name).strip()] if isinstance(names, list) else []

    def _resolve_task_skill_assignment(self, task_goal: str) -> Dict[str, Any]:
        normalized = self._normalize_task_name(task_goal)
        mapping = self._load_task_skill_mapping()
        record = mapping.get(normalized)
        if record:
            return {
                "skills": self._ensure_task_skill_bounds(record.get("skills") or []),
                "source": record.get("source", "task_skill_mapping"),
                "group": record.get("group", ""),
            }
        return {
            "skills": self._ensure_task_skill_bounds([]),
            "source": "fallback_no_task_mapping",
            "group": "",
        }

    def _load_task_skill_mapping(self) -> Dict[str, Dict[str, Any]]:
        if self._task_skill_mapping_cache is not None:
            return dict(self._task_skill_mapping_cache)
        path = Path(self.task_skill_mapping_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            self._task_skill_mapping_cache = {}
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self._task_skill_mapping_cache = {}
            return {}

        raw_records: List[Dict[str, Any]] = []
        if isinstance(payload, dict) and isinstance(payload.get("assignments"), list):
            raw_records = [item for item in payload["assignments"] if isinstance(item, dict)]
        elif isinstance(payload, list):
            raw_records = [item for item in payload if isinstance(item, dict)]
        elif isinstance(payload, dict):
            for task_name, skills in payload.items():
                if isinstance(skills, list):
                    raw_records.append({"task": task_name, "skills": skills})

        records: Dict[str, Dict[str, Any]] = {}
        for item in raw_records:
            task_name = str(item.get("task") or item.get("normalized_task") or "").strip()
            normalized = str(item.get("normalized_task") or self._normalize_task_name(task_name)).strip()
            if not normalized:
                continue
            skills = item.get("skills") or item.get("assigned_skills") or []
            if item.get("primary_skill"):
                skills = [item["primary_skill"]] + list(skills)
            records[normalized] = {
                "task": task_name,
                "group": str(item.get("group", "") or ""),
                "skills": self._ensure_task_skill_bounds(skills if isinstance(skills, list) else []),
                "source": str(path),
            }
        self._task_skill_mapping_cache = records
        return dict(records)

    def _ensure_task_skill_bounds(self, skill_names: List[Any]) -> List[str]:
        available = {Path(meta.directory).name for meta in self._skill_metadata()}
        selected: List[str] = []
        for raw_name in skill_names:
            name = str(raw_name or "").strip()
            if name and name in available and name not in selected:
                selected.append(name)
            if len(selected) >= self.max_task_skills:
                break
        if len(selected) < self.min_task_skills:
            for fallback in self.fallback_skill_names:
                name = str(fallback or "").strip()
                if name and name in available and name not in selected:
                    selected.append(name)
                if len(selected) >= self.min_task_skills:
                    break
        if len(selected) < self.min_task_skills:
            for name in sorted(available):
                if name not in selected:
                    selected.append(name)
                if len(selected) >= self.min_task_skills:
                    break
        return selected[: self.max_task_skills]

    def _run_skill_branch(
        self,
        *,
        history: List[dict],
        state: Dict[str, Any],
        skill_name: str,
        main_trigger_response: str,
        step_idx: int,
    ) -> Dict[str, Any]:
        branch_id = len(state.get("skill_invocations") or []) + 1
        content = self.skill_loader.load_skill_content(skill_name)
        if content is None:
            return self._failed_branch_log(branch_id, step_idx, skill_name, main_trigger_response, "Failed to load skill text.")

        selected_requests: List[Dict[str, Any]] = []
        selected_selections: List[ResolvedSkillStateSelection] = []
        stage1_rounds: List[Dict[str, Any]] = []
        stage1_decision: Optional[Dict[str, Any]] = None
        feedback: List[str] = []

        if self.skill_mode == "multimodal" and self.max_stage1_selected_views > 0:
            for round_idx in range(self.max_state_view_selection_rounds):
                response = self._llm_inference(
                    self._build_state_view_selection_history(history, skill_name=skill_name, round_feedback=feedback)
                )
                parsed, error, decision = self._extract_load_state_views_request(response)
                record = {"stage": "state_view_selection", "round": round_idx + 1, "response": response}
                if error:
                    record["status"] = "invalid"
                    record["error"] = error
                    stage1_rounds.append(record)
                    feedback.append(error)
                    continue
                stage1_decision = decision
                selected_requests = parsed or []
                selected_selections, missing = self.skill_loader.load_selected_state_views(skill_name, selected_requests)
                record.update(
                    {
                        "status": "ok" if not missing else "missing_views",
                        "stage1_selection_decision": stage1_decision,
                        "requested_state_view_requests": selected_requests,
                        "selected_state_view_ids": self._flatten_selection_view_ids(selected_selections),
                        "selected_state_view_paths": self._flatten_selection_image_paths(selected_selections),
                        "missing_state_views": missing,
                    }
                )
                stage1_rounds.append(record)
                if missing:
                    feedback.append("Some requested state IDs or view types could not be resolved. Use exact bundle values.")
                    selected_selections = []
                    continue
                break

        planner_rounds: List[Dict[str, Any]] = []
        planner_feedback: List[str] = []
        final_response = ""
        final_summary: Optional[Dict[str, str]] = None
        success = False
        for round_idx in range(self.max_planner_rounds):
            response = self._llm_inference(
                self._build_planner_history(
                    history,
                    skill_name=skill_name,
                    selected_selections=selected_selections,
                    stage1_decision=stage1_decision,
                    round_feedback=planner_feedback,
                )
            )
            final_response = response
            summary, error = self._extract_planner_summary(response)
            record = {"stage": "planner", "round": round_idx + 1, "response": response}
            if error:
                record["status"] = "invalid"
                record["error"] = error
                planner_rounds.append(record)
                planner_feedback.append(error)
                continue
            success = True
            final_summary = summary
            record["status"] = "ok"
            record["planner_summary"] = dict(summary)
            planner_rounds.append(record)
            break

        log = {
            "architecture_version": ARCHITECTURE_VERSION,
            "branch_id": branch_id,
            "step": step_idx,
            "skill_mode": self.skill_mode,
            "trigger_skill_name": skill_name,
            "main_trigger_response": main_trigger_response,
            "success": success,
            "selected_state_view_requests": selected_requests,
            "selected_state_view_ids": self._flatten_selection_view_ids(selected_selections),
            "selected_state_view_paths": self._flatten_selection_image_paths(selected_selections),
            "stage1_selection_decision": dict(stage1_decision) if stage1_decision else None,
            "rounds": stage1_rounds + planner_rounds,
            "final_response": final_response,
            "final_summary": dict(final_summary) if final_summary else None,
            "final_feedback": planner_feedback[-1] if planner_feedback else None,
        }
        return {
            "success": success,
            "summary": final_summary,
            "feedback": planner_feedback[-1] if planner_feedback else "Skill planner branch did not produce a valid summary.",
            "log": log,
        }

    def _build_state_view_selection_history(
        self,
        history: List[dict],
        *,
        skill_name: str,
        round_feedback: Optional[List[str]],
    ) -> List[dict]:
        system = f"""
You are inside a temporary VAB-Minecraft state-view selection branch.
Return only one LOAD_STATE_VIEWS({{...}}) call. Do not return ACTION, OBSERVATION, THOUGHT, or LOAD_SKILL.

First decide whether visual reference images are needed at all.
- Current screenshot, inventory, equipment, location, and action feedback are authoritative.
- Skill images are examples from prior trajectories. Never copy coordinates, terrain layout, or exact counts.
- Use exact state_id and view_type values from the runtime state bundles.
- Select at most {self.max_stage1_selected_states} states and {self.max_stage1_selected_views} total views.
- It is valid to request no views when text metadata is enough or no state matches.

Evidence goals: {", ".join(sorted(STAGE1_EVIDENCE_GOALS))}

Correct format:
```python
LOAD_STATE_VIEWS({{
  "visual_reference_needed": false,
  "why_not_text_only": "Inventory and recipe feedback are enough; reference images would not change the next action.",
  "requests": []
}})
```
""".strip()
        user_parts = self._branch_reference_parts(skill_name)
        user_parts.append({"type": "text", "text": self._current_state_text(history, extra_feedback=round_feedback)})
        image_part = self._latest_image_part(history)
        if image_part is not None:
            user_parts.append(image_part)
        return [{"role": "system", "content": system}, {"role": "user", "content": user_parts}]

    def _build_planner_history(
        self,
        history: List[dict],
        *,
        skill_name: str,
        selected_selections: List[ResolvedSkillStateSelection],
        stage1_decision: Optional[Dict[str, Any]],
        round_feedback: Optional[List[str]],
    ) -> List[dict]:
        system = """
You are inside a temporary planner-only VAB-Minecraft skill branch.
Return structured JSON guidance for the current state, not an executable ACTION.

Rules:
- Current screenshot, inventory, equipment, location, and action feedback are authoritative.
- Use skill materials only for transferable procedural knowledge.
- Do not output OBSERVATION, THOUGHT, ACTION, LOAD_SKILL, or LOAD_STATE_VIEWS.
- Do not recommend a concrete ACTION function call.

Return exactly one JSON object in one code block with:
skill_applicability, subgoal, plan, do_not_do, fallback_if_no_progress,
expected_state, completion_scope.
""".strip()
        user_parts = self._branch_reference_parts(skill_name)
        selected_ids = self._flatten_selection_view_ids(selected_selections)
        user_parts.append(
            {
                "type": "text",
                "text": (
                    "Stage-1 visual selection decision: "
                    + (json.dumps(stage1_decision, ensure_ascii=False) if stage1_decision else "None")
                    + "\nSelected reference views: "
                    + (", ".join(selected_ids) if selected_ids else "None")
                    + "\n"
                    + self._selected_view_metadata_text(skill_name, selected_selections)
                ),
            }
        )
        for selection in selected_selections:
            for loaded_view in selection.loaded_views:
                user_parts.append(
                    {
                        "type": "text",
                        "text": f"[Selected View: {skill_name}/{selection.state.state_id}/{loaded_view.view.view_type}] {loaded_view.view.label or loaded_view.view.use_for}",
                    }
                )
                user_parts.append({"type": "image_url", "image_url": {"url": loaded_view.image_path}})
        user_parts.append({"type": "text", "text": self._current_state_text(history, extra_feedback=round_feedback)})
        image_part = self._latest_image_part(history)
        if image_part is not None:
            user_parts.append(image_part)
        return [{"role": "system", "content": system}, {"role": "user", "content": user_parts}]

    def _branch_reference_parts(self, skill_name: str) -> List[dict]:
        content = self.skill_loader.load_skill_content(skill_name)
        skill_text = content.text if content else "(skill text missing)"
        return [
            {
                "type": "text",
                "text": (
                    f"# Branch Skill Reference: {skill_name}\n\n"
                    "Use this skill as supplemental procedural knowledge only.\n\n"
                    f"{skill_text}\n\n"
                    f"{self.skill_loader.format_state_bundles_for_branch(skill_name)}"
                ),
            }
        ]

    def _current_state_text(self, history: List[dict], extra_feedback: Optional[List[str]]) -> str:
        task_goal = self._extract_task_goal(history) or "(unknown)"
        latest_text, _ = self._latest_user_text_and_image(history)
        sections = [
            "# Current VAB-Minecraft State",
            f"Task goal: {task_goal}",
            "Latest environment message:",
            latest_text or "(missing)",
            "Recent agent responses:",
            self._previous_agent_responses_text(history),
        ]
        if extra_feedback:
            sections.extend(["Internal branch feedback:", "\n".join(f"- {item}" for item in extra_feedback if item)])
        return "\n\n".join(sections)

    def _extract_load_skill_request(self, response: str) -> Optional[str]:
        if not response:
            return None
        for target in [self._extract_first_code_block_text(response) or "", response]:
            match = LOAD_SKILL_PATTERN.search(target.strip())
            if match:
                skill_name = match.group(1).strip()
                if skill_name:
                    return skill_name
        return None

    def _extract_load_state_views_request(
        self,
        response: str,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], Optional[Dict[str, Any]]]:
        code = self._extract_first_code_block_text(response) or response.strip()
        match = LOAD_STATE_VIEWS_PATTERN.search(code)
        if not match:
            return None, "Return exactly LOAD_STATE_VIEWS({...}) in one code block.", None
        payload_text = match.group(1).strip()
        try:
            payload = json.loads(payload_text)
        except Exception:
            try:
                payload = ast.literal_eval(payload_text)
            except Exception as exc:
                return None, f"LOAD_STATE_VIEWS payload must be a JSON object: {exc}", None

        if not isinstance(payload, dict):
            return None, "LOAD_STATE_VIEWS payload must be an object with requests.", None
        visual_reference_needed = self._parse_bool(payload.get("visual_reference_needed"))
        requests_raw = payload.get("requests", [])
        why_not_text_only = str(payload.get("why_not_text_only", "") or "").strip()
        if not isinstance(requests_raw, list):
            return None, "`requests` must be a list.", None
        if visual_reference_needed is None:
            visual_reference_needed = bool(requests_raw)
        if visual_reference_needed is False and requests_raw:
            return None, "When visual_reference_needed is false, requests must be empty.", None
        if visual_reference_needed is True and not requests_raw:
            return None, "When visual_reference_needed is true, requests must not be empty.", None
        if not why_not_text_only:
            return None, "why_not_text_only must explain the image-gating decision.", None

        normalized: List[Dict[str, Any]] = []
        total_views = 0
        for item in requests_raw:
            if not isinstance(item, dict):
                return None, "Every request must be an object.", None
            state_id = str(item.get("state_id", "") or "").strip()
            views = [str(view).strip() for view in item.get("views", []) if str(view).strip()] if isinstance(item.get("views"), list) else []
            evidence_goal = str(item.get("evidence_goal", "") or "").strip()
            reason = str(item.get("reason", "") or "").strip()
            if not state_id or not views or evidence_goal not in STAGE1_EVIDENCE_GOALS or not reason:
                return None, "Each request needs state_id, views, evidence_goal, and reason.", None
            total_views += len(views)
            normalized.append(
                {
                    "state_id": state_id,
                    "views": views,
                    "evidence_goal": evidence_goal,
                    "visual_reference_needed": visual_reference_needed,
                    "why_not_text_only": why_not_text_only,
                    "reason": reason,
                }
            )
        if len(normalized) > self.max_stage1_selected_states:
            return None, f"Select at most {self.max_stage1_selected_states} states.", None
        if total_views > self.max_stage1_selected_views:
            return None, f"Select at most {self.max_stage1_selected_views} total views.", None
        decision = {
            "visual_reference_needed": bool(visual_reference_needed),
            "why_not_text_only": why_not_text_only,
            "request_count": len(normalized),
            "total_view_count": total_views,
            "evidence_goals": sorted({item["evidence_goal"] for item in normalized}),
        }
        return normalized, None, decision

    def _extract_planner_summary(self, response: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        code = self._extract_first_code_block_text(response) or response.strip()
        try:
            payload = json.loads(code)
        except Exception as exc:
            return None, f"Planner response must be valid JSON in one code block: {exc}"
        if not isinstance(payload, dict):
            return None, "Planner JSON must be an object."
        applicability = str(payload.get("skill_applicability", "") or "").strip().lower()
        if applicability not in {"effective", "ineffective", "uncertain"}:
            return None, "skill_applicability must be effective, ineffective, or uncertain."
        completion_scope = str(payload.get("completion_scope", "") or "").strip().lower()
        if completion_scope not in {"local_only", "needs_verification", "maybe_complete"}:
            return None, "completion_scope must be local_only, needs_verification, or maybe_complete."
        required = ["subgoal", "plan", "do_not_do", "fallback_if_no_progress", "expected_state"]
        for key in required:
            if not str(payload.get(key, "") or "").strip():
                return None, f"{key} must be a non-empty string."
        return {
            "skill_applicability": applicability,
            "subgoal": str(payload["subgoal"]).strip(),
            "plan": str(payload["plan"]).strip(),
            "do_not_do": str(payload["do_not_do"]).strip(),
            "fallback_if_no_progress": str(payload["fallback_if_no_progress"]).strip(),
            "expected_state": str(payload["expected_state"]).strip(),
            "completion_scope": completion_scope,
        }, None

    def _planner_summary_to_record(self, skill_name: str, summary: Dict[str, str], state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "skill_name": skill_name,
            **summary,
            "consult_count": int((state.get("skill_consult_counts") or {}).get(skill_name, 0)),
            "consult_exhausted": self._is_skill_exhausted(state, skill_name),
            "last_consult_step": int(state.get("current_step", 0)),
        }

    def _handle_planner_summary(self, state: Dict[str, Any], summary_record: Dict[str, Any]) -> Tuple[bool, str]:
        applicability = str(summary_record.get("skill_applicability", "") or "").lower()
        skill_name = str(summary_record.get("skill_name", "") or "")
        if applicability in {"ineffective", "uncertain"}:
            return False, f"Skill planner for {skill_name} was {applicability}. Choose the next ACTION from the current state."
        state["current_step_planner_summaries"].append(summary_record)
        state["active_planner_memo"] = dict(summary_record)
        return True, ""

    def _active_memo_text(self, state: Dict[str, Any]) -> str:
        memo = state.get("active_planner_memo")
        if not memo:
            return "None"
        current_step = int(state.get("current_step", 0))
        last_step = int(memo.get("last_consult_step", 0) or 0)
        if last_step and current_step - last_step > self.active_memo_ttl_steps:
            return "None"
        return "\n".join(
            [
                f"- Skill: {memo.get('skill_name', 'Unknown')}",
                f"- Applicability: {memo.get('skill_applicability', 'unknown')}",
                f"- Subgoal: {memo.get('subgoal', 'None')}",
                f"- Plan: {memo.get('plan', 'None')}",
                f"- Do not do: {memo.get('do_not_do', 'None')}",
                f"- Fallback if no progress: {memo.get('fallback_if_no_progress', 'None')}",
                f"- Expected state: {memo.get('expected_state', 'None')}",
                f"- Completion scope: {memo.get('completion_scope', 'needs_verification')}",
            ]
        )

    @staticmethod
    def _planner_summaries_text(summaries: List[Dict[str, Any]]) -> str:
        if not summaries:
            return "None"
        chunks = []
        for idx, item in enumerate(summaries, start=1):
            chunks.append(
                "\n".join(
                    [
                        f"Planner note {idx}:",
                        f"- Skill: {item.get('skill_name', 'Unknown')}",
                        f"- Subgoal: {item.get('subgoal', 'None')}",
                        f"- Plan: {item.get('plan', 'None')}",
                        f"- Expected state: {item.get('expected_state', 'None')}",
                    ]
                )
            )
        return "\n\n".join(chunks)

    def _selected_view_metadata_text(self, skill_name: str, selections: List[ResolvedSkillStateSelection]) -> str:
        if not selections:
            return "No state-view images were selected."
        lines = []
        for selection in selections:
            lines.extend(
                [
                    f"[Selection {skill_name}/{selection.state.state_id}]",
                    f"stage: {selection.state.stage or '(unknown)'}",
                    f"reason: {selection.reason or '(none provided)'}",
                    f"when_to_use: {selection.state.when_to_use or '(missing)'}",
                    f"verification_cue: {selection.state.verification_cue or '(none listed)'}",
                    f"visual_risk: {selection.state.visual_risk or '(none listed)'}",
                ]
            )
            for loaded_view in selection.loaded_views:
                lines.append(
                    f"- view {loaded_view.view.view_type}: {loaded_view.view.use_for or '(missing)'} at {loaded_view.image_path}"
                )
        return "\n".join(lines)

    @staticmethod
    def _flatten_selection_view_ids(selections: List[ResolvedSkillStateSelection]) -> List[str]:
        return [
            f"{selection.state.state_id}/{loaded_view.view.view_type}"
            for selection in selections
            for loaded_view in selection.loaded_views
        ]

    @staticmethod
    def _flatten_selection_image_paths(selections: List[ResolvedSkillStateSelection]) -> List[str]:
        return [loaded_view.image_path for selection in selections for loaded_view in selection.loaded_views]

    def _save_skill_logs(self, state: Dict[str, Any]) -> None:
        if not self.save_skill_logs:
            return
        case_dir = state.get("case_dir")
        if not case_dir:
            return
        path = Path(case_dir)
        if not path.exists():
            return
        try:
            invocations_payload = {
                "architecture_version": ARCHITECTURE_VERSION,
                "skill_mode": self.skill_mode,
                "skills_library_dir": str(self.skill_loader.skills_dir),
                "task_goal": state.get("task_goal"),
                "assigned_skill_names": self._assigned_skill_names(state),
                "skill_assignment_source": state.get("skill_assignment_source"),
                "invocations": state.get("skill_invocations") or [],
            }
            summary_payload = {
                "architecture_version": ARCHITECTURE_VERSION,
                "task_goal": state.get("task_goal"),
                "assigned_skill_names": self._assigned_skill_names(state),
                "consulted_skill_names": sorted(state.get("consulted_skills") or []),
                "skill_consult_counts": dict(state.get("skill_consult_counts") or {}),
                "skill_branch_invocations": len(state.get("skill_invocations") or []),
                "skill_branch_successes": sum(1 for item in state.get("skill_invocations") or [] if item.get("success")),
                "current_step": state.get("current_step", 0),
                "active_planner_memo": state.get("active_planner_memo"),
            }
            (path / "skill_invocations.json").write_text(json.dumps(invocations_payload, indent=2, ensure_ascii=False), encoding="utf-8")
            (path / "skill_usage_summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            print(f"Warning: failed to save Minecraft skill logs: {exc}", flush=True)

    def _failed_branch_log(self, branch_id: int, step_idx: int, skill_name: str, main_trigger_response: str, feedback: str) -> Dict[str, Any]:
        log = {
            "architecture_version": ARCHITECTURE_VERSION,
            "branch_id": branch_id,
            "step": step_idx,
            "skill_mode": self.skill_mode,
            "trigger_skill_name": skill_name,
            "main_trigger_response": main_trigger_response,
            "success": False,
            "rounds": [],
            "final_response": "",
            "final_summary": None,
            "final_feedback": feedback,
        }
        return {"success": False, "summary": None, "feedback": feedback, "log": log}

    @staticmethod
    def _latest_user_index(history: List[dict]) -> Optional[int]:
        for idx in range(len(history) - 1, -1, -1):
            if history[idx].get("role") == "user":
                return idx
        return None

    @staticmethod
    def _append_text_to_message(message: Dict[str, Any], text: str) -> None:
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = content + text
        elif isinstance(content, list):
            content.append({"type": "text", "text": text})
        else:
            message["content"] = str(content or "") + text

    def _latest_user_text_and_image(self, history: List[dict]) -> Tuple[str, Optional[str]]:
        idx = self._latest_user_index(history)
        if idx is None:
            return "", None
        content = history[idx].get("content")
        texts: List[str] = []
        image_url: Optional[str] = None
        if isinstance(content, str):
            return content, None
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    texts.append(str(item.get("text", "") or ""))
                elif item.get("type") == "image_url":
                    image = item.get("image_url") or {}
                    if isinstance(image, dict):
                        image_url = image.get("url") or image_url
        return "\n".join(texts).strip(), image_url

    def _latest_image_part(self, history: List[dict]) -> Optional[dict]:
        _, image_url = self._latest_user_text_and_image(history)
        return {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}} if image_url else None

    def _case_dir_from_history(self, history: List[dict]) -> Optional[str]:
        _, image_url = self._latest_user_text_and_image(history)
        image_path = self._path_from_image_url(image_url)
        if image_path and Path(image_path).exists():
            return str(Path(image_path).parent)
        return None

    @staticmethod
    def _path_from_image_url(image_url: Optional[str]) -> Optional[str]:
        if not image_url:
            return None
        value = image_url
        if value.startswith("file://"):
            value = value[len("file://") :]
        return value if value.startswith("/") else None

    def _extract_task_goal(self, history: List[dict]) -> Optional[str]:
        for item in history:
            if item.get("role") != "user":
                continue
            text = self._message_text(item)
            match = re.search(r"Your task is to get a\s+(.+?)\s+in your inventory", text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _normalize_task_name(value: str) -> str:
        cleaned = str(value or "").strip().lower().replace("minecraft:", "")
        return re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")

    @staticmethod
    def _message_text(item: Dict[str, Any]) -> str:
        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(part.get("text", "") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return str(content or "")

    @staticmethod
    def _previous_agent_responses_text(history: List[dict], max_items: int = 4) -> str:
        responses = [str(item.get("content", "") or "") for item in history if item.get("role") == "agent"]
        if not responses:
            return "None"
        start = max(0, len(responses) - max_items)
        return "\n\n".join(f"Step {start + idx + 1}:\n{response}" for idx, response in enumerate(responses[start:]))

    @staticmethod
    def _extract_first_code_block_text(response: str) -> Optional[str]:
        if not response:
            return None
        matches = re.findall(r"```(?:\w+\s*)?(.*?)```", response, re.DOTALL)
        return matches[0].strip() if len(matches) == 1 else None

    @staticmethod
    def _parse_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1"}:
                return True
            if normalized in {"false", "no", "0"}:
                return False
        return None

    @staticmethod
    def _compact(text: str, max_chars: int) -> str:
        cleaned = re.sub(r"\s+", " ", str(text or "").strip())
        cleaned = cleaned.lstrip(":,- ").strip().rstrip(" .;") or "no usage guidance was provided"
        return cleaned if len(cleaned) <= max_chars else cleaned[: max(0, max_chars - 3)].rstrip(" ,;:.") + "..."
