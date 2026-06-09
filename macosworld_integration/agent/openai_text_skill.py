import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from agent.openai import GPT_SYSTEM_PROMPT
from agent.openai_skill import (
    ACTIVE_PLANNER_MEMO_TTL_STEPS,
    MAX_HISTORY_RECORDS,
    MAX_MAIN_RESPONSE_ROUNDS,
    MAX_PLANNER_ROUNDS,
    MAX_SKILL_CONSULTS_PER_SKILL,
    MM_BRANCH_MODE,
)
from agent.openai_skill_v2 import OpenAISkillAgentV2
from utils.log import print_message
from utils.timeout import timeout


TEXT_BRANCH_MODE = "text_branch"
ARCHITECTURE_VERSION = "openai_text_skill_agent_branch_planner_v1"


OPENAI_TEXT_SKILL_MAIN_SYSTEM_PROMPT = (
    GPT_SYSTEM_PROMPT
    + "\n\n"
    + f"""
Task skills are optional text-only procedural planners.
- The final user message includes each non-exhausted skill's name and short description. Use those descriptions to judge whether a skill is relevant BEFORE calling `LOAD_SKILL(...)`.
- Call `LOAD_SKILL("<exact_skill_name>")` only when the CURRENT screenshot, recent screenshot-action history, and the skill descriptions suggest that extra procedural guidance is likely useful.
- `LOAD_SKILL(...)` opens a temporary text-only planner branch. It loads the corresponding SKILL.md text and returns structured planner guidance. It does NOT execute the action for you.
- `LOAD_SKILL(...)` does not consume an environment interaction step. After the planner summary returns, you must still choose the concrete grounded GUI action in the same step.
- Skill text and planner notes are procedural references only. They are never coordinate templates.
- Trust the CURRENT screenshot over any skill when they conflict.
- Each skill may be consulted at most {MAX_SKILL_CONSULTS_PER_SKILL} times in the same trajectory.
- Skills that reached this limit are removed from the available skill list. If a skill is not listed, do not call it; act from the current screenshot and any existing planner memo instead.

Additional output options:
- In addition to the raw GUI actions defined above, you may return exactly one `LOAD_SKILL("<exact_skill_name>")` call in a fenced plaintext code block.
- Do not return raw GUI actions and `LOAD_SKILL(...)` in the same response.
- Do not load more than one skill in a single response.
- Use only the exact skill names listed in the current user message.
- For clicks, emit `move_to x y` before the click action unless the cursor is already clearly positioned.
- In high-confidence simple UI states, compact 2-4 action blocks are encouraged when every action is grounded in the CURRENT screenshot.
""".strip()
)


OPENAI_TEXT_SKILL_BRANCH_SYSTEM_PROMPT = """
You are inside a temporary planner-only text skill consultation branch for a single Mac desktop step.
Your job is NOT to return a GUI action. Your job is to summarize whether the loaded SKILL.md text is useful for the CURRENT state and what the main agent should optimize for next.

Branch rules:
- Do not return raw GUI actions, WAIT, DONE, FAIL, or LOAD_SKILL.
- Do not request another skill in this branch.
- The main agent will choose the real GUI action after reading your planner summary.
- Use the CURRENT screenshot and recent screenshot-action history to judge whether the skill applies now.
- Use the loaded SKILL.md text as procedural knowledge only, never as a coordinate template.
- If the skill is ineffective for the CURRENT state, say so clearly and avoid forcing the plan toward the skill.
- If the skill is effective, summarize the current subgoal, the planning guidance, the expected next visible state, and whether the task still needs verification.
- Before implying that the task might be complete, consider whether the main agent should still do a verification action before DONE.

Output format:
- Return ONLY one code block.
- The code block must contain exactly one JSON object with these keys:
  - `"skill_applicability"`: one of `"effective"`, `"ineffective"`, `"uncertain"`
  - `"subgoal"`: a short local milestone string
  - `"plan"`: a short behavior plan grounded in the current state
  - `"expected_state"`: a short string describing visible screenshot cues the main agent should aim for next
  - `"completion_scope"`: one of `"local_only"`, `"needs_verification"`, `"maybe_complete"`
- Do not return prose outside the code block.

Correct minimal example:
```json
{
  "skill_applicability": "effective",
  "subgoal": "open the relevant settings surface",
  "plan": "Use the visible sidebar or search field to reach the requested setting, then verify the target control is visible before editing it.",
  "expected_state": "The requested control is visible and ready to edit on the active settings surface",
  "completion_scope": "local_only"
}
```
""".strip()


class OpenAITextSkillAgent(OpenAISkillAgentV2):
    def __init__(
        self,
        model: str,
        remote_client,
        screenshot_rolling_window: int,
        top_p: float,
        temperature: float,
        skill_mode: str = TEXT_BRANCH_MODE,
        skills_library_dir: str = "skills_library",
    ):
        super().__init__(
            model=model,
            remote_client=remote_client,
            screenshot_rolling_window=screenshot_rolling_window,
            top_p=top_p,
            temperature=temperature,
            skill_mode=MM_BRANCH_MODE,
            skills_library_dir=skills_library_dir,
        )
        self.skill_mode = skill_mode
        self.system_prompt = OPENAI_TEXT_SKILL_MAIN_SYSTEM_PROMPT
        self._skill_usage_summary = self._empty_skill_usage_summary()

    def _empty_skill_usage_summary(self) -> Dict[str, object]:
        return {
            "architecture_version": ARCHITECTURE_VERSION,
            "skill_mode": "text_only",
            "text_skill_mode": self.skill_mode,
            "task_skill_names": [],
            "consulted_skill_names": [],
            "loaded_skill_names": [],
            "load_skill_calls": 0,
            "load_skill_successes": 0,
            "skill_branch_invocations": 0,
            "skill_branch_successes": 0,
            "skill_consult_counts": {},
            "exhausted_skill_load_blocks": 0,
            "active_skill_state": None,
        }

    def set_task_skills(self, skill_names: List[str]):
        super().set_task_skills(skill_names)
        self._skill_usage_summary.update(
            {
                "architecture_version": ARCHITECTURE_VERSION,
                "skill_mode": "text_only",
                "text_skill_mode": self.skill_mode,
                "task_skill_names": list(self._task_skill_names),
            }
        )

    def _available_skills_text(self, include_state_previews: bool = False) -> str:
        if not self._task_skill_names:
            return "None"
        lines: List[str] = []
        meta_map = {Path(meta.directory).name: meta for meta in self._task_skill_metadatas}
        for skill_name in self._task_skill_names:
            if self._is_skill_exhausted(skill_name):
                continue
            meta = meta_map.get(skill_name)
            description = ((meta.description or "").strip() if meta is not None else "") or "(no description)"
            consult_count = self._skill_consult_counts.get(skill_name, 0)
            lines.append(f"- {skill_name}: {description} [consulted {consult_count}/{self._consult_limit()}]")
        if lines:
            return "\n".join(lines)
        return "None (all mapped skills are exhausted for this trajectory)"

    def _active_skill_state_text(self) -> str:
        state = self._visible_active_skill_state()
        if not state:
            return "None"
        lines = [
            f"- Skill: {state.get('skill_name', 'Unknown')}",
            f"- Applicability: {state.get('skill_applicability', 'unknown')}",
            f"- Plan: {state.get('plan', 'None')}",
            f"- Expected state: {state.get('expected_state', 'None')}",
            f"- Completion scope: {state.get('completion_scope', 'needs_verification')}",
            f"- Last consulted at outer step: {state.get('last_consult_step', 'unknown')}",
            f"- Consult count: {state.get('consult_count', 0)}/{self._consult_limit()}",
        ]
        if state.get("consult_exhausted"):
            lines.append("- Consult exhausted: true; this skill is no longer loadable, so act from the memo and current screenshot.")
        return "\n".join(lines)

    def _current_step_planner_summaries_text(self) -> str:
        if not self._current_step_planner_summaries:
            return "None"
        chunks: List[str] = []
        for idx, item in enumerate(self._current_step_planner_summaries, start=1):
            lines = [
                f"Planner note {idx}:",
                f"- Skill: {item.get('skill_name', 'Unknown')}",
                f"- Applicability: {item.get('skill_applicability', 'unknown')}",
                f"- Subgoal: {item.get('subgoal', 'None')}",
                f"- Plan: {item.get('plan', 'None')}",
                f"- Expected state: {item.get('expected_state', 'None')}",
                f"- Completion scope: {item.get('completion_scope', 'needs_verification')}",
                f"- Consult count: {item.get('consult_count', 0)}/{self._consult_limit()}",
            ]
            if item.get("consult_exhausted"):
                lines.append("- Consult exhausted: true; do not call this skill again.")
            chunks.append("\n".join(lines))
        return "\n\n".join(chunks)

    def _planner_summary_to_record(self, skill_name: str, summary: Dict[str, str]) -> Dict[str, Any]:
        consult_count = self._skill_consult_counts.get(skill_name, 0)
        return {
            "skill_name": skill_name,
            "skill_applicability": summary["skill_applicability"],
            "subgoal": summary["subgoal"],
            "plan": summary["plan"],
            "expected_state": summary["expected_state"],
            "completion_scope": summary["completion_scope"],
            "consult_count": consult_count,
            "consult_exhausted": consult_count >= self._consult_limit(),
        }

    def _update_active_skill_state(self, planner_note: Dict[str, Any]) -> None:
        applicability = planner_note.get("skill_applicability")
        if applicability == "ineffective":
            active_skill_name = self._active_skill_state.get("skill_name") if self._active_skill_state else None
            if active_skill_name == planner_note.get("skill_name"):
                self._active_skill_state = None
            return
        self._active_skill_state = {
            "skill_name": planner_note.get("skill_name"),
            "skill_applicability": planner_note.get("skill_applicability"),
            "plan": planner_note.get("plan"),
            "expected_state": planner_note.get("expected_state"),
            "completion_scope": planner_note.get("completion_scope"),
            "consult_count": planner_note.get("consult_count", 0),
            "consult_exhausted": bool(planner_note.get("consult_exhausted")),
            "last_consult_step": len(self.responses) + 1,
        }
        self._skill_usage_summary["active_skill_state"] = dict(self._active_skill_state)

    def _build_main_user_content(
        self,
        task: str,
        current_screenshot: Image.Image,
        round_feedback: Optional[List[str]] = None,
    ) -> List[dict]:
        active_memo_text = self._active_skill_state_text()
        active_skill_name = self._active_skill_state.get("skill_name") if self._active_skill_state else None
        current_step_skills = {item.get("skill_name") for item in self._current_step_planner_summaries}
        if active_skill_name and active_skill_name in current_step_skills:
            active_memo_text = "Covered by the planner notes returned in this same outer step."

        elements: List[Any] = [
            "Please decide the next grounded response for the CURRENT screenshot. Return either the next raw GUI action block or `LOAD_SKILL(...)` when text-only procedural guidance is useful.",
            "\nInstruction:\n" + task,
            "\nAvailable non-exhausted text skills for this task (name + description only):\n"
            + self._available_skills_text(include_state_previews=False),
            "\nActive planner memo:\n" + active_memo_text,
            "\nPlanner notes returned in this step:\n" + self._current_step_planner_summaries_text(),
        ]
        if round_feedback:
            feedback_lines = "\n".join(f"- {item}" for item in round_feedback if item)
            if feedback_lines:
                elements.insert(4, "\nFeedback for this step:\n" + feedback_lines)
        repetition_warning = self._build_repetition_warning_text()
        if repetition_warning:
            elements.append("\nLoop warning:\n" + repetition_warning)
        elements.extend(self._build_previous_history_parts())
        elements.extend(
            [
                "\nCurrent screenshot (authoritative for the next decision):",
                current_screenshot,
                "\nRules:\n"
                "- Ground every action in the CURRENT screenshot.\n"
                "- Planner notes are fallible text-only references. They may still be incomplete or partially wrong for the live UI.\n"
                "- Re-decide the real action from the CURRENT screenshot, recent screenshot-action history, and any step feedback before acting.\n"
                "- Treat skills and planner notes as procedural references only, never as coordinate templates.\n"
                f"- Do not reload a skill after {self._consult_limit()} consults; exhausted skills are intentionally absent from the available list.\n"
                "- If no listed skill is clearly useful, act directly from the current screenshot.\n"
                "- If planner notes already exist for this step, use them before consulting again.\n"
                "- If recent actions repeated without progress, change strategy.\n"
                "- Before DONE, verify the full instruction, not just a local subgoal.\n"
                "- To click a target, emit `move_to x y` followed by the click action.\n"
                "- In simple high-confidence states, compact 2-4 action blocks are encouraged.",
            ]
        )
        return self._format_content_elements(elements)

    def _load_skill_for_branch(self, skill_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        self._skill_usage_summary["load_skill_calls"] = int(self._skill_usage_summary.get("load_skill_calls", 0)) + 1
        if not skill_name:
            return None, "Missing skill name in LOAD_SKILL(...)."
        if skill_name not in self._task_skill_names:
            return None, f"Unknown skill '{skill_name}'. Use only a skill from the available skill list."
        if self._is_skill_exhausted(skill_name):
            self._skill_usage_summary["exhausted_skill_load_blocks"] = int(
                self._skill_usage_summary.get("exhausted_skill_load_blocks", 0)
            ) + 1
            return (
                None,
                "That skill has reached its consult limit and is no longer available. "
                "You must continue from the CURRENT screenshot, recent history, and any active planner memo without loading it again.",
            )

        content = self._skill_loader.load_skill_content(skill_name)
        if content is None:
            return None, f"Failed to load SKILL.md text for '{skill_name}'."

        self._skill_consult_counts[skill_name] = self._skill_consult_counts.get(skill_name, 0) + 1
        self._consulted_skills.add(skill_name)
        self._skill_usage_summary["load_skill_successes"] = int(self._skill_usage_summary.get("load_skill_successes", 0)) + 1
        self._skill_usage_summary["consulted_skill_names"] = sorted(self._consulted_skills)
        self._skill_usage_summary["loaded_skill_names"] = sorted(self._consulted_skills)
        self._skill_usage_summary["skill_consult_counts"] = dict(self._skill_consult_counts)
        return {"content": content}, None

    def _build_branch_user_content(
        self,
        task: str,
        current_screenshot: Image.Image,
        trigger_skill_name: str,
        main_trigger_response: str,
        skill_payload: Dict[str, Any],
        round_feedback: Optional[List[str]] = None,
    ) -> List[dict]:
        content = skill_payload["content"]
        elements: List[Any] = [
            "Skill reference package for this temporary text-only planner branch.",
            f"\nRequested skill in the main context: LOAD_SKILL(\"{trigger_skill_name}\")",
            "\nMain-context response that triggered this branch:\n" + self._strip_outer_code_fence(main_trigger_response),
            (
                "\nLoaded SKILL.md reference: "
                f"{content.name} ({trigger_skill_name})\n"
                f"Description: {(content.description or '').strip() or '(no description)'}\n\n"
                "Use the material below as supplemental procedural knowledge only.\n"
                "Do NOT treat it as a coordinate template.\n"
                "The CURRENT screenshot remains authoritative for concrete GUI actions.\n\n"
                f"{content.text}"
            ),
            "\nPlease inspect the CURRENT UI screenshot and return planner JSON only.",
            "\nInstruction:\n" + task,
        ]
        if round_feedback:
            feedback_lines = "\n".join(f"- {item}" for item in round_feedback if item)
            if feedback_lines:
                elements.append("\nAdditional feedback for this branch round:\n" + feedback_lines)
        repetition_warning = self._build_repetition_warning_text()
        if repetition_warning:
            elements.append("\nLoop warning:\n" + repetition_warning)
        elements.extend(self._build_previous_history_parts())
        elements.extend(
            [
                "\nCurrent screenshot (authoritative for planner reasoning):",
                current_screenshot,
                "\nRules:\n"
                "- Return planner JSON only, not an action.\n"
                "- Treat SKILL.md text as procedural guidance, not a coordinate template.\n"
                "- If the current screenshot conflicts with the skill, trust the current screenshot.\n"
                "- `subgoal` should stay local and immediate: the next small milestone under the live UI.\n"
                "- `plan` should explain the next 2 to 4 checks/actions/transitions the main agent should consider.\n"
                "- `expected_state` must describe visible screenshot cues, not an abstract goal.\n"
                "- Use `completion_scope` to indicate whether the task is only locally advanced or still needs verification before DONE.",
            ]
        )
        return self._format_content_elements(elements)

    def _extract_planner_summary(self, response: str) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
        if self._count_code_blocks(response) != 1:
            return None, (
                "The planner response must contain exactly one code block with a JSON object containing "
                "`skill_applicability`, `subgoal`, `plan`, `expected_state`, and `completion_scope`."
            )
        code_body = self._extract_first_code_block_text(response)
        if not code_body:
            return None, "The planner response code block was empty."
        try:
            payload = json.loads(code_body)
        except Exception as exc:
            return None, f"The planner response must be valid JSON. Parse error: {exc}"
        if not isinstance(payload, dict):
            return None, "The planner response JSON must be an object."

        applicability = str(payload.get("skill_applicability", "") or "").strip().lower()
        if applicability not in {"effective", "ineffective", "uncertain"}:
            return None, "The `skill_applicability` field must be one of: effective, ineffective, uncertain."
        subgoal = str(payload.get("subgoal", "") or "").strip()
        plan = str(payload.get("plan", "") or "").strip()
        expected_state = str(payload.get("expected_state", "") or "").strip()
        completion_scope = str(payload.get("completion_scope", "") or "").strip().lower()
        if not subgoal:
            return None, "The `subgoal` field must be a non-empty string."
        if not plan:
            return None, "The `plan` field must be a non-empty string."
        if not expected_state:
            return None, "The `expected_state` field must be a non-empty string."
        if completion_scope not in {"local_only", "needs_verification", "maybe_complete"}:
            return None, "The `completion_scope` field must be one of: local_only, needs_verification, maybe_complete."

        return {
            "skill_applicability": applicability,
            "subgoal": subgoal,
            "plan": plan,
            "expected_state": expected_state,
            "completion_scope": completion_scope,
        }, None

    def _run_skill_branch(
        self,
        task: str,
        current_screenshot: Image.Image,
        trigger_skill_name: str,
        main_trigger_response: str,
        step_idx: int,
    ) -> Dict[str, Any]:
        self._skill_invocation_counter += 1
        branch_id = self._skill_invocation_counter
        branch_rounds: List[dict] = []
        planner_feedback: List[str] = []
        final_response = ""
        final_summary: Optional[Dict[str, str]] = None
        success = False

        skill_payload, load_error = self._load_skill_for_branch(trigger_skill_name)
        if skill_payload is None:
            branch_log = {
                "architecture_version": ARCHITECTURE_VERSION,
                "branch_id": branch_id,
                "step": step_idx,
                "skill_mode": self.skill_mode,
                "trigger_skill_name": trigger_skill_name,
                "main_trigger_response": main_trigger_response,
                "success": False,
                "loaded_skills": [],
                "rounds": [],
                "final_response": "",
                "final_summary": None,
                "final_feedback": load_error or "Failed to load the requested SKILL.md.",
            }
            return {
                "success": False,
                "response": "",
                "summary": None,
                "feedback": load_error or "Failed to load the requested SKILL.md.",
                "log": branch_log,
            }

        for round_idx in range(MAX_PLANNER_ROUNDS):
            user_content = self._build_branch_user_content(
                task=task,
                current_screenshot=current_screenshot,
                trigger_skill_name=trigger_skill_name,
                main_trigger_response=main_trigger_response,
                skill_payload=skill_payload,
                round_feedback=planner_feedback,
            )
            response = self._call_messages(OPENAI_TEXT_SKILL_BRANCH_SYSTEM_PROMPT, user_content)
            final_response = response or ""
            summary, parse_error = self._extract_planner_summary(final_response)
            round_record = {
                "stage": "text_skill_planner",
                "round": round_idx + 1,
                "timestamp": time.time(),
                "system_message": OPENAI_TEXT_SKILL_BRANCH_SYSTEM_PROMPT,
                "contents": self._serialize_content_for_json(user_content),
                "response": final_response,
            }
            if parse_error:
                planner_feedback.append(parse_error)
                round_record["status"] = "invalid_planner_summary"
                round_record["error"] = parse_error
                branch_rounds.append(round_record)
                continue

            success = True
            final_summary = summary
            round_record["status"] = "planner_summary_returned"
            round_record["planner_summary"] = dict(summary)
            branch_rounds.append(round_record)
            break

        branch_log = {
            "architecture_version": ARCHITECTURE_VERSION,
            "branch_id": branch_id,
            "step": step_idx,
            "skill_mode": self.skill_mode,
            "trigger_skill_name": trigger_skill_name,
            "main_trigger_response": main_trigger_response,
            "success": success,
            "loaded_skills": [trigger_skill_name],
            "rounds": branch_rounds,
            "final_response": final_response,
            "final_summary": dict(final_summary) if final_summary else None,
            "final_feedback": planner_feedback[-1] if planner_feedback else None,
        }
        return {
            "success": success,
            "response": final_response,
            "summary": final_summary,
            "feedback": planner_feedback[-1] if planner_feedback else "Text skill planner branch did not produce a valid summary.",
            "log": branch_log,
        }

    def step(
        self,
        task_id: int,
        current_step: int,
        max_steps: int,
        env_language: str,
        task_language: str,
        task: str,
        task_step_timeout: int,
        save_dir: str,
    ):
        self._result_dir = save_dir
        with timeout(task_step_timeout):
            print_message(
                title=f"Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}",
                content="Capturing screenshot...",
            )
            current_screenshot = self.remote_client.capture_screenshot().copy()
            self.screenshots.append(current_screenshot.copy())
            self.screenshots = self.screenshots[-self.screenshot_rolling_window :]

            print_message(
                title=f"Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}",
                content="Calling GUI agent...",
            )
            self._current_step_planner_summaries = []
            round_feedback: List[str] = []
            parsed_actions: List[dict] = []
            final_response = ""

            for round_idx in range(MAX_MAIN_RESPONSE_ROUNDS):
                user_content = self._build_main_user_content(
                    task=task,
                    current_screenshot=current_screenshot,
                    round_feedback=round_feedback,
                )
                main_response = self._call_messages(OPENAI_TEXT_SKILL_MAIN_SYSTEM_PROMPT, user_content)
                final_response = main_response
                self._append_conversation_log(
                    "main_round",
                    {
                        "architecture_version": ARCHITECTURE_VERSION,
                        "step": current_step,
                        "round": round_idx + 1,
                        "system_message": OPENAI_TEXT_SKILL_MAIN_SYSTEM_PROMPT,
                        "response": main_response,
                        "feedback_before_round": list(round_feedback),
                        "contents": self._serialize_content_for_json(user_content),
                    },
                )

                skill_request = self._extract_skill_request(main_response)
                if skill_request:
                    branch_result = self._run_skill_branch(
                        task=task,
                        current_screenshot=current_screenshot,
                        trigger_skill_name=skill_request,
                        main_trigger_response=main_response,
                        step_idx=current_step,
                    )
                    self._skill_invocation_log.append(branch_result["log"])
                    self._append_conversation_log(
                        "skill_branch",
                        {
                            "architecture_version": ARCHITECTURE_VERSION,
                            "step": current_step,
                            "round": round_idx + 1,
                            "skill_request": skill_request,
                            "branch_success": branch_result["success"],
                            "branch_feedback": branch_result["feedback"],
                            "branch_summary": branch_result["summary"],
                        },
                    )
                    if branch_result["success"] and branch_result["summary"]:
                        planner_note = self._planner_summary_to_record(skill_request, branch_result["summary"])
                        self._upsert_current_step_planner_summary(planner_note)
                        self._update_active_skill_state(planner_note)
                        self._skill_usage_summary["skill_branch_invocations"] = len(self._skill_invocation_log)
                        self._skill_usage_summary["skill_branch_successes"] = sum(
                            1 for item in self._skill_invocation_log if item.get("success")
                        )
                        round_feedback.append(
                            f"Text-only planner note loaded from skill '{skill_request}'. Use the planner note plus the CURRENT screenshot to decide the real next action. Do not return LOAD_SKILL unless another non-exhausted skill is still necessary."
                        )
                    else:
                        round_feedback.append(branch_result["feedback"] or f"Text skill branch for '{skill_request}' failed.")
                    continue

                parsed_actions = self.parse_agent_output(main_response)
                if parsed_actions:
                    break
                round_feedback.append(
                    "Invalid action format. Use only supported action names. To click a target, emit `move_to x y` followed by the click action."
                )

            print_message(
                title=f"Task {task_id}/{env_language}/{task_language} Step {current_step}/{max_steps}",
                content="Actuating...",
            )
            status, _ = self.execute_actions(parsed_actions)

        context_dir = os.path.join(save_dir, "context")
        os.makedirs(context_dir, exist_ok=True)
        current_screenshot.save(os.path.join(context_dir, f"step_{str(current_step).zfill(3)}.png"))
        with open(os.path.join(context_dir, f"step_{str(current_step).zfill(3)}_raw_response.txt"), "w") as f:
            f.write(final_response)
        with open(os.path.join(context_dir, f"step_{str(current_step).zfill(3)}_parsed_actions.json"), "w") as f:
            json.dump(parsed_actions, f, indent=4)

        action_summary = (
            " | ".join(json.dumps(action, ensure_ascii=False) for action in parsed_actions)
            if parsed_actions
            else "No valid action"
        )
        self.responses.append(final_response)
        self.actions.append(action_summary)
        self._history_records.append(
            {
                "step": current_step,
                "screenshot": current_screenshot.copy(),
                "response": final_response,
                "action_summary": action_summary,
            }
        )
        self._history_records = self._history_records[-MAX_HISTORY_RECORDS:]
        self._skill_usage_summary["architecture_version"] = ARCHITECTURE_VERSION
        self._skill_usage_summary["skill_mode"] = "text_only"
        self._skill_usage_summary["text_skill_mode"] = self.skill_mode
        self._skill_usage_summary["consulted_skill_names"] = sorted(self._consulted_skills)
        self._skill_usage_summary["loaded_skill_names"] = sorted(self._consulted_skills)
        self._skill_usage_summary["skill_consult_counts"] = dict(self._skill_consult_counts)
        self._skill_usage_summary["exhausted_skill_names"] = sorted(
            skill_name for skill_name in self._task_skill_names if self._is_skill_exhausted(skill_name)
        )
        self._skill_usage_summary["active_skill_state"] = dict(self._active_skill_state) if self._active_skill_state else None
        return status

    def save_conversation_history(self, save_dir: str):
        super().save_conversation_history(save_dir)
        usage_path = os.path.join(save_dir, "skill_usage_summary.json")
        if not os.path.exists(usage_path):
            return
        try:
            with open(usage_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            payload.update(
                {
                    "architecture_version": ARCHITECTURE_VERSION,
                    "skill_mode": "text_only",
                    "text_skill_mode": self.skill_mode,
                    "loaded_skill_names": sorted(self._consulted_skills),
                    "exhausted_skill_names": sorted(
                        skill_name for skill_name in self._task_skill_names if self._is_skill_exhausted(skill_name)
                    ),
                    "active_skill_state_ttl_steps": ACTIVE_PLANNER_MEMO_TTL_STEPS,
                }
            )
            with open(usage_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception:
            return
