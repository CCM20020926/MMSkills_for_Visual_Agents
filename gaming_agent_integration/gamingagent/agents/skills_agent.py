import json
import os
import re
import datetime
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from gamingagent.agents.base_agent import BaseAgent
from gamingagent.modules import Observation
from tools.utils import scale_image_up


ARCHITECTURE_VERSION = "gamingagent_skills_agent_v1_branch_load_stage1_views_stage2_planner"
ALLOWED_VIEWS = {"full_frame", "before", "after"}
EVIDENCE_GOALS = {
    "inspect_action_example",
    "recognize_before",
    "verify_after",
    "compare_transition",
}


@dataclass
class SkillBundle:
    name: str
    domain: str
    title: str
    purpose: str
    path: Path
    skill_md: str
    runtime_cards: dict[str, Any]
    state_cards: list[dict[str, Any]]


class SkillsAgent(BaseAgent):
    """BaseAgent-compatible multimodal-skill agent for lmgame 2D games.

    The main context receives only compact text skill hints. When the model
    emits LOAD_SKILL("<name>"), this agent opens a temporary branch:
    Stage 1 selects whether/which skill images to load, and Stage 2 returns a
    planner JSON memo. The main action call then uses that memo to emit the
    actual game move.
    """

    def __init__(
        self,
        *args,
        skills_root: Optional[str] = None,
        skill_names: Optional[str | list[str]] = None,
        skill_cooldown_steps: int = 5,
        max_stage1_selected_states: int = 2,
        max_stage1_selected_views: int = 4,
        **kwargs,
    ):
        self.skills_root_arg = skills_root
        self.skills_root: Optional[Path] = None
        self.skill_name_filter = self._normalize_skill_names(skill_names)
        self.skill_cooldown_steps = max(0, int(skill_cooldown_steps))
        self.max_stage1_selected_states = max(1, int(max_stage1_selected_states))
        self.max_stage1_selected_views = max(1, int(max_stage1_selected_views))
        self._skill_bundles: dict[str, SkillBundle] = {}
        self._skill_consult_counts: dict[str, int] = {}
        self._skill_usage_summary: dict[str, Any] = {
            "architecture_version": ARCHITECTURE_VERSION,
            "load_skill_calls": 0,
            "successful_branch_plans": 0,
            "failed_branch_plans": 0,
            "stage1_visual_reference_needed_counts": {},
            "stage1_evidence_goal_counts": {},
            "cooldown_load_blocks": 0,
        }
        self._active_planner_notes: list[dict[str, Any]] = []
        self._skills_agent_step_index = 0
        self._last_skill_load_step: Optional[int] = None
        self.skill_logs_dir: Optional[Path] = None
        super().__init__(*args, **kwargs)
        self.skill_logs_dir = Path(self.cache_dir) / "skills_agent"
        self.skill_logs_dir.mkdir(parents=True, exist_ok=True)
        self.skills_root = self._resolve_skills_root()
        self._skill_bundles = self._load_skill_bundles()
        self._skill_consult_counts = {name: 0 for name in self._skill_bundles}
        self._save_agent_config()
        self._write_skill_usage_summary()

    @staticmethod
    def _normalize_skill_names(skill_names: Optional[str | list[str]]) -> Optional[set[str]]:
        if not skill_names:
            return None
        if isinstance(skill_names, str):
            names = [part.strip() for part in skill_names.split(",")]
        else:
            names = [str(part).strip() for part in skill_names]
        normalized = {name for name in names if name}
        return normalized or None

    def _resolve_skills_root(self) -> Optional[Path]:
        explicit = self.skills_root_arg or os.getenv("LMGAME_SKILLS_ROOT")
        if explicit:
            path = Path(explicit).expanduser()
            return path if path.exists() else None

        root = Path("runs/lmgame_multimodal_skills")
        candidates = [path for path in root.glob("*/phase4/skills") if path.exists()]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _load_skill_bundles(self) -> dict[str, SkillBundle]:
        if not self.skills_root:
            print("[SkillsAgent] No skills root found; running without skills.")
            return {}

        bundles: dict[str, SkillBundle] = {}
        for skill_dir in sorted(self.skills_root.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md_path = skill_dir / "SKILL.md"
            runtime_path = skill_dir / "runtime_state_cards.json"
            if not skill_md_path.exists() or not runtime_path.exists():
                continue
            try:
                runtime_cards = json.loads(runtime_path.read_text())
            except Exception as exc:
                print(f"[SkillsAgent] Skipping {skill_dir}: cannot read runtime_state_cards.json: {exc}")
                continue

            name = str(runtime_cards.get("skill_name") or skill_dir.name)
            if self.skill_name_filter and name not in self.skill_name_filter and skill_dir.name not in self.skill_name_filter:
                continue
            domain = str(runtime_cards.get("domain") or name.split("__", 1)[0])
            if domain != self.game_name:
                continue

            skill_md = skill_md_path.read_text(errors="replace")
            bundle = SkillBundle(
                name=name,
                domain=domain,
                title=str(runtime_cards.get("title") or name),
                purpose=str(runtime_cards.get("purpose") or ""),
                path=skill_dir,
                skill_md=skill_md,
                runtime_cards=runtime_cards,
                state_cards=list(runtime_cards.get("state_cards") or []),
            )
            bundles[name] = bundle

        print(f"[SkillsAgent] Loaded {len(bundles)} skill(s) from {self.skills_root}.")
        return bundles

    def _save_agent_config(self):
        super()._save_agent_config()
        config_file = Path(self.cache_dir) / "agent_config.json"
        try:
            data = json.loads(config_file.read_text())
            data["agent_type"] = "skills"
            data["skills_agent"] = {
                "architecture_version": ARCHITECTURE_VERSION,
                "skills_root": str(self.skills_root or self.skills_root_arg or ""),
                "loaded_skills": sorted(getattr(self, "_skill_bundles", {}).keys()),
                "skill_cooldown_steps": getattr(self, "skill_cooldown_steps", None),
                "max_stage1_selected_states": getattr(self, "max_stage1_selected_states", None),
                "max_stage1_selected_views": getattr(self, "max_stage1_selected_views", None),
            }
            config_file.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[SkillsAgent] Warning: could not update agent_config.json with skills fields: {exc}")

    def get_action(self, observation):
        self._skills_agent_step_index += 1
        if self.harness:
            print("[SkillsAgent] Harness mode requested; delegating to BaseAgent harness path.")
            return super().get_action(observation)

        processed_observation = self._coerce_observation(observation)
        if not self._skill_bundles:
            print("[SkillsAgent] No loaded skills for this game; using BaseModule directly.")
            return self._base_plan(processed_observation, self.custom_prompt), processed_observation

        main_prompt = self._build_main_custom_prompt(self.custom_prompt)
        main_plan = self._base_plan(
            processed_observation,
            main_prompt,
            system_prompt_suffix=self._main_skill_system_prompt_suffix(),
        )
        raw_main = str(main_plan.get("raw_response_str") or "")
        trigger_skill_name = self._extract_load_skill(raw_main)

        if not trigger_skill_name:
            main_plan["skills_agent"] = {
                "architecture_version": ARCHITECTURE_VERSION,
                "used_skill": False,
            }
            return main_plan, processed_observation

        cooldown_remaining = self._skill_cooldown_remaining()
        if cooldown_remaining > 0:
            self._skill_usage_summary["cooldown_load_blocks"] += 1
            branch_record = {
                "architecture_version": ARCHITECTURE_VERSION,
                "step_index": self._skills_agent_step_index,
                "requested_skill_name": trigger_skill_name,
                "main_trigger_response": raw_main,
                "error": (
                    f"Skill loading is on cooldown for {cooldown_remaining} more step(s); "
                    "continue from the current screenshot and active planner memo."
                ),
            }
            self._write_branch_record(branch_record)
        else:
            branch_record = self._run_skill_branch(trigger_skill_name, processed_observation, raw_main)
        planner_note = branch_record.get("planner_json")
        if planner_note:
            self._update_active_planner_notes(planner_note)
            final_prompt = self._build_action_with_planner_prompt(self.custom_prompt, planner_note)
        else:
            final_prompt = self._build_action_after_failed_branch_prompt(
                self.custom_prompt,
                trigger_skill_name,
                str(branch_record.get("error") or "The skill branch did not return a usable planner JSON."),
            )

        final_plan = self._base_plan(processed_observation, final_prompt)
        final_retry_count = 0
        while self._extract_load_skill(str(final_plan.get("raw_response_str") or final_plan.get("action") or "")):
            final_retry_count += 1
            final_prompt = "\n\n".join(
                [
                    final_prompt,
                    "The previous response incorrectly requested another skill branch. "
                    "A skill branch has already been handled for this decision step. "
                    "The environment has not advanced; answer again with the actual next game action only.",
                    "Do not output LOAD_SKILL or load_skill. Do not output JSON.",
                    "Required final format:",
                    "thought: [brief reason based on the current screenshot and planner guidance]",
                    "move: (action_name, frame_count)",
                ]
            )
            print(
                "[SkillsAgent] Final action retry "
                f"{final_retry_count} after repeated LOAD_SKILL; environment has not advanced."
            )
            sleep_seconds = float(os.getenv("SKILLS_FINAL_ACTION_RETRY_SLEEP_SECONDS", "2"))
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            final_plan = self._base_plan(processed_observation, final_prompt)
        final_plan["skills_agent"] = {
            "architecture_version": ARCHITECTURE_VERSION,
            "used_skill": bool(planner_note),
            "trigger_skill_name": trigger_skill_name,
            "branch_log_path": branch_record.get("branch_log_path"),
            "planner_json": planner_note,
            "main_trigger_response": raw_main,
        }
        return final_plan, processed_observation

    def _coerce_observation(self, observation) -> Observation:
        if isinstance(observation, Observation):
            return observation

        img_path_for_observation = None
        symbolic_representation_for_observation = None
        if isinstance(observation, str) and os.path.exists(observation):
            img_path_for_observation = observation
        elif isinstance(observation, np.ndarray):
            saved_img_path = self.save_obs(observation)
            if saved_img_path:
                img_path_for_observation = saved_img_path
            elif self.observation_mode in {"vision", "both"}:
                raise ValueError("Failed to process visual observation: cannot save image.")
        else:
            symbolic_representation_for_observation = str(observation)

        if self.observation_mode == "vision":
            if not img_path_for_observation:
                raise ValueError("Vision mode requires a valid image path or image data.")
            return Observation(img_path=img_path_for_observation, max_memory=self.max_memory)
        if self.observation_mode == "text":
            return Observation(textual_representation=symbolic_representation_for_observation, max_memory=self.max_memory)
        if self.observation_mode == "both":
            return Observation(
                img_path=img_path_for_observation,
                textual_representation=symbolic_representation_for_observation,
                max_memory=self.max_memory,
            )
        raise ValueError(f"Unsupported observation_mode: {self.observation_mode}")

    def _base_plan(
        self,
        observation: Observation,
        custom_prompt: Optional[str],
        system_prompt_suffix: Optional[str] = None,
    ) -> dict[str, Any]:
        base_module = self.modules["base_module"]
        if not system_prompt_suffix:
            return base_module.plan_action(observation=observation, custom_prompt=custom_prompt)

        original_system_prompt = base_module.system_prompt or ""
        suffix = system_prompt_suffix.strip()
        if suffix:
            base_module.system_prompt = (
                original_system_prompt + "\n\n" + suffix
                if original_system_prompt
                else suffix
            )
        try:
            return base_module.plan_action(observation=observation, custom_prompt=custom_prompt)
        finally:
            base_module.system_prompt = original_system_prompt

    def _main_skill_system_prompt_suffix(self) -> str:
        if self.game_name == "super_mario_bros":
            return (
                "You are operating inside a multimodal-skill-capable Super Mario Bros agent. "
                "The main decision call must always return visible final text in exactly this two-line structure:\n"
                "thought: [brief reasoning about the current screenshot and why the chosen move is appropriate]\n"
                "move: [one of two allowed move types]\n\n"
                "Allowed move type 1 is an environment interaction: "
                "`move: (action_name, frame_count)`, where `action_name` is one of the legal Super Mario Bros actions "
                "from the user prompt and `frame_count` is a legal integer frame count. "
                "Allowed move type 2 is a skill-branch request: "
                "`move: LOAD_SKILL(\"<exact_skill_name>\")`. "
                "Use the skill-branch request when the current visual state is uncertain, strategically risky, "
                "requires jump/run timing near enemies, pipes, pits, separated platforms, or airborne hazards, "
                "or when recent movement/no-op patterns suggest weak x-progress. "
                "The state previews in the main prompt are only triggers for deciding whether to consult a skill; "
                "do not treat them as a direct action recipe. If a state preview is guiding your move and cooldown allows, "
                "load the matching skill branch first so the planner can return candidate action constraints. "
                "A skill branch provides advisory planning help and examples only; it does not execute the action, "
                "does not replace the current screenshot, and is not a decisive authority. "
                "After any internal reasoning, always output visible final text. "
                "Never stop with an empty answer, a hidden-thought-only answer, whitespace, JSON, or prose outside the two-line format."
            )
        return (
            "You are operating inside a multimodal-skill-capable game agent. "
            "For the main decision call, you are expected to use a relevant skill branch whenever "
            "the current visual state is uncertain, ambiguous, strategically risky, or has several plausible legal moves. "
            "In the multimodal-skills variant, branch consultation is part of the intended control flow, "
            "so use it whenever a loaded skill plausibly applies rather than treating it as a last resort. "
            "Request one relevant skill branch by outputting exactly "
            "`LOAD_SKILL(\"<exact_skill_name>\")`. "
            "The state previews in the main prompt are only triggers for deciding whether to consult a skill; "
            "do not use them as direct action recipes when cooldown allows a branch. "
            "Prefer consulting skills before irreversible Sokoban pushes, before choosing a Candy Crush swap on a busy board, "
            "before Tetris rotation/placement/drop commitments, after invalid/None/no_op actions, after repeated movement, "
            "before Super Mario Bros jump/run commitments near enemies, pipes, pits, platforms, or airborne hazards, "
            "or when recent reward/perf progress is weak. "
            "A skill branch provides advisory planning help and examples only; it is not decisive, "
            "does not execute the action, and must never override the current screenshot, legal action rules, "
            "or your own final judgment. "
            "If the base game prompt requires a `move:` line, request a skill as "
            "`move: LOAD_SKILL(\"<exact_skill_name>\")` so the response still has visible final text. "
            "Always produce visible final text after any internal reasoning; never stop with an empty answer, "
            "a hidden-thought-only answer, or whitespace."
        )

    def _build_main_custom_prompt(self, custom_prompt: Optional[str]) -> str:
        sections: list[str] = []
        if custom_prompt:
            sections.append(custom_prompt)
        cooldown_remaining = self._skill_cooldown_remaining()
        if cooldown_remaining > 0:
            if self.game_name == "super_mario_bros":
                skill_policy_lines = [
                    "Multimodal skill policy for this Super Mario Bros step:",
                    f"- Skill loading is on cooldown for {cooldown_remaining} more environment step(s). Do not output LOAD_SKILL this step.",
                    "- Continue from the current screenshot and any active planner memo.",
                    "- Output `thought:` followed by `move: (action_name, frame_count)` using a legal Mario action.",
                    "- Skill examples are strategy references, never coordinate templates.",
                ]
            else:
                skill_policy_lines = [
                    "Multimodal skill policy for this step:",
                    f"- Skill loading is on cooldown for {cooldown_remaining} more environment step(s). Do not output LOAD_SKILL this step.",
                    "- Continue from the current screenshot and any active planner memo.",
                    "- Skill examples are strategy references, never coordinate templates.",
                ]
            available_skills_text = "Skill loading temporarily unavailable because of the 5-step cooldown."
        else:
            if self.game_name == "super_mario_bros":
                skill_policy_lines = [
                    "Multimodal skill policy for this Super Mario Bros step:",
                    "- You may either output an environment move or request one temporary skill branch.",
                    "- Environment move format: `move: (action_name, frame_count)`.",
                    '- Skill request format: `move: LOAD_SKILL("<exact_skill_name>")`.',
                    "- Use a skill branch when the current screenshot plausibly matches a listed skill/state preview and the next action needs platforming judgment.",
                    "- Do not use the listed state preview text as a direct action recipe; if it affects your move choice and cooldown allows, load the exact skill first.",
                    "- Strongly prefer requesting a relevant skill before jump/run commitments near enemies, pipes, pits, separated platforms, broken floor, Lakitu/fireballs/plants, or crowded enemy lanes.",
                    "- Also request a skill after repeated no-op/plain-right/similar jumps, after weak x-progress, or after a life-loss restart where the previous failure point should guide the next attempt.",
                    "- Directly output an environment move only when the next Mario action is visually obvious, legal, low-risk, and does not require comparing skill patterns.",
                    "- A skill branch returns planning guidance only; it does not execute the move and is not a decisive authority.",
                    "- The current screenshot is authoritative. Skill examples are strategy references, never coordinate templates, and can be ignored if they do not fit.",
                    "- After a branch returns, explicitly consider its `subgoal`, `plan`, `control_intent`, `risk_profile`, `action_constraints`, `state_checks`, `do_not_do`, `fallback_if_no_progress`, and `expected_state` before choosing the next environment move.",
                    "- Treat `action_constraints.candidate_actions` as a constrained action-name set, not as a final move; infer frame count from current visual distance and the planner `frame_count_band`.",
                    "- Your visible final answer must contain a `thought:` line and either `move: (action_name, frame_count)` or `move: LOAD_SKILL(\"<exact_skill_name>\")`; never return only hidden thinking or an empty response.",
                    f"- After a skill branch is loaded, no other skill can be loaded for the next {self.skill_cooldown_steps} environment step(s).",
                ]
            else:
                skill_policy_lines = [
                    "Multimodal skill policy for this step:",
                    "- You may either return the next game action in the required base format, or request one temporary skill branch.",
                    '- To request a branch, put `LOAD_SKILL("<exact_skill_name>")` in the action/move field, for example `move: LOAD_SKILL("<exact_skill_name>")`.',
                    "- In this multimodal-skills variant, branch consultation is expected whenever any loaded skill plausibly applies; do not reserve skills only for extreme uncertainty.",
                    "- Do not use listed state previews as direct action recipes; if a preview affects your move choice and cooldown allows, load the exact skill first.",
                    "- Strongly prefer requesting a relevant skill branch when the board is non-trivial, risky, ambiguous, or has multiple plausible legal actions.",
                    "- Request a skill before irreversible pushes/drops/commitments, after invalid/None/no_op outputs, after repeated movement or oscillation, and when reward/perf progress has stalled.",
                    "- Only request a skill when the current screenshot plausibly matches its hints and the branch can help with planning.",
                    "- Directly act only when the next action is visually obvious, legal, low-risk, and does not require strategic comparison.",
                    "- A skill branch returns planning guidance only; it does not execute the move and is not a decisive authority.",
                    "- The current screenshot is authoritative. Skill examples are strategy references, never coordinate templates, and can be ignored if they do not fit.",
                    "- After a branch returns, explicitly consider its `subgoal`, `plan`, `control_intent`, `risk_profile`, `action_constraints`, `state_checks`, `do_not_do`, `fallback_if_no_progress`, and `expected_state` before choosing the next move.",
                    "- Treat `action_constraints.candidate_actions` as a constrained action-name set, not as a final action or sequence.",
                    "- Your visible final answer must contain a `thought:` line and either a legal `move:` line or `move: LOAD_SKILL(\"<exact_skill_name>\")`; never return only hidden thinking or an empty response.",
                    f"- After a skill branch is loaded, no other skill can be loaded for the next {self.skill_cooldown_steps} environment step(s).",
                ]
            available_skills_text = self._available_skills_text(include_state_previews=True)
        sections.extend(
            [
                "\n".join(skill_policy_lines),
                "",
                "Available skills:",
                available_skills_text,
                "",
                "Game-specific skill trigger guidance:",
                self._game_skill_trigger_guidance(),
                "",
                "Active planner memo from recent skill branches:",
                self._active_planner_notes_text(),
                "",
                "Final instruction before responding:",
                self._main_skill_final_instruction(cooldown_remaining),
            ]
        )
        return "\n".join(sections)

    def _main_skill_final_instruction(self, cooldown_remaining: int) -> str:
        if cooldown_remaining > 0:
            if self.game_name == "super_mario_bros":
                return (
                    "Skill loading is currently unavailable because of cooldown. "
                    "Use the current screenshot, legal Mario actions, and active planner memo if present; "
                    "if the active memo has `action_constraints`, choose from its candidate action names when visually compatible; "
                    "output `thought:` and `move: (action_name, frame_count)`."
                )
            return (
                "Skill loading is currently unavailable because of cooldown. "
                "If you are uncertain, use the current screenshot, legal action rules, and active planner memo if present; "
                "output the best legal action in the required format."
            )
        if self.game_name == "super_mario_bros":
            return (
                "If the next Mario action is completely obvious and low-risk, output `thought:` and "
                "`move: (action_name, frame_count)`. "
                "If you are relying on any listed state preview to choose that action, request the matching skill branch instead. "
                "If there is meaningful uncertainty, hazard timing, platforming risk, recovery need, or weak x-progress, "
                "request one relevant skill branch with `move: LOAD_SKILL(\"<exact_skill_name>\")`. "
                "Use skills as advisory planning support only; they do not decide the move for you. "
                "Do not return an empty response."
            )
        return (
            "If the next legal action is completely obvious and low-risk, output the normal action response. "
            "If there is any meaningful uncertainty, strategic comparison, recovery need, or risk of wasting a move, "
            "use one relevant skill branch by putting "
            "`LOAD_SKILL(\"<exact_skill_name>\")` in the action/move field. "
            "Use skills as advisory planning support only; they do not decide the move for you. "
            "Do not return an empty response."
        )

    def _game_skill_trigger_guidance(self) -> str:
        if self.game_name == "sokoban":
            return "\n".join(
                [
                    "- Sokoban: consult a skill before pushing a box unless the push destination, worker push-side square, and deadlock safety are all visually clear.",
                    "- Sokoban: consult a skill after several worker-only moves, no boxes_on_target progress, or uncertainty about which box should be pushed next.",
                    "- Sokoban: prefer stall/deadlock/order skills when repeated movement is visible; prefer positioning skills only when a concrete target box and push direction are identifiable.",
                ]
            )
        if self.game_name == "candy_crush":
            return "\n".join(
                [
                    "- Candy Crush: consult a skill whenever several swaps are possible, the best scoring swap is unclear, or there is risk of None/invalid output.",
                    "- Candy Crush: prefer cascade/four-or-five-match skills for score improvement; use valid-swap skills as guardrails, not as the default scoring strategy.",
                    "- Candy Crush: the branch should compare candidate adjacent swaps and justify the recommended coordinate pair.",
                ]
            )
        if self.game_name == "tetris":
            return "\n".join(
                [
                    "- Tetris: consult a skill for each non-trivial piece placement before committing rotation, target lane, or hard_drop.",
                    "- Tetris: prefer skills when the model is repeatedly moving left/right/rotating without a clear landing plan.",
                    "- Tetris: the branch should produce a short sequence such as rotate/shift until aligned, then hard_drop when safe.",
                ]
            )
        if self.game_name == "super_mario_bros":
            return "\n".join(
                [
                    "- Super Mario Bros: consult a skill before committing to a jump/run action near enemies, pipes, pits, broken platforms, or airborne hazards.",
                    "- Super Mario Bros: consult a skill when Mario is repeatedly using no-op, plain right movement, or similar jumps without x-progress.",
                    "- Super Mario Bros: the branch should identify the local platforming pattern, recommend an action name and frame count, and state the visible safety check before following it.",
                    "- Super Mario Bros: direct action is acceptable only on clearly safe open ground or when the next jump/run timing is visually obvious.",
                ]
            )
        return "- Consult a skill whenever the board is strategically non-trivial or the next action needs comparison."

    def _available_skills_text(self, include_state_previews: bool) -> str:
        chunks: list[str] = []
        for name, bundle in self._skill_bundles.items():
            lines = [
                f"- `{name}`",
                f"  title: {bundle.title}",
                f"  purpose: {bundle.purpose}",
            ]
            if include_state_previews:
                state_lines = []
                for card in bundle.state_cards[:4]:
                    when = "; ".join(str(item) for item in (card.get("when_to_use") or [])[:2])
                    if not when:
                        continue
                    state_lines.append(
                        f"  state `{card.get('state_id')}`: when_to_use={when}"
                    )
                lines.extend(state_lines)
            chunks.append("\n".join(lines))
        return "\n\n".join(chunks) if chunks else "No skills are available for this game."

    def _active_planner_notes_text(self) -> str:
        if not self._active_planner_notes:
            return "None."
        chunks = []
        for note in self._active_planner_notes[-3:]:
            chunks.append(
                "\n".join(
                    [
                        f"- skill: {note.get('skill_name')}",
                        f"  applicability: {note.get('skill_applicability')}",
                        f"  subgoal: {note.get('subgoal')}",
                        f"  plan: {note.get('plan')}",
                        f"  state_pattern: {note.get('state_pattern')}",
                        f"  control_intent: {note.get('control_intent')}",
                        f"  risk_profile: {note.get('risk_profile')}",
                        f"  action_constraints: {note.get('action_constraints')}",
                        f"  state_checks: {note.get('state_checks')}",
                        f"  do_not_do: {note.get('do_not_do')}",
                        f"  fallback_if_no_progress: {note.get('fallback_if_no_progress')}",
                        f"  expected_state: {note.get('expected_state')}",
                    ]
                )
            )
        return "\n".join(chunks)

    @staticmethod
    def _extract_load_skill(response: str) -> Optional[str]:
        match = re.search(r"LOAD_SKILL\(\s*['\"]([^'\"]+)['\"]\s*\)", response or "", re.IGNORECASE)
        if not match:
            return None
        return match.group(1).strip()

    def _run_skill_branch(self, skill_name: str, observation: Observation, main_trigger_response: str) -> dict[str, Any]:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        record: dict[str, Any] = {
            "architecture_version": ARCHITECTURE_VERSION,
            "timestamp": timestamp,
            "step_index": self._skills_agent_step_index,
            "game_name": self.game_name,
            "requested_skill_name": skill_name,
            "main_trigger_response": main_trigger_response,
        }

        if skill_name not in self._skill_bundles:
            record["error"] = f"Unknown skill requested: {skill_name}"
            self._record_failed_branch(record)
            return record

        self._skill_usage_summary["load_skill_calls"] += 1
        self._skill_consult_counts[skill_name] = self._skill_consult_counts.get(skill_name, 0) + 1
        self._last_skill_load_step = self._skills_agent_step_index
        bundle = self._skill_bundles[skill_name]

        try:
            current_image = self._model_facing_current_image(observation)
            stage1_prompt = self._build_stage1_prompt(bundle, observation)
            stage1_response = self._call_branch_single_image(
                system_prompt=self._stage1_system_prompt(bundle),
                prompt=stage1_prompt,
                image_path=current_image,
            )
            stage1_decision = self._parse_stage1_decision(stage1_response, bundle)
            record["stage1_response"] = stage1_response
            record["stage1_decision"] = stage1_decision

            selected_views = self._resolve_selected_views(bundle, stage1_decision)
            record["selected_views"] = selected_views

            stage2_prompt = self._build_stage2_prompt(bundle, observation, stage1_decision, selected_views)
            if selected_views:
                image_paths = [current_image] + [item["image_path"] for item in selected_views]
                image_labels = ["Image 0: CURRENT live game screenshot."] + [
                    f"Image {idx}: skill={skill_name}; state={item['state_id']}; view={item['view']}."
                    for idx, item in enumerate(selected_views, start=1)
                ]
                stage2_response = self._call_branch_multi_image(
                    system_prompt=self._stage2_system_prompt(bundle),
                    prompt=stage2_prompt,
                    image_paths=image_paths,
                    image_labels=image_labels,
                )
            else:
                stage2_response = self._call_branch_single_image(
                    system_prompt=self._stage2_system_prompt(bundle),
                    prompt=stage2_prompt,
                    image_path=current_image,
                )

            planner_json, parse_error = self._parse_planner_json(stage2_response)
            if planner_json is None:
                planner_json = self._fallback_planner_json(bundle, parse_error or "Planner JSON parse failed.")
                record["planner_parse_error"] = parse_error
            planner_json["skill_name"] = skill_name
            planner_json["consult_count"] = self._skill_consult_counts.get(skill_name, 0)
            planner_json["skill_cooldown_steps"] = self.skill_cooldown_steps
            planner_json["stage1_decision"] = stage1_decision
            planner_json["selected_views"] = [
                {key: item[key] for key in ("state_id", "view", "rel_path", "evidence_goal", "reason") if key in item}
                for item in selected_views
            ]
            record["stage2_response"] = stage2_response
            record["planner_json"] = planner_json
            self._skill_usage_summary["successful_branch_plans"] += 1
        except Exception as exc:
            record["error"] = f"{exc.__class__.__name__}: {exc}"
            self._record_failed_branch(record)
            return record

        self._write_branch_record(record)
        return record

    def _stage1_system_prompt(self, bundle: SkillBundle) -> str:
        if self.game_name == "super_mario_bros":
            return (
                "You are Stage 1 of a temporary multimodal-skill branch for a Super Mario Bros agent. "
                "Inspect the current live screenshot and the selected skill metadata. "
                "Decide whether full-frame skill reference images are needed, and if so select the minimal relevant state cards. "
                "Do not choose the final Mario move in Stage 1."
            )
        return (
            "You are Stage 1 of a temporary multimodal-skill branch for a 2D game agent. "
            "Decide whether visual skill reference images are needed for the current live screenshot, "
            "and if so select minimal state-card views. Prefer loading a compact visual reference when it can help compare "
            "strategy, direction, transition, or action commitment. Do not choose the game action."
        )

    def _stage2_system_prompt(self, bundle: SkillBundle) -> str:
        if self.game_name == "super_mario_bros":
            return (
                "You are Stage 2 of a temporary planner-only multimodal-skill branch for a Super Mario Bros agent. "
                "Return structured JSON planning guidance only. Do not output the final environment action. "
                "Make the guidance concrete enough that the main agent can map it to a legal `move: (action_name, frame_count)` "
                "from the live screenshot."
            )
        return (
            "You are Stage 2 of a temporary planner-only multimodal-skill branch for a 2D game agent. "
            "Return structured JSON planning guidance only. Do not output the final game action. "
            "Make the guidance concrete enough that the main agent can map it to the next legal move from the live screenshot."
        )

    def _build_stage1_prompt(self, bundle: SkillBundle, observation: Observation) -> str:
        if self.game_name == "super_mario_bros":
            return "\n\n".join(
                [
                    "Inspect the CURRENT live Super Mario Bros screenshot and the selected skill metadata below.",
                    "Use the game-specific rules and action names below to understand legal moves, but do not choose the final move in Stage 1.",
                    "Game-specific rules and action format:",
                    self._branch_game_rules_text(observation),
                    "Return a JSON object only. Do not return prose.",
                    "View semantics for this Mario skill package:",
                    "- `full_frame`: the model-facing action example with a red arrow and action label for the recommended local Mario pattern.",
                    "Evidence goal:",
                    "- `inspect_action_example`: use `full_frame` to compare the live screenshot against the reusable platforming pattern.",
                    f"Select at most {self.max_stage1_selected_states} state(s) and {self.max_stage1_selected_views} total view(s).",
                    "Default toward visual_reference_needed=true when a state card could resolve jump/run timing, hazard interpretation, platforming commitment, or restart recovery.",
                    "Set visual_reference_needed=false only when no state card is relevant to the live screenshot or the selected skill is purely textual for this state.",
                    "JSON schema:",
                    json.dumps(
                        {
                            "visual_reference_needed": True,
                            "why_not_text_only": "why full-frame skill images are or are not needed",
                            "requests": [
                                {
                                    "state_id": "<exact state_id>",
                                    "views": ["full_frame"],
                                    "evidence_goal": "inspect_action_example",
                                    "reason": "why this state card helps with the current Mario screenshot",
                                }
                            ],
                        },
                        indent=2,
                    ),
                    "Skill text:",
                    self._truncate(bundle.skill_md, 5000),
                    "Runtime state cards:",
                    self._runtime_state_cards_text(bundle),
                ]
            )
        return "\n\n".join(
            [
                "Inspect the CURRENT live game screenshot and the skill metadata below.",
                "Use the game-specific rules and action names below to understand legal moves, but do not choose the final move in Stage 1.",
                "Game-specific rules and action format:",
                self._branch_game_rules_text(observation),
                "Return a JSON object only. Do not return prose.",
                "View semantics for this lmgame skill package:",
                "- `full_frame`: the model-facing action example with a red arrow for the next action direction or swap.",
                "- `before`: nearby predecessor frame for recognizing the pre-action condition.",
                "- `after`: nearby successor frame for verifying transition/progress.",
                "Evidence goals:",
                "- `inspect_action_example`: use `full_frame` only.",
                "- `recognize_before`: use `before`, optionally `full_frame`.",
                "- `verify_after`: use `after`, optionally `full_frame`.",
                "- `compare_transition`: use a minimal combination of `before`, `full_frame`, and/or `after`.",
                f"Select at most {self.max_stage1_selected_states} state(s) and {self.max_stage1_selected_views} total view(s).",
                "Default toward visual_reference_needed=true when a state card could resolve strategy, direction, coordinate, or transition uncertainty.",
                "Do not set visual_reference_needed=false merely because the live screenshot is visible; skill images are for comparison against reusable patterns.",
                "Set visual_reference_needed=false only when the selected skill is a pure textual guardrail or no state card is relevant to the live board.",
                "JSON schema:",
                json.dumps(
                    {
                        "visual_reference_needed": True,
                        "why_not_text_only": "why skill images are or are not needed",
                        "requests": [
                            {
                                "state_id": "<exact state_id>",
                                "views": ["full_frame"],
                                "evidence_goal": "inspect_action_example",
                                "reason": "why these views resolve the current uncertainty",
                            }
                        ],
                    },
                    indent=2,
                ),
                "Skill text:",
                self._truncate(bundle.skill_md, 5000),
                "Runtime state cards:",
                self._runtime_state_cards_text(bundle),
            ]
        )

    def _runtime_state_cards_text(self, bundle: SkillBundle) -> str:
        rows = []
        for card in bundle.state_cards:
            views = sorted((card.get("views") or {}).keys())
            rows.append(
                json.dumps(
                    {
                        "state_id": card.get("state_id"),
                        "state_name": card.get("state_name"),
                        "when_to_use": card.get("when_to_use"),
                        "when_not_to_use": card.get("when_not_to_use"),
                        "expected_action": card.get("expected_action"),
                        "action_sequence_hint": card.get("action_sequence_hint"),
                        "verification_cue": card.get("verification_cue"),
                        "priority": card.get("priority"),
                        "planner_requirements": card.get("planner_requirements"),
                        "quality_note": card.get("quality_note"),
                        "transfer_note": card.get("transfer_note"),
                        "non_transferable": card.get("non_transferable"),
                        "available_views": views,
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(rows)

    def _build_stage2_prompt(
        self,
        bundle: SkillBundle,
        observation: Observation,
        stage1_decision: dict[str, Any],
        selected_views: list[dict[str, Any]],
    ) -> str:
        image_labels = ["Image 0 is the CURRENT live game screenshot."]
        for idx, item in enumerate(selected_views, start=1):
            image_labels.append(
                f"Image {idx} is skill reference `{item['state_id']}` view `{item['view']}` "
                f"for evidence_goal `{item.get('evidence_goal')}`."
            )
        if self.game_name == "super_mario_bros":
            planner_requirements = [
                "- Ground the plan in visible live-state features; do not merely restate the skill title.",
                "- Name the visible local platforming pattern: safe run, enemy/pipe jump, enemy cluster, gap/platform, airborne hazard, restart, or uncertain.",
                "- Produce a concrete local subgoal and 2-4 executable checks/actions/transitions grounded in the live screenshot.",
                "- Do not output a concrete final move, exact frame count, or action sequence.",
                "- Provide `action_constraints.candidate_actions` as a small set of legal Mario action names that the main agent may choose from; do not include frame counts.",
                "- Provide `action_constraints.discouraged_actions` for action names that would be risky in this screenshot.",
                "- Use `frame_count_band` only as a qualitative band: tap, short, medium, long, commit, or not_applicable.",
                "- Explain timing qualitatively using visible distance, enemy/hazard spacing, jump arc, momentum need, or landing safety.",
                "- Include visible `state_checks` such as landing ground, enemy distance, pipe/gap position, airborne hazard clearance, or whether Mario is stable/falling.",
                "- Include `do_not_do` for the most likely unsafe action, such as walking off a gap, late jump, overlong run into a pipe, panic no-op, or copying the skill image when the live screenshot differs.",
            ]
        else:
            planner_requirements = [
                "- Ground the plan in visible live-state features; do not merely restate the skill title.",
                "- Produce a concrete local subgoal and a short executable plan, not a vague strategy slogan.",
                "- Do not output a concrete final move, exact frame count, or action sequence; explain the state checks needed before acting.",
                "- If the game has discrete actions, provide `action_constraints.candidate_actions` as a compact set of plausible action names, not a full ordered action sequence.",
                "- Sokoban: name the target box, push direction, worker push-side square, destination safety/deadlock check, and whether the immediate next move is positioning or pushing.",
                "- Candy Crush: compare at least two plausible adjacent swaps when visible; include the recommended coordinate pair, expected immediate match, and why it is better than the fallback.",
                "- Tetris: name piece type/orientation if visible, target lane/column, rotation/shift sequence, hard_drop condition, and avoid vague repeated left/right/rotate.",
                "- Super Mario Bros: name the visible local pattern (safe run, enemy/pipe jump, enemy cluster, gap/platform, airborne hazard, or restart), explain the local timing/commitment qualitatively, and include a safety check such as landing ground, enemy distance, or hazard clearance.",
            ]

        return "\n\n".join(
            [
                "Return planner JSON only. The main agent will choose the actual move after reading it.",
                "The CURRENT live screenshot is authoritative. Skill images are supplemental strategy references only.",
                "Use the game-specific rules and action names below to keep the planner legal, but do not output the final move in Stage 2.",
                "Game-specific rules and action format:",
                self._branch_game_rules_text(observation),
                "Image labels:",
                "\n".join(f"- {item}" for item in image_labels),
                "Stage-1 decision:",
                json.dumps(stage1_decision, ensure_ascii=False, indent=2),
                "Selected skill view metadata:",
                json.dumps(
                    [
                        {key: item[key] for key in ("state_id", "view", "rel_path", "evidence_goal", "reason") if key in item}
                        for item in selected_views
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                "Skill text:",
                self._truncate(bundle.skill_md, 5000),
                "Runtime state cards:",
                self._runtime_state_cards_text(bundle),
                "Planner requirements:",
                "\n".join(planner_requirements),
                "Planner JSON schema:",
                json.dumps(
                    {
                        "skill_applicability": "effective | ineffective | uncertain",
                        "state_pattern": "safe_run | enemy_or_pipe | enemy_cluster | gap_or_platform | vertical_hazard | projectile_pressure | castle_lava_firebar | restart_recovery | uncertain",
                        "subgoal": "next immediate local milestone",
                        "plan": "2-4 concrete checks/actions/transitions grounded in the live screenshot",
                        "control_intent": {
                            "direction": "right | left | neutral",
                            "speed": "walk | run | hold | slow_down",
                            "jump": "none | tap | short | medium | long",
                            "timing": "now | after_small_positioning | wait_for_hazard_window | after_landing | after_enemy_spacing",
                            "commitment": "low | medium | high",
                        },
                        "risk_profile": {
                            "overjump_risk": "low | medium | high",
                            "underjump_risk": "low | medium | high",
                            "collision_risk": "low | medium | high",
                            "stall_risk": "low | medium | high",
                        },
                        "action_constraints": {
                            "candidate_actions": ["right_a", "right_a_b"],
                            "discouraged_actions": ["noop", "right"],
                            "frame_count_band": "tap | short | medium | long | commit | not_applicable",
                            "selection_rule": "how the main agent should choose among candidates using the live screenshot",
                        },
                        "state_checks": "visible checks that justify following the recommendation",
                        "do_not_do": "most likely wrong path to avoid",
                        "fallback_if_no_progress": "concrete alternate plan if the skill-guided path stalls",
                        "expected_state": "visible cue to aim for after applying the local plan",
                        "completion_scope": "local_only | needs_verification | maybe_complete",
                    },
                    indent=2,
                ),
                "Return exactly one JSON object and no Markdown prose.",
            ]
        )

    def _branch_game_rules_text(self, observation: Observation) -> str:
        base_module = self.modules["base_module"]
        try:
            rendered_prompt = observation.get_complete_prompt(
                observation_mode=self.observation_mode,
                prompt_template=base_module.prompt,
            )
        except Exception:
            rendered_prompt = base_module.prompt
        sections = []
        if base_module.system_prompt:
            sections.append("System role:\n" + base_module.system_prompt)
        if rendered_prompt:
            sections.append("Main action prompt:\n" + rendered_prompt)
        return self._truncate("\n\n".join(sections), 6500)

    def _parse_stage1_decision(self, response: str, bundle: SkillBundle) -> dict[str, Any]:
        payload = self._extract_json_object(response)
        if not isinstance(payload, dict):
            payload = {
                "visual_reference_needed": True,
                "why_not_text_only": "Stage-1 response was not parseable, so load one minimal action example.",
                "requests": [],
            }

        visual_needed = payload.get("visual_reference_needed")
        if isinstance(visual_needed, str):
            visual_needed = visual_needed.strip().lower() in {"true", "yes", "1"}
        else:
            visual_needed = bool(visual_needed)
        requests_raw = payload.get("requests") if isinstance(payload.get("requests"), list) else []
        why = str(payload.get("why_not_text_only") or "").strip()

        normalized_requests: list[dict[str, Any]] = []
        known_cards = {str(card.get("state_id")): card for card in bundle.state_cards}
        total_views = 0
        if visual_needed:
            for item in requests_raw:
                if not isinstance(item, dict):
                    continue
                state_id = str(item.get("state_id") or "").strip()
                if state_id not in known_cards:
                    continue
                evidence_goal = str(item.get("evidence_goal") or "inspect_action_example").strip()
                if evidence_goal not in EVIDENCE_GOALS:
                    evidence_goal = "inspect_action_example"
                card_views = known_cards[state_id].get("views") or {}
                views: list[str] = []
                for view in item.get("views") or []:
                    view_name = str(view).strip()
                    if view_name in ALLOWED_VIEWS and view_name in card_views and view_name not in views:
                        views.append(view_name)
                if not views:
                    views = ["full_frame"] if "full_frame" in card_views else sorted(card_views.keys())[:1]
                if not views:
                    continue
                remaining = self.max_stage1_selected_views - total_views
                if remaining <= 0 or len(normalized_requests) >= self.max_stage1_selected_states:
                    break
                views = views[:remaining]
                total_views += len(views)
                normalized_requests.append(
                    {
                        "state_id": state_id,
                        "views": views,
                        "evidence_goal": evidence_goal,
                        "reason": str(item.get("reason") or "Selected by Stage 1.").strip(),
                    }
                )

            if not normalized_requests and bundle.state_cards:
                first = bundle.state_cards[0]
                card_views = first.get("views") or {}
                fallback_views = ["full_frame"] if "full_frame" in card_views else sorted(card_views.keys())[:1]
                if fallback_views:
                    normalized_requests.append(
                        {
                            "state_id": str(first.get("state_id")),
                            "views": fallback_views,
                            "evidence_goal": "inspect_action_example",
                            "reason": "Fallback minimal visual reference because Stage 1 did not select valid views.",
                        }
                    )

        decision = {
            "visual_reference_needed": bool(normalized_requests) if visual_needed else False,
            "why_not_text_only": why or ("Skill image references selected." if normalized_requests else "Text and current screenshot are sufficient."),
            "requests": normalized_requests if visual_needed else [],
        }
        self._record_stage1_decision(decision)
        return decision

    def _resolve_selected_views(self, bundle: SkillBundle, decision: dict[str, Any]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        card_by_id = {str(card.get("state_id")): card for card in bundle.state_cards}
        for request in decision.get("requests") or []:
            card = card_by_id.get(str(request.get("state_id")))
            if not card:
                continue
            views = card.get("views") or {}
            for view in request.get("views") or []:
                rel_path = views.get(view)
                if not rel_path:
                    continue
                image_path = bundle.path / rel_path
                if not image_path.exists():
                    continue
                selected.append(
                    {
                        "state_id": str(card.get("state_id")),
                        "view": str(view),
                        "rel_path": str(rel_path),
                        "image_path": str(image_path),
                        "evidence_goal": request.get("evidence_goal"),
                        "reason": request.get("reason"),
                    }
                )
                if len(selected) >= self.max_stage1_selected_views:
                    return selected
        return selected

    def _parse_planner_json(self, response: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        payload = self._extract_json_object(response)
        if not isinstance(payload, dict):
            return None, "No JSON object found in Stage-2 response."

        applicability = str(payload.get("skill_applicability") or "").strip().lower()
        if applicability not in {"effective", "ineffective", "uncertain"}:
            return None, "`skill_applicability` must be effective, ineffective, or uncertain."
        completion_scope = str(payload.get("completion_scope") or "").strip().lower()
        if completion_scope not in {"local_only", "needs_verification", "maybe_complete"}:
            return None, "`completion_scope` must be local_only, needs_verification, or maybe_complete."

        required = ["subgoal", "plan", "do_not_do", "fallback_if_no_progress", "expected_state"]
        missing = [key for key in required if not str(payload.get(key) or "").strip()]
        if missing:
            return None, f"Missing required planner fields: {', '.join(missing)}."

        planner = {
            "skill_applicability": applicability,
            "state_pattern": str(payload.get("state_pattern") or "uncertain").strip(),
            "subgoal": str(payload["subgoal"]).strip(),
            "plan": str(payload["plan"]).strip(),
            "control_intent": self._normalize_control_intent(payload.get("control_intent")),
            "risk_profile": self._normalize_risk_profile(payload.get("risk_profile")),
            "action_constraints": self._normalize_action_constraints(payload.get("action_constraints"), payload.get("control_intent")),
            "do_not_do": str(payload["do_not_do"]).strip(),
            "fallback_if_no_progress": str(payload["fallback_if_no_progress"]).strip(),
            "expected_state": str(payload["expected_state"]).strip(),
            "completion_scope": completion_scope,
        }
        for key in [
            "state_checks",
            "confidence",
            "why_this_skill",
            "risk_if_wrong",
        ]:
            if key in payload:
                value = payload.get(key)
                if isinstance(value, (list, dict)):
                    planner[key] = value
                else:
                    planner[key] = str(value or "").strip()
        return planner, None

    def _normalize_control_intent(self, value: Any) -> dict[str, str]:
        data = value if isinstance(value, dict) else {}
        defaults = {
            "direction": "neutral",
            "speed": "hold",
            "jump": "none",
            "timing": "now",
            "commitment": "medium",
        }
        allowed = {
            "direction": {"right", "left", "neutral"},
            "speed": {"walk", "run", "hold", "slow_down"},
            "jump": {"none", "tap", "short", "medium", "long"},
            "timing": {"now", "after_small_positioning", "wait_for_hazard_window", "after_landing", "after_enemy_spacing"},
            "commitment": {"low", "medium", "high"},
        }
        result: dict[str, str] = {}
        for key, default in defaults.items():
            item = str(data.get(key) or default).strip().lower()
            result[key] = item if item in allowed[key] else default
        return result

    def _normalize_risk_profile(self, value: Any) -> dict[str, str]:
        data = value if isinstance(value, dict) else {}
        result: dict[str, str] = {}
        for key in ["overjump_risk", "underjump_risk", "collision_risk", "stall_risk"]:
            item = str(data.get(key) or "medium").strip().lower()
            result[key] = item if item in {"low", "medium", "high"} else "medium"
        return result

    def _normalize_action_constraints(self, value: Any, control_intent: Any = None) -> dict[str, Any]:
        data = value if isinstance(value, dict) else {}
        legal_mario_actions = {
            "noop", "right", "right_a", "right_b", "right_a_b", "a", "b",
            "left", "left_a", "left_b", "left_a_b", "down", "up",
        }

        def normalize_action_list(raw: Any) -> list[str]:
            if isinstance(raw, str):
                items = [raw]
            elif isinstance(raw, list):
                items = raw
            else:
                items = []
            output: list[str] = []
            for item in items:
                action = str(item or "").strip().lower()
                if action in legal_mario_actions and action not in output:
                    output.append(action)
            return output[:5]

        candidates = normalize_action_list(data.get("candidate_actions"))
        discouraged = normalize_action_list(data.get("discouraged_actions"))
        if self.game_name == "super_mario_bros" and not candidates:
            candidates = self._infer_mario_candidate_actions(control_intent)
        band = str(data.get("frame_count_band") or "medium").strip().lower()
        if band not in {"tap", "short", "medium", "long", "commit", "not_applicable"}:
            band = "medium"
        selection_rule = str(data.get("selection_rule") or "").strip()
        if not selection_rule:
            selection_rule = "Choose among candidate actions using only the current screenshot; avoid discouraged actions unless the candidates are visibly unsafe."
        return {
            "candidate_actions": candidates,
            "discouraged_actions": discouraged,
            "frame_count_band": band,
            "selection_rule": selection_rule,
        }

    def _infer_mario_candidate_actions(self, control_intent: Any) -> list[str]:
        intent = self._normalize_control_intent(control_intent)
        direction = intent.get("direction")
        speed = intent.get("speed")
        jump = intent.get("jump")
        timing = intent.get("timing")
        if timing == "wait_for_hazard_window":
            return ["noop", "right"]
        if direction == "left":
            if jump in {"short", "medium", "long"} and speed == "run":
                return ["left_a_b", "left_a"]
            if jump in {"tap", "short", "medium", "long"}:
                return ["left_a", "left"]
            if speed == "run":
                return ["left_b", "left"]
            return ["left", "noop"]
        if direction == "right":
            if jump in {"short", "medium", "long"} and speed == "run":
                return ["right_a_b", "right_a"]
            if jump in {"tap", "short", "medium", "long"}:
                return ["right_a", "right_a_b"]
            if speed == "run":
                return ["right_b", "right"]
            if speed == "walk":
                return ["right", "right_b"]
            return ["right", "noop"]
        if jump in {"tap", "short", "medium", "long"}:
            return ["a", "right_a"]
        return ["noop", "right"]

    @staticmethod
    def _extract_json_object(response: str) -> Optional[Any]:
        text = response or ""
        code_blocks = re.findall(r"```(?:json|python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        candidates = code_blocks + [text]
        for candidate in candidates:
            body = candidate.strip()
            load_match = re.search(r"LOAD_STATE_VIEWS\((.*)\)\s*;?\s*$", body, flags=re.DOTALL)
            if load_match:
                body = load_match.group(1).strip()
            try:
                return json.loads(body)
            except Exception:
                pass
            start = body.find("{")
            end = body.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(body[start : end + 1])
                except Exception:
                    pass
        return None

    def _fallback_planner_json(self, bundle: SkillBundle, reason: str) -> dict[str, Any]:
        return {
            "skill_applicability": "uncertain",
            "state_pattern": "uncertain",
            "subgoal": f"Use {bundle.title} cautiously only if the live screenshot matches the skill conditions.",
            "plan": (
                f"Apply the skill purpose if it matches the current board: {bundle.purpose} "
                "First verify the current screenshot, then choose one legal action that improves the local game state."
            ),
            "control_intent": {
                "direction": "neutral",
                "speed": "hold",
                "jump": "none",
                "timing": "now",
                "commitment": "low",
            },
            "risk_profile": {
                "overjump_risk": "medium",
                "underjump_risk": "medium",
                "collision_risk": "medium",
                "stall_risk": "medium",
            },
            "action_constraints": {
                "candidate_actions": [],
                "discouraged_actions": ["noop"],
                "frame_count_band": "medium",
                "selection_rule": "Ignore fallback candidates if the current screenshot does not visibly match the skill condition.",
            },
            "state_checks": "Verify that the live board visibly matches the skill condition before using the fallback guidance.",
            "do_not_do": "Do not copy an example action or coordinate from the skill image when the live board differs.",
            "fallback_if_no_progress": (
                "If the skill pattern is not clearly visible, ignore the skill and choose the simplest legal action "
                "that avoids invalid moves, deadlocks, or lateral oscillation."
            ),
            "expected_state": "The next screenshot shows a legal visible state transition or preserves safety without invalid action.",
            "completion_scope": "needs_verification",
            "planner_parse_error": reason,
        }

    def _build_action_with_planner_prompt(self, custom_prompt: Optional[str], planner_note: dict[str, Any]) -> str:
        sections: list[str] = []
        if custom_prompt:
            sections.append(custom_prompt)
        sections.extend(
            [
                "A temporary multimodal-skill branch returned the planner JSON below.",
                "Use it as supplemental guidance for the CURRENT live screenshot, then output the actual next game action in the required base format.",
                "Explicitly consider the planner `subgoal`, `plan`, `state_pattern`, `control_intent`, `risk_profile`, `action_constraints`, `state_checks`, `do_not_do`, `fallback_if_no_progress`, and `expected_state` before acting.",
                "The planner does not provide a final move or exact frame count. It provides candidate action constraints that narrow the action space.",
                "If `action_constraints.candidate_actions` contains legal actions that are visually compatible with the CURRENT screenshot, choose one of those action names and infer the frame count from `frame_count_band`.",
                "Frame-count band guidance for Mario: tap=1-4, short=5-10, medium=11-18, long=19-25, commit=26-30. Adjust within the band using visible distance and hazard timing.",
                "Avoid `action_constraints.discouraged_actions` unless every candidate action is visibly unsafe. If you choose outside the candidates, explain the visual reason in `thought:`.",
                "Do not output LOAD_SKILL now. Do not output JSON. Do not copy skill-image coordinates blindly.",
                "Do not ignore the planner without a concrete visual reason. If the planner conflicts with the current screenshot, follow the current screenshot.",
                json.dumps(planner_note, ensure_ascii=False, indent=2),
            ]
        )
        return "\n\n".join(sections)

    def _build_action_after_failed_branch_prompt(self, custom_prompt: Optional[str], skill_name: str, error: str) -> str:
        sections: list[str] = []
        if custom_prompt:
            sections.append(custom_prompt)
        sections.extend(
            [
                f"The requested skill branch `{skill_name}` failed: {error}",
                "Continue from the CURRENT live screenshot without loading another skill in this step.",
                "Output the actual next game action in the required base format. Do not output LOAD_SKILL.",
            ]
        )
        return "\n\n".join(sections)

    def _call_branch_single_image(self, system_prompt: str, prompt: str, image_path: str) -> str:
        response = self.modules["base_module"].api_manager.vision_text_completion(
            model_name=self.model_name,
            system_prompt=system_prompt,
            prompt=prompt,
            image_path=image_path,
            thinking=True,
            reasoning_effort=self.modules["base_module"].reasoning_effort,
            token_limit=self.token_limit,
        )
        return response[0] if isinstance(response, tuple) else str(response)

    def _call_branch_multi_image(
        self,
        system_prompt: str,
        prompt: str,
        image_paths: list[str],
        image_labels: list[str],
    ) -> str:
        labeled_prompt = "\n\n".join(["Image labels:", *image_labels, "", prompt])
        response = self.modules["base_module"].api_manager.multi_image_completion(
            model_name=self.model_name,
            system_prompt=system_prompt,
            prompt=labeled_prompt,
            list_content=image_labels,
            list_image_paths=image_paths,
            temperature=0,
            reasoning_effort=self.modules["base_module"].reasoning_effort,
            token_limit=self.token_limit,
        )
        return response[0] if isinstance(response, tuple) else str(response)

    def _model_facing_current_image(self, observation: Observation) -> str:
        image_path = scale_image_up(observation.get_img_path())
        if not image_path:
            raise ValueError("SkillsAgent requires a current image for vision-mode skill branches.")
        return image_path

    def _skill_cooldown_remaining(self) -> int:
        if self._last_skill_load_step is None or self.skill_cooldown_steps <= 0:
            return 0
        elapsed = self._skills_agent_step_index - self._last_skill_load_step
        return max(0, self.skill_cooldown_steps - elapsed)

    def _record_stage1_decision(self, decision: dict[str, Any]) -> None:
        visual_key = "true" if decision.get("visual_reference_needed") else "false"
        counts = self._skill_usage_summary.setdefault("stage1_visual_reference_needed_counts", {})
        counts[visual_key] = int(counts.get(visual_key, 0)) + 1
        goal_counts = self._skill_usage_summary.setdefault("stage1_evidence_goal_counts", {})
        for item in decision.get("requests") or []:
            goal = str(item.get("evidence_goal") or "")
            if goal:
                goal_counts[goal] = int(goal_counts.get(goal, 0)) + 1

    def _update_active_planner_notes(self, planner_note: dict[str, Any]) -> None:
        if planner_note.get("skill_applicability") == "ineffective":
            return
        compact = {
            key: planner_note.get(key)
            for key in [
                "skill_name",
                "skill_applicability",
                "state_pattern",
                "subgoal",
                "plan",
                "control_intent",
                "risk_profile",
                "action_constraints",
                "state_checks",
                "do_not_do",
                "fallback_if_no_progress",
                "expected_state",
                "completion_scope",
                "consult_count",
                "skill_cooldown_steps",
            ]
        }
        compact["created_step_index"] = self._skills_agent_step_index
        self._active_planner_notes.append(compact)
        self._active_planner_notes = self._active_planner_notes[-3:]

    def _record_failed_branch(self, record: dict[str, Any]) -> None:
        self._skill_usage_summary["failed_branch_plans"] += 1
        self._write_branch_record(record)

    def _write_branch_record(self, record: dict[str, Any]) -> None:
        if not self.skill_logs_dir:
            return
        branch_log = self.skill_logs_dir / "branch_log.jsonl"
        with branch_log.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        record["branch_log_path"] = str(branch_log)
        self._write_skill_usage_summary()

    def _write_skill_usage_summary(self) -> None:
        if not self.skill_logs_dir:
            return
        summary_path = self.skill_logs_dir / "skill_usage_summary.json"
        summary = dict(self._skill_usage_summary)
        summary["skill_consult_counts"] = dict(self._skill_consult_counts)
        summary["loaded_skills"] = sorted(self._skill_bundles)
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]..."
