import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from models import Plan

class TextDrafter:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(Plan)

    def draft_plan(self, merged_skill, domain):
        prompt = ChatPromptTemplate.from_template("""
You are a skill documenter. Create a detailed plan for the following skill from the **{domain}** domain.
Skill summary:
- skill_name: {skill_name}
- description: {description}
- workflow_boundary: {workflow_boundary}
- completion_criteria: {completion_criteria}
Common failure modes for this skill: {failure_modes}

The plan must include:
- overview
- when_to_use (list of scenarios)
- preconditions
- atomic_capabilities (typically reach_surface and execute_and_verify)
- decision_guide (if surface not open, use reach; otherwise use execute)
- procedures: one procedure with states. Each state must have:
   - state_id (1,2,3...)
   - state_name
   - visual_grounding (text description of what the screen looks like)
   - trigger_condition
   - action (what to do)
   - is_result_state (bool)
   - has_image (True)
   - text_description
   - key_frame: with image_filename (placeholder like "step1.png") and highlight_targets
- common_failure_modes (use the provided list)
- skill_slug (derive from domain and skill_name, e.g., CHROME_Add_Shortcut)
- skill_name

Return a JSON object matching the Plan schema.
""")
        plan = self.llm.invoke(prompt.format_messages(
            domain=domain,
            skill_name=merged_skill.get('skill_name', ''),
            description=merged_skill.get('description', ''),
            workflow_boundary=merged_skill.get('workflow_boundary', ''),
            completion_criteria=merged_skill.get('completion_criteria', ''),
            failure_modes=json.dumps(merged_skill.get('common_failure_modes', []), indent=2)
        ))
        return plan

    def draft_markdown(self, plan: Plan):
        lines = [
            f"---\nname: {plan.skill_name}\ndescription: {plan.overview}\n---",
            f"# {plan.skill_name}",
            "## Overview",
            plan.overview,
            "## When This Skill Applies",
            "\n".join([f"- {w}" for w in plan.when_to_use]),
            "## Visual State Card Usage",
            "Use `runtime_state_cards.json` for runtime branch loading. The runtime should load only the card whose `when_to_use` matches the current screenshot.",
        ]
        for proc in plan.procedures:
            for state in proc.states:
                lines.append(f"- `Images/{state.key_frame.image_filename}`: {state.text_description}")
        lines.append("Red boxes mark interaction cues. Green boxes mark state or verification cues.")
        lines.append("## Procedure")
        for proc in plan.procedures:
            for i, state in enumerate(proc.states):
                lines.append(f"{i+1}. {state.text_description} (see `Images/{state.key_frame.image_filename}`)")
        lines.append("## Visual Transfer Limits")
        lines.append("- Do not copy example values or window layout from the images.")
        lines.append("- Do not assume elements appear at the same screen coordinates.")
        lines.append("## Result Verification Cues")
        result_states = [s for proc in plan.procedures for s in proc.states if s.is_result_state]
        if result_states:
            last = result_states[-1]
            lines.append(f"- {last.text_description} and verify that the expected outcome is visible.")
        lines.append("## Common Failure Modes")
        for fail in plan.common_failure_modes:
            lines.append(f"- {fail}")
        return "\n".join(lines)