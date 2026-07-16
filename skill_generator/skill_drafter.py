import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from models import SkillPlan, Trajectory, RuntimeStateCards

class TextDrafter:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(SkillPlan)

    def draft_plan(self, merged_skill, domain, trajectories: list[Trajectory], example_plans: list[dict]):
        
        examples_text = ""
        if example_plans:
            for idx, ex_plan in enumerate(example_plans):
                ex_json = json.dumps(ex_plan, indent=2, ensure_ascii=False)
                examples_text += f"\n--- Example Plan {idx+1} ---\n```json\n{ex_json}\n```\n"
        
        prompt = ChatPromptTemplate.from_template("""
You are a skill documenter. Create a detailed plan for the following skill from the **{domain}** domain.

Skill summary:
- skill_name: {skill_name}
- description: {description}
- workflow_boundary: {workflow_boundary}
- completion_criteria: {completion_criteria}
Common failure modes for this skill: {failure_modes}

Below are **representative trajectories** (real execution steps). 
They may include successful completions, error handling, or recovery actions. 
Use them as the factual basis to derive a generalized and robust procedure:
- Identify the common stages across trajectories.
- For each state, base `visual_grounding` and `trigger_condition` on the observations.
- Base `action` on the actual actions taken (normalize variations).
- Account for possible error paths in the `decision_guide` and `common_failure_modes`.

Each trajectory is a JSON object with the following fields:
- instruction: the task instruction
- steps: a list of steps, each with:
  - observation: what the agent sees (text description)
  - action: what the agent does (text description)
  - reflection: The demonstration or analysis of the action result

Trajectories:
{trajectories}

**Reference Examples of well-structured SkillPlan:**
{examples_text}

The plan must include:
- overview
- when_to_use (list of scenarios)
- preconditions
- atomic_capabilities (typically reach_surface and execute_and_verify)
- decision_guide (if surface not open, use reach; otherwise use execute)
- procedures: one procedure with states. Each state must have:
   - state_id (1,2,3...)
   - state_name (short, descriptive)
   - visual_grounding (text description of what the screen looks like)
   - trigger_condition (when this state becomes active)
   - action (what to do, based on trajectory actions)
   - is_result_state (bool)
   - has_image (True)
   - text_description
   - key_frame: with image_filename (placeholder like "step1.png") and highlight_targets (name, target_type, annotation_query, color)
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
            trajectories=json.dumps(self._format_trajectories(trajectories), indent=2),
            failure_modes=json.dumps(merged_skill.get('common_failure_modes', []), indent=2),
            examples_text=examples_text
        ))
        return plan

    def _format_trajectories(self, trajectories: list[Trajectory]):
        results = []
        
        for traj in trajectories:
            traj_dict = {
                "instruction": traj.instruction,
                "steps": []
            }
            for step in traj.steps:
                step_dict = {
                    "observation": step.observation,
                    "action": step.action,
                    "reflection": step.reflection
                }
                traj_dict["steps"].append(step_dict)
            results.append(traj_dict)
        
        return results

    def draft_markdown(self, plan: SkillPlan, cards: RuntimeStateCards, example_markdowns) -> str:
        """
        结合 plan 和 runtime_state_cards，利用少样本示例生成 SKILL.md。
        """

        # 将 plan 和 cards 转为结构化文本
        plan_text = self._plan_to_text(plan)
        cards_text = self._cards_to_text(cards)

        # 构建 few-shot 示例
        examples_text = "\n\n---\n\n".join(example_markdowns)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a technical writer creating SKILL.md documentation for AI agents. The documentation should be clear, structured, and practical."),
            ("user", f"""
Here are some examples of well-written SKILL.md files:

{examples_text}

Now, based on the following skill plan and runtime state cards, generate a new SKILL.md file.
Follow the same structure and level of detail as the examples.

**Skill Plan (overview, procedures, atomic capabilities, etc.):**
{plan_text}

**Runtime State Cards (per-state decision cues):**
{cards_text}

**Requirements:**
- Output pure Markdown, with frontmatter (---) at the top.
- Include sections: Overview, When This Skill Applies, Preconditions, Visual State Card Usage, Visual Transfer Limits, Result Verification Cues, Atomic Capabilities, Procedures, Common Failure Modes.
- Use the runtime state cards to enrich the Procedure descriptions: each state's `when_to_use` and `visible_cues` can be summarized in the corresponding procedure step.
- The "Visual State Card Usage" section should explain how to use the cards at runtime.
- Ensure the output is self-contained and actionable for an agent.
""")
        ])
        response = self.llm.invoke(prompt)
        return response.content
    
    
    def _plan_to_text(self, plan: SkillPlan):
        text = f"```json\n{json.dumps(plan.model_dump(), ensure_ascii=False, indent=2)}\n```"
        return text
    
    def _plan_to_cards(self, cards: RuntimeStateCards):
        text = f"```json\n{json.dumps(cards.model_dump(), ensure_ascii=False, indent=2)}\n```"
        return text
        
        