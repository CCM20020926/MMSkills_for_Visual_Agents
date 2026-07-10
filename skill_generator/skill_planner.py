from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from models import ClusterPlan
from dataloader import AgentNetLoader

class SkillPlanner:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(ClusterPlan)

    def plan_cluster(self, cluster_trajs) -> ClusterPlan:
        domain = cluster_trajs[0].domain if cluster_trajs else 'unknown'

        all_success_segments = []
        all_failed_steps = []
        for traj in cluster_trajs:
            segs, fails = AgentNetLoader.split_by_success_with_id(traj)
            all_success_segments.extend(segs)
            all_failed_steps.extend(fails)

        success_summary = ""
        for task_id, seg in all_success_segments[:15]:
            actions = [f"{s.action}" for s in seg if s.action]
            success_summary += f"Task {task_id}: " + " -> ".join(actions) + "\n"
        failure_summary = ""
        for task_id, fstep in all_failed_steps[:15]:
            failure_summary += (f"Task {task_id}: Action: {fstep.action}, "
                                f"Observation: {fstep.observation}, "
                                f"Reflection: {fstep.reflection}\n")

        prompt = ChatPromptTemplate.from_template("""
You are a skill planner. Given successful action sequences and failed steps for a cluster of similar tasks from the **{domain}** application domain, extract:
1. A set of atomic skills (reusable workflows) from the successful sequences.
2. A set of common failure patterns from the failed steps.

Successful sequences (each task has an ID):
{success_summary}

Failed steps (each task has an ID):
{failure_summary}

Output a JSON object with two keys:
- "skills": list of objects, each with:
    {skill_name, description, workflow_boundary (a concise textual description of the entire workflow, e.g., 'From opening the New Tab page to adding a shortcut and verifying its appearance'), completion_criteria, covered_task_ids (list of original task IDs that this skill applies to)}
- "failure_patterns": list of objects, each with:
    {failure_description, typical_state_before_failure, action_that_caused_failure, suggested_prevention}
""")
        return self.llm.invoke(prompt.format_messages(
            domain=domain,
            success_summary=success_summary,
            failure_summary=failure_summary
        ))