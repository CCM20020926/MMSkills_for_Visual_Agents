import json
from langchain_openai import ChatOpenAI

class SkillPlanner:
    def __init__(self, llm_client, meta_guide):
        self.llm = llm_client
        self.guide = meta_guide
    
    def plan_for_cluster(self, cluster_trajs):
        prompt = f"""
        You are a skill planner. Analyze the following task trajectories (each includes instruction and action summary) and propose atomic skills that can be reused across multiple tasks.
        Each skill must have:
        - "skill_name": concise name
        - "description": what it accomplishes
        - "workflow": list of high-level steps (e.g., ["open app", "click menu", "fill form"])
        - "completion_criteria": how to know it's done
        - "covered_task_ids": list of task indices (0-based) from the cluster that this skill applies to.
        
        Trajectories:
        {json.dumps(cluster_trajs, indent=2)}
        
        Output only a JSON array of skill objects.
        """
        response = self.llm.invoke(prompt)
        return json.loads(response)