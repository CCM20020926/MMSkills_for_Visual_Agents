import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class SkillMerger:
    def __init__(self, llm: ChatOpenAI, config):
        self.llm = llm.with_structured_output(method="json_mode")
        self.config = config

    def merge(self, plans):
        """
        合并同一 domain 下的多个规划结果（ClusterPlan）。
        每个技能将携带其专属的 common_failure_modes。
        """
        if not plans:
            return []

        # 准备候选技能数据，每个技能绑定其来源簇的失败模式
        candidates_with_failures = []
        for plan in plans:
            for skill in plan.get('skills', []):
                candidates_with_failures.append({
                    "skill": skill,
                    "failure_patterns": plan.get('failure_patterns', [])
                })

        if not candidates_with_failures:
            return []

        # 调用 LLM 进行合并，输出每个技能专属的 common_failure_modes
        merged_skills = self._merge_skills(candidates_with_failures)
        return merged_skills

    def _merge_skills(self, candidates_with_failures):
        if not candidates_with_failures:
            return []
        
        prompt = ChatPromptTemplate.from_template("""
You are a skill merger. Merge the following skill candidates into a set of unique, generalized skills.
Each candidate includes a "skill" object and its associated "failure_patterns".

For each merged skill, you MUST derive a **unique** list of "common_failure_modes" 
based ONLY on the failure patterns of the skills that are merged into it.
DO NOT copy failure modes from unrelated skills.

The output skill must contain: 
skill_name, description, workflow_boundary, completion_criteria, covered_task_ids, common_failure_modes.

Candidates: {candidates}

Return a JSON array of merged skills (each with a dedicated common_failure_modes field).
""")
        response = self.llm.invoke(prompt.format_messages(
            candidates=json.dumps(candidates_with_failures, indent=2)
        ))
        merged = json.loads(response.content) if hasattr(response, 'content') else response

        # 过滤过宽技能（覆盖任务比例过高）
        total_tasks = sum([len(s.get('covered_task_ids', [])) for s in merged])
        threshold = self.config.get('coverage_ratio_threshold', 0.6)
        filtered = [s for s in merged if len(s.get('covered_task_ids', [])) / max(1, total_tasks) <= threshold]
        return filtered
