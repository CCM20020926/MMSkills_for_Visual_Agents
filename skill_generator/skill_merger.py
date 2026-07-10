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
        假定所有 plans 都属于同一个 domain。
        返回: 合并后的技能列表，每个技能包含 common_failure_modes
        """
        if not plans:
            return []

        # 收集所有候选技能和失败模式
        all_skill_candidates = []
        all_failure_patterns = []
        for plan in plans:
            all_skill_candidates.extend(plan.get('skills', []))
            all_failure_patterns.extend(plan.get('failure_patterns', []))

        if not all_skill_candidates:
            return []

        # 合并技能（去重+泛化）
        merged_skills = self._merge_skills(all_skill_candidates)

        # 泛化失败模式
        failure_modes = self._generalize_failures(all_failure_patterns)

        # 为每个技能添加失败模式
        for skill in merged_skills:
            skill['common_failure_modes'] = failure_modes

        return merged_skills

    def _merge_skills(self, candidates):
        """使用 LLM 合并候选技能"""
        if not candidates:
            return []
        prompt = ChatPromptTemplate.from_template("""
You are a skill merger. Merge the following skill candidates into a set of unique, generalized skills.
Each skill should have: skill_name, description, workflow_boundary (a concise textual description summarizing the whole workflow), completion_criteria, covered_task_ids (union of all covered_task_ids from the candidates).
Candidates: {candidates}
Return a JSON array of merged skills.
""")
        response = self.llm.invoke(prompt.format_messages(
            candidates=json.dumps(candidates, indent=2)
        ))
        merged = json.loads(response.content) if hasattr(response, 'content') else response

        # 过滤过宽技能（覆盖任务比例过高）
        total_tasks = sum([len(s.get('covered_task_ids', [])) for s in merged])
        threshold = self.config.get('coverage_ratio_threshold', 0.6)
        filtered = [s for s in merged if len(s.get('covered_task_ids', [])) / max(1, total_tasks) <= threshold]
        return filtered

    def _generalize_failures(self, failure_patterns):
        """泛化失败模式为通用列表"""
        if not failure_patterns:
            return []
        prompt = ChatPromptTemplate.from_template("""
You are given many specific failure patterns. Generalize them into a concise list of common failure modes.
Failure patterns:
{failure_patterns}
Return a JSON list of strings.
""")
        response = self.llm.invoke(prompt.format_messages(
            failure_patterns=json.dumps(failure_patterns, indent=2)
        ))
        return json.loads(response.content) if hasattr(response, 'content') else response