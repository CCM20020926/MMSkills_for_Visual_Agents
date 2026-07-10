import yaml
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from dataloader import AgentNetLoader
from task_cluster import Clusterer
from skill_planner import SkillPlanner
from skill_merger import SkillMerger
from text_drafter import TextDrafter
from image_grounder import GroundingDINODetector, ImageGrounder
from auditor import Auditor

class Pipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.llm = ChatOpenAI(model=self.config['llm_model'], temperature=0.2)
        self.loader = AgentNetLoader(self.config['data_path'], self.config['image_root'])
        self.clusterer = Clusterer(n_clusters=self.config['n_clusters'])
        self.planner = SkillPlanner(self.llm)
        self.merger = SkillMerger(self.llm, self.config)
        self.drafter = TextDrafter(self.llm)
        self.detector = GroundingDINODetector(model_name=self.config['grounding_dino_model'])
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.auditor = Auditor(str(self.output_dir))

    def run(self):
        # 1. 加载数据（按 domain 聚合）
        trajs_data = self.loader.load()
        all_trajs = [traj for traj in trajs_data.values()]

        # 2. 收集全局失败步骤（用于负向约束）
        failure_steps_with_context = []   # 存储所有与失败相关的步骤（含上下文）
        seen = set()
        for traj in all_trajs:
            contexts = AgentNetLoader.extract_failure_contexts(traj, window_size=1)
            for ctx in contexts:
                for step in ctx['context']:
                    key = (step.image, step.action)   # 简单去重依据
                    if key not in seen:
                        seen.add(key)
                        failure_steps_with_context.append(step)

        # 3. 对每个 domain 分别聚类、规划  <--- 核心改动
        domain_plans = {}  # domain -> list of plans
        for domain, domain_trajs in trajs_data.items():
            if len(domain_trajs) < 2:
                print(f"Skipping domain {domain}: only {len(domain_trajs)} trajectories")
                continue

            clusters = self.clusterer.fit_predict(domain_trajs)
            plans_for_domain = []
            for cluster in clusters.values():
                if len(cluster) < 2:
                    continue
                cluster_plan = self.planner.plan_cluster(cluster)
                plans_for_domain.append({
                    'skills': [s.model_dump() for s in cluster_plan.skills],
                    'failure_patterns': [f.model_dump() for f in cluster_plan.failure_patterns]
                })
            domain_plans[domain] = plans_for_domain
            
        # 4. 合并所有技能（内部按 domain 分组）
        merged_skills = []
        for domain, plans in domain_plans.items():
            if not plans:
                continue
            domain_merged = self.merger.merge(plans)   # 内部不再分组
            # 为每个技能添加 domain
            for skill in domain_merged:
                skill['domain'] = domain
            merged_skills.extend(domain_merged)
        
        # 5. 生成最终技能包
        for skill in merged_skills:
            rep_traj = self._select_representative_trajectory(skill, all_trajs)
            if rep_traj is None:
                continue
                
            domain = rep_traj.domain
            
            success_segments, _ = AgentNetLoader.split_by_success_with_id(rep_traj)
            if not success_segments:
                continue
            best_segment = max(success_segments, key=lambda x: len(x[1]))[1]

            plan = self.drafter.draft_plan(skill, domain)

            grounder = ImageGrounder(self.detector, str(self.output_dir), failure_steps_with_context)
            plan = grounder.ground_plan(plan, best_segment)

            # [CHANGED] 收集该技能相关的失败步骤（同样使用上下文）
            skill_failures = []
            seen_skill = set()
            covered_ids = skill.get('covered_task_ids', [])
            for traj in all_trajs:
                if traj.task_id in covered_ids:
                    contexts = AgentNetLoader.extract_failure_contexts(traj, window_size=1)
                    for ctx in contexts:
                        for step in ctx['context']:
                            key = (step.image, step.action)
                            if key not in seen_skill:
                                seen_skill.add(key)
                                skill_failures.append(step)

            cards = grounder.generate_runtime_cards(plan, domain, skill_failures)

            if self.auditor.audit(plan, cards):
                skill_dir = self.output_dir / plan.skill_slug
                skill_dir.mkdir(exist_ok=True)
                src_images = self.output_dir / "Images"
                if src_images.exists():
                    src_images.rename(skill_dir / "Images")
                with open(skill_dir / "plan.json", 'w') as f:
                    json.dump(plan.dict(), f, indent=2)
                with open(skill_dir / "runtime_state_cards.json", 'w') as f:
                    json.dump(cards.dict(), f, indent=2)
                markdown = self.drafter.draft_markdown(plan)
                with open(skill_dir / "SKILL.md", 'w') as f:
                    f.write(markdown)
                print(f"Generated skill: {plan.skill_slug} (domain: {domain})")
            else:
                print(f"Audit failed for {plan.skill_slug}, skipping.")
        

    def _select_representative_trajectory(self, skill, all_trajs):
        covered_ids = skill.get('covered_task_ids', [])
        for traj in all_trajs:
            if traj.task_id in covered_ids:
                return traj
        return None
    

if __name__ == "__main__":
    pipeline = Pipeline(config_path="config.yaml")
    pipeline.run()
