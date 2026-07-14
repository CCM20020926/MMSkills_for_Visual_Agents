import yaml
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from dataloader import AgentNetLoader
from task_cluster import Clusterer
from skill_planner import SkillPlanner
from skill_merger import SkillMerger
from skill_drafter import TextDrafter
from image_grounder import GroundingDINODetector, ImageGrounder
from auditor import Auditor
from models import *


def pipeline(domain, trajs: list[Trajectory], config: dict):
    # 1. 聚类轨迹
    llm = ChatOpenAI(model_name=config.get('llm_model', 'gpt-4o'), temperature=0)
    
    cluster = Clusterer(n_clusters=config.get('n_clusters', 20))
    cluster = clusters = cluster.fit_predict(trajs)
    
    # 2. 规划技能
    planner = SkillPlanner(llm)
    
    plans = []
    
    for cluster_trajs in clusters.values():
        plan: ClusterPlan = planner.plan_cluster(cluster_trajs)
        plans.append(plan)
    
    # 3. 合并技能
    merger = SkillMerger(llm, config)
    skills = merger.merge(plans)
    
    # 4. 草拟技能文档
    drafter = TextDrafter(llm)
    
    for skill in skills:
        repr_trajs = select_representative_trajectory(skill, trajs)
        plan = drafter.draft_plan(skill, domain, repr_trajs)
        drafter.draft_markdown(plan)
    
    # 5. 图像标注
    image_grounder = ImageGrounder()
    
    success_segments = extract_success_segments(trajs)
    
    image_grounder.ground_plan(skills, success_segments)
    
    image_grounder.generate_runtime_cards()
    
    # 6. 审核
    
    # 7. 返回
    ...
    


def main():
        # 0. 加载数据（按 domain 聚合）
    loader = AgentNetLoader()
    trajs_data: dict[str, list[Trajectory]] = loader.load()
        
 
def extract_success_segments(trajs: list[Trajectory]):
    ...

def select_representative_trajectory(self, skill, all_trajs, k=10):
    covered_ids = skill.covered_task_ids
    candidates = [traj for traj in all_trajs if traj.task_id in covered_ids]
    if not candidates:
        return None

    def score(traj: Trajectory):
        # 1. 完成度得分（40%）
        completion_score = 1.0 if traj.task_completed else 0.3

        # 2. 对齐分数（30%），范围 0-100
        align = traj.alignment_score if traj.alignment_score is not None else 50
        align_score = max(0.0, min(1.0, align / 100.0))

        # 3. 效率分数（20%），范围 0-100
        eff = traj.efficiency_score if traj.efficiency_score is not None else 50
        eff_score = max(0.0, min(1.0, eff / 100.0))

        # 4. 步骤长度得分（10%）
        # < 3 代表性差
        # >=3, <5 代表性一般
        # >=5, <=20 代表性好
        # >20, <=30 代表性中等
        # >30 代表性差
        step_count = len(traj.steps)
        if step_count < 3:
            length_score = 0.1
            
        elif 3 <= step_count < 5:
            length_score = 0.6
        
        elif 5 <= step_count <= 20:
            length_score = 1.0
        
        elif 20 < step_count <= 30:
            length_score = 0.7
        
        else:
            length_score = 0.4

        # 5. 任务难度微调（权重约为 ±1.5%，仅作为加分/减分项）
        # [改进] 若难度数据缺失，则不进行任何调整（factor = 1.0）
        if traj.task_difficulty is not None:
            diff = traj.task_difficulty
            
            difficulty_score = 1.0 - 0.15 * abs(diff - 3)
            
            difficulty_score = max(0.7, min(1.0, difficulty_score))
            # 微调因子：范围 0.985 ~ 1.0，幅度极小
            factor = 0.95 + 0.05 * difficulty_score
        else:
            # 数据缺失：保持中性，不倾斜
            factor = 1.0

        # 加权综合得分
        total = (0.40 * completion_score +
                    0.30 * align_score +
                    0.20 * eff_score +
                    0.10 * length_score)

        # 应用难度微调
        total *= factor
        return total

    results = sorted(candidates, key=score, reverse=True)
    return results[:k]


if __name__ == "__main__":
    main()