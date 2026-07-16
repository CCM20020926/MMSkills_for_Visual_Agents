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
from tqdm import tqdm
from util import get_logger
import os
import json
from concurrent.futures import ThreadPoolExecutor

NAME = "skill_generator"

def save_cache_json(data, name, logger):
    logger.info(f"Save cache `{name}.json`.")
    with open(f"cache/{name}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
def pipeline(domain, trajs: list[Trajectory], config: dict):
    # 1. 聚类轨迹
    logger = get_logger(NAME)

    logger.info(f"Start {domain} clustering.")
    llm = ChatOpenAI(model_name=config.get('llm_model', 'gpt-4o'), temperature=0)
    
    cluster = Clusterer(n_clusters=config.get('n_clusters', 20))
    clusters = cluster.fit_predict(trajs)
    
    logger.info(f"Total {len(clusters)} clusters.")
    
    save_cache_json(clusters, f"{domain}_clusters.json", logger)
    
    # 2. 规划技能
    logger.info(f"Start {domain} planning atomic skills.")
    planner = SkillPlanner(llm)
    
    plans = []
    
    for cluster_trajs in tqdm(clusters.values(), desc=f"{domain} - plan"):
        plan: ClusterPlan = planner.plan_cluster(cluster_trajs)
        plans.append(plan)
    
    logger.info(f"Total {len(plans)} atomic skills.")
    
    cache_plans = [plan.model_dump() for plan in plans]
    save_cache_json(cache_plans, f"{domain}_plans.json", logger)


    # 3. 合并技能
    logger.info(f"Start {domain} merging atomic skills.")
    merger = SkillMerger(llm, config)
    skills = merger.merge(plans)
    
    logger.info(f"Total {len(skills)} after merging.")
    
    cache_skills = [skill.model_dump() for skill in skills]
    save_cache_json(cache_skills, f"{domain}_skills.json", logger)
    
    # 4. 草拟技能文档
    drafter = TextDrafter(llm)
    
    detector = GroundingDINODetector(device=config.get('device', 'cuda'))
    
    pass_audit = {}
    
    for skill in skills:
        logger.info(f"Drafting skill {domain} - {skill.skill_name}.")
        repr_trajs = select_representative_trajectory(skill, trajs)
        logger.debug(f"Selected representative trajectories for {domain} - {skill.skill_name} {repr_trajs}.")
        
        plan = drafter.draft_plan(skill, domain, repr_trajs, ...)  # 加入少样本示例
            
        skill_dir = os.path.join("skill_library", domain, f"{domain.upper()}_{skill.skill_name}")
            
        # 5. 图像标注
        logger.info(f"Start {domain} - {skill.skill_name} grounding images.")
        image_grounder = ImageGrounder(output_dir=skill_dir, detector=detector)
                
        success_segments = []
        skill_failure_steps = []
        
        for traj in repr_trajs:
            success_steps, failure_steps = AgentNetLoader.split_by_success_with_id(traj)
            
            success_segments.extend(success_steps)
            skill_failure_steps.extend(failure_steps)
            
        # 从所有片段中选最长的一条作为接地源
        if success_segments:
            # 按步骤长度降序排序，取最长片段
            best_segment = max(success_segments, key=lambda x: len(x[1]))[1]   # 只取 steps 列表
        
        else:
            
            logger.warning(f"No success segments found for skill {skill.skill_name}.")
            
            best_segment = []
        
        updated_plan = image_grounder.ground_plan(skills, best_segment)
        
        with open(f"{skill_dir}/plan.json", "w", encoding="utf-8") as f:
            json.dump(updated_plan.model_dump(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Save plan.json for {domain} - {skill.skill_name}.")
        
        cards = image_grounder.generate_runtime_cards(llm, updated_plan, domain, success_segments, skill_failure_steps, ...)  # 选取代表性 runtime_state_cards
        
        with open(f"{skill_dir}/runtime_state_cards.json", "w", encoding="utf-8") as f:
            json.dump(cards.model_dump(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Save runtime_state_cards.json for {domain} - {skill.skill_name}.")
        
        text = drafter.draft_markdown(updated_plan, cards, ...)   # 选取代表性 SKILL.md
        
        with open(f"{skill_dir}/SKILL.md", "w", encoding="utf-8") as f:
            f.write(text)
        
        logger.info(f"Save SKILL.md for {domain} - {skill.skill_name}.")

        # 6. 审核
        logger.info(f"Start {domain} - {skill.skill_name} auditing.")
        
        auditor = Auditor(skill_dir)
        
        result = auditor.audit(plan, cards)
        
        pass_audit[skill.skill_name] = result
        
        logger.info(f"Audit result for {domain} - {skill.skill_name}: {result}.")
        
        
    return {"domain": pass_audit}
    
   
def main():
        # 0. 加载数据（按 domain 聚合）
    loader = AgentNetLoader()
    trajs_data: dict[str, list[Trajectory]] = loader.load()
    
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    all_audit_results = {}
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for domain, trajs in trajs_data.items():
            futures.append(executor.submit(pipeline, domain, trajs, config))
        
        for future in futures:
            result = future.result()
            all_audit_results.update(result)
    
    with open("audit_results.json", "w", encoding="utf-8") as f:
        json.dump(all_audit_results, f, ensure_ascii=False, indent=4)
        
 
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
        # 5. 任务难度微调（factor 范围 0.85~1.0，基于 diff 自然值）
        if traj.task_difficulty is not None:
            diff = traj.task_difficulty
            best = 6
            max_dev = 4  # max(10-6, 6-4) = 4
            norm_dev = abs(diff - best) / max_dev
            factor = 1.0 - 0.15 * norm_dev
        else:
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
    os.makedirs("cache", exist_ok=True)
    main()