from pathlib import Path
import json
from typing import List, Tuple,  Dict
from models import Trajectory, TrajectoryStep

class AgentNetLoader:
    def __init__(self, data_path: str, image_root: str):
        self.data_path = Path(data_path)
        self.image_root = Path(image_root)

    def load(self) -> Dict[str, List[Trajectory]]:
        """
        加载数据并按 domain 聚合
        返回: {domain: [Trajectory, ...]}
        """
        trajs_data = {}
        
        with open(self.data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                steps = []
                for step_data in item.get('traj', []):
                    value = step_data.get('value', {})
                    step = TrajectoryStep(
                        image=str(self.image_root / step_data.get('image', '')),
                        observation=value.get('observation', ''),
                        action=value.get('action', ''),
                        code=value.get('code', ''),
                        correct=value.get('last_step_correct'),  # 映射字段
                        reflection=value.get('reflection', '')
                    )
                    steps.append(step)
                    
                traj = Trajectory(
                    task_id=item.get('task_id', ''),
                    instruction=item.get('instruction', ''),
                    domain=item.get('domain', 'unknown'),
                    task_completed=item.get('task_completed'),
                    task_difficulty=item.get('task_difficulty'),
                    alignment_score=item.get('alignment_score'),
                    efficiency_score=item.get('efficiency_score'),
                    steps=steps
                )
                
                domain = traj.domain
                if domain not in trajs_data:
                    trajs_data[domain] = []
                trajs_data[domain].append(traj)
                
        return trajs_data


    @staticmethod
    def split_by_success_with_id(
        traj: Trajectory,
        min_segment_len: int = 2
    ) -> Tuple[List[Tuple[str, List[TrajectoryStep]]], List[Tuple[str, TrajectoryStep]]]:
        """
        将轨迹拆分为成功步骤片段和失败步骤。
        返回: ( (task_id, 成功步骤列表)列表, (task_id, 失败步骤)列表 )
        """
        success_segments = []
        current_segment = []
        failed_steps = []
        for step in traj.steps:
            if step.correct:
                current_segment.append(step)
            else:
                if len(current_segment) >= min_segment_len:
                    success_segments.append((traj.task_id, current_segment))
                current_segment = []
                failed_steps.append((traj.task_id, step))
        if len(current_segment) >= min_segment_len:
            success_segments.append((traj.task_id, current_segment))
        return success_segments, failed_steps
    
    @staticmethod
    def extract_failure_contexts(
        traj: Trajectory,
        window_size: int = 1
    ) -> List[Dict]:
        """
        提取轨迹中所有失败步骤及其上下文窗口。
        返回: list of dict，每个包含 task_id, failed_step, context_steps, step_index
        """
        contexts = []
        for i, step in enumerate(traj.steps):
            if not step.correct:
                start = max(0, i - window_size)
                end = min(len(traj.steps), i + window_size + 1)
                context_steps = traj.steps[start:end]
                contexts.append({
                    'task_id': traj.task_id,
                    'failed_step': step,
                    'context': context_steps,
                    'step_index': i
                })
        return contexts
