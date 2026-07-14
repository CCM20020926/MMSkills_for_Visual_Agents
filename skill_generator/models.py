from pydantic import BaseModel
from typing import List, Optional, Literal

# ========== 轨迹数据模型（原 dataclass） ==========
class TrajectoryStep(BaseModel):
    image: str
    observation: str
    action: str
    code: str
    correct: bool
    reflection: Optional[str] = ""

class Trajectory(BaseModel):
    task_id: str
    instruction: str
    domain: str
    task_completed: bool
    task_difficulty: int
    alignment_score: int
    efficiency_score: int
    steps: List[TrajectoryStep]

# ========== plan.json 模型 ==========
class HighlightTarget(BaseModel):
    name: str
    target_type: Literal["action_target", "state_signal"]
    annotation_query: str
    color: Literal["red", "green"]

class KeyFrame(BaseModel):
    image_filename: str
    highlight_targets: List[HighlightTarget]

class State(BaseModel):
    state_id: int
    state_name: str
    visual_grounding: str
    trigger_condition: str
    action: str
    is_result_state: bool
    has_image: bool = True
    text_description: str
    key_frame: KeyFrame

class Procedure(BaseModel):
    procedure_id: int
    procedure_name: str
    when_to_use: List[str]
    derived_from_source_skills: List[str]
    states: List[State]

class AtomicCapability(BaseModel):
    name: str
    purpose: str
    derived_from_source_skills: List[str]

class DecisionGuide(BaseModel):
    condition: str
    choose_capability: str
    reason: str

class SkillPlan(BaseModel):
    overview: str
    when_to_use: List[str]
    preconditions: List[str]
    atomic_capabilities: List[AtomicCapability]
    decision_guide: List[DecisionGuide]
    procedures: List[Procedure]
    common_failure_modes: List[str]
    skill_slug: str
    skill_name: str

# ========== runtime_state_cards.json 模型 ==========
class AvailableView(BaseModel):
    view_type: Literal["full_frame", "focus_crop", "before", "after"]
    image_path: str
    use_for: str
    label: str

class RuntimeState(BaseModel):
    state_id: str
    state_name: str
    stage: Literal["entry_state", "operation_state", "expected_after_action"]
    image_role: Literal["state_cue", "expected_after_action"]
    when_to_use: str
    when_not_to_use: str
    visible_cues: List[str]
    verification_cue: str
    visual_evidence_chain: dict
    visual_risk: str
    preferred_view_order: List[str]
    available_views: List[AvailableView]

class RuntimeStateCards(BaseModel):
    schema_version: str = "2026-04-17.runtime_state_bundles.v4"
    skill_slug: str
    domain: str
    card_granularity: str = "one_state_many_views"
    states: List[RuntimeState]

# ========== Skill Planner 相关模型 ==========
class AtomicSkill(BaseModel):
    skill_name: str
    description: str
    workflow_boundary: str          # 宏观边界描述
    completion_criteria: str
    covered_task_ids: List[str]     # 覆盖的任务ID列表
    
class GeneralSkill(BaseModel):
    skill_name: str
    description: str
    workflow_boundary: str          # 宏观边界描述
    completion_criteria: str
    covered_task_ids: List[str]     # 覆盖的任务ID列表
    common_failure_mode: str

class GeneralSkillList(BaseModel):
    skills: List[GeneralSkill]

class FailurePattern(BaseModel):
    failure_description: str
    typical_state_before_failure: str
    action_that_caused_failure: str
    suggested_prevention: str

class ClusterPlan(BaseModel):
    skills: List[AtomicSkill]
    failure_patterns: List[FailurePattern]