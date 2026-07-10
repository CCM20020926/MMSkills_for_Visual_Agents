from pathlib import Path
from typing import List, Tuple, Optional
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw
from models import Plan, RuntimeStateCards, RuntimeState, AvailableView
from models import TrajectoryStep

class GroundingDINODetector:
    def __init__(self, model_name="IDEA-Research/grounding-dino-tiny", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def detect(self, image_path: str, query: str, box_threshold=0.35, text_threshold=0.25) -> Optional[Tuple[int,int,int,int]]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )
        if len(results) > 0 and len(results[0]['boxes']) > 0:
            boxes = results[0]['boxes']
            if len(boxes) > 0:
                box = boxes[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                return (x1, y1, x2, y2)
        return None

class ImageGrounder:
    def __init__(self, detector, output_dir: str, failure_pool: List[TrajectoryStep] = None):
        self.detector = detector
        self.output_dir = Path(output_dir)
        self.Images_dir = self.output_dir / "Images"
        self.Images_dir.mkdir(parents=True, exist_ok=True)
        self.failure_pool = failure_pool or []

    def ground_plan(self, plan: Plan, success_segment: List[TrajectoryStep]) -> Plan:
        step_idx = 0
        for proc in plan.procedures:
            for state in proc.states:
                if step_idx >= len(success_segment):
                    step_idx = len(success_segment) - 1
                step = success_segment[step_idx]
                screenshot_path = step.image
                img = Image.open(screenshot_path).convert("RGB")
                draw = ImageDraw.Draw(img)

                for target in state.key_frame.highlight_targets:
                    bbox = self.detector.detect(screenshot_path, target.annotation_query)
                    if bbox is not None:
                        x1,y1,x2,y2 = bbox
                        draw.rectangle([x1,y1,x2,y2], outline=target.color, width=3)
                        draw.text((x1, max(0,y1-10)), target.name, fill=target.color)

                full_filename = f"{state.state_name}.png"
                full_path = self.Images_dir / full_filename
                img.save(full_path)
                state.key_frame.image_filename = full_filename

                # focus_crop
                focus_crop = None
                for target in state.key_frame.highlight_targets:
                    if target.target_type == "action_target":
                        bbox = self.detector.detect(screenshot_path, target.annotation_query)
                        if bbox:
                            x1,y1,x2,y2 = bbox
                            pad = 20
                            x1 = max(0, x1-pad)
                            y1 = max(0, y1-pad)
                            x2 = min(img.width, x2+pad)
                            y2 = min(img.height, y2+pad)
                            focus_crop = img.crop((x1,y1,x2,y2))
                            break
                if focus_crop is None:
                    w,h = img.size
                    focus_crop = img.crop((w//4, h//4, w*3//4, h*3//4))
                focus_filename = f"{state.state_name}_focus_crop.png"
                focus_crop.save(self.Images_dir / focus_filename)

                step_idx += 1
        return plan

    def generate_runtime_cards(self, plan: Plan, domain: str, skill_failure_steps: List[TrajectoryStep]) -> RuntimeStateCards:
        states = []
        for proc in plan.procedures:
            for state in proc.states:
                if state.state_id == 1:
                    stage = "entry_state"
                    image_role = "state_cue"
                elif state.is_result_state:
                    stage = "expected_after_action"
                    image_role = "expected_after_action"
                else:
                    stage = "operation_state"
                    image_role = "state_cue"

                related_failures = [s for s in skill_failure_steps if (state.state_name in s.observation) or (state.action in s.action)]
                when_not_to_use = "Do not use this card when the current UI state does not match the expected condition. "
                if related_failures:
                    fail_descs = [f"{s.reflection}" for s in related_failures if s.reflection]
                    if fail_descs:
                        when_not_to_use += " ".join(fail_descs[:2])
                    else:
                        when_not_to_use += "For example, when you observe " + "; ".join([s.observation for s in related_failures[:2] if s.observation])
                else:
                    when_not_to_use += "the necessary preconditions are not met."

                visible_cues = []
                if related_failures:
                    for s in related_failures[:2]:
                        if s.observation:
                            visible_cues.append(f"Observed: {s.observation}")
                if not visible_cues:
                    visible_cues = [state.visual_grounding]
                for target in state.key_frame.highlight_targets:
                    visible_cues.append(f"A {target.color} box marks the {target.name} as {target.target_type}.")

                full_path = f"Images/{state.key_frame.image_filename}"
                focus_path = f"Images/{state.key_frame.image_filename.replace('.png', '_focus_crop.png')}"
                available_views = [
                    AvailableView(view_type="full_frame", image_path=full_path, use_for="recognize_global_ui_state", label=state.state_name),
                    AvailableView(view_type="focus_crop", image_path=focus_path, use_for="inspect_contextual_work_region", label=f"{state.state_name}_focus")
                ]

                runtime_state = RuntimeState(
                    state_id=state.state_name,
                    state_name=state.state_name,
                    stage=stage,
                    image_role=image_role,
                    when_to_use=f"Use this card when {state.trigger_condition}.",
                    when_not_to_use=when_not_to_use,
                    visible_cues=visible_cues,
                    verification_cue=state.text_description + " and verify the expected visual outcome.",
                    visual_evidence_chain={"focus_crop": "preserves broader working region", "before": "not needed", "after": "not needed"},
                    visual_risk="Treat the example as state evidence only. Do not transfer literal coordinates, example values, or example-specific content.",
                    preferred_view_order=["full_frame", "focus_crop"],
                    available_views=available_views
                )
                states.append(runtime_state)

        cards = RuntimeStateCards(
            skill_slug=plan.skill_slug,
            domain=domain,
            states=states
        )
        return cards