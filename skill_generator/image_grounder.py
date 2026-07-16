from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw
from models import Plan, RuntimeStateCards, RuntimeState, AvailableView
from models import TrajectoryStep
from util import get_logger
import json
from typing import cast


logger = get_logger("ImageGrounder")

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
                    logger.warning(f"Not enough steps in success_segment to ground all states. Stopping at step index {step_idx}.")
                    return plan
                    
                step = success_segment[step_idx]
                screenshot_path = step.image
                img = Image.open(screenshot_path).convert("RGB")
                draw = ImageDraw.Draw(img)

                for target in state.key_frame.highlight_targets:
                    bbox = self.detector.detect(screenshot_path, target.annotation_query)
                    if bbox is not None:
                        x1,y1,x2,y2 = bbox
                        draw.rectangle([x1,y1,x2,y2], outline=target.color, width=3)
                        # draw.text((x1, max(0,y1-10)), target.name, fill=target.color)

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

    def generate_runtime_cards(
        self, 
        llm,
        plan: Plan,
        domain: str,
        success_segments_ctx: List[Tuple[str, List[TrajectoryStep]]],
        failure_steps_ctx: List[Tuple[str, TrajectoryStep]],
        example_cards: List[Dict],
    ) -> RuntimeStateCards:
         # 构造示例文本（直接使用 dict 的 JSON 字符串）
        examples_text = ""
        for idx, cards_dict in enumerate(example_cards):
            cards_json = json.dumps(cards_dict, indent=2, ensure_ascii=False)
            examples_text += f"\n--- Example {idx+1} ---\nRuntimeStateCards:\n{cards_json}\n"

        current_plan_json = json.dumps(plan.model_dump(), indent=2, ensure_ascii=False)
        
        success_context = ""
        if success_segments_ctx:
            for instr, seg in success_segments_ctx:
                steps_text = "\n".join([
                    f"  Step {i+1}: Observation: {s.observation} | Action: {s.action} | Reflection: {s.reflection or 'N/A'}"
                    for i, s in enumerate(seg[:8])
                ])
                success_context += f"Instruction: {instr}\nSteps:\n{steps_text}\n"
                
        failure_context = ""
        if failure_steps_ctx:
            for instr, step in failure_steps_ctx:
                failure_context += f"Instruction: {instr} | Observation: {step.observation} | Action: {step.action} | Reflection: {step.reflection or 'N/A'}\n"
                
        
        prompt = f"""
You are an expert in creating runtime_state_cards for AI agents.
Generate a JSON object following the RuntimeStateCards schema based on the given plan and execution traces.

**Examples:**
{examples_text}

**Current Plan:**
{current_plan_json}

**Domain:** {domain}

**Execution Context:**
Successful segments (with task instructions):
{success_context}

Failed steps (with task instructions):
{failure_context}

**Instructions:**
- Output pure JSON only, no explanations.
- Use the same structure and detail level as examples.
- `when_to_use` from successful observations, and reference the task instruction if helpful.
- `when_not_to_use` from failed steps.
- `visible_cues` from visual features in observations.
- `verification_cue` from completion patterns.
- Include all states from the plan.

Only JSON object.
"""
        structured_llm = llm.with_structured_output(RuntimeStateCards)
        cards = cast(RuntimeStateCards, structured_llm.invoke(prompt))
        
        return cards
        
