import json

class TextDraftGenerator:
    def __init__(self, llm_client, meta_guide):
        self.llm = llm_client
        self.guide = meta_guide
    
    def generate_draft(self, merged_skill):
        prompt = f"""
        You are drafting a multimodal skill package. Given the skill specification below, produce:
        - "descriptor": short name and description (max 50 words)
        - "procedure": step-by-step textual instructions (each step should be actionable)
        - "state_cards": an array of objects, each with fields:
            "when_to_use": text
            "when_not_to_use": text
            "visible_cues": text (what to look for)
            "verification_cue": text (how to confirm progress or completion)
            "available_views": list of views (options: full_frame, focus_crop, before, after)
        
        Specification:
        {json.dumps(merged_skill, indent=2)}
        
        Output only a JSON object with the above keys.
        """
        return json.loads(self.llm.invoke(prompt))