from pathlib import Path
from models import Plan, RuntimeStateCards

class Auditor:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def audit(self, plan: Plan, cards: RuntimeStateCards) -> bool:
        plan_state_names = set()
        for proc in plan.procedures:
            for state in proc.states:
                plan_state_names.add(state.state_name)
        card_state_names = {s.state_name for s in cards.states}
        missing = plan_state_names - card_state_names
        if missing:
            print(f"Audit failed: missing runtime states {missing}")
            return False
        for card in cards.states:
            for view in card.available_views:
                img_path = self.output_dir / view.image_path
                if not img_path.exists():
                    print(f"Audit failed: missing image {img_path}")
                    return False
        return True