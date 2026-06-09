# macOSWorld Integration

This folder contains the minimal MMSkills agent files for macOSWorld.

## Included Files

- `agent/openai_skill_v2.py`: recommended multimodal MMSkills agent.
- `agent/openai_text_skill.py`: text-only branch-planner ablation.
- `agent/openai_skill.py`: base MMSkills branch implementation required by the two agents above.
- `agent/skill_loader.py`: macOSWorld skill package loader.
- `agent/task_skill_resolver.py`: task-to-skill mapping resolver.

## Install Into macOSWorld

Run from this repository root:

```bash
cp macosworld_integration/agent/openai_skill.py /path/to/macosworld/agent/openai_skill.py
cp macosworld_integration/agent/openai_skill_v2.py /path/to/macosworld/agent/openai_skill_v2.py
cp macosworld_integration/agent/openai_text_skill.py /path/to/macosworld/agent/openai_text_skill.py
cp macosworld_integration/agent/skill_loader.py /path/to/macosworld/agent/skill_loader.py
cp macosworld_integration/agent/task_skill_resolver.py /path/to/macosworld/agent/task_skill_resolver.py
```

The macOSWorld runner must register these names in `agent/get_gui_agent.py`:

```text
openai-skill-v2-mm-branch
openai-skill-text-branch
```

The local macOSWorld checkout already uses these names in the synced source. If
your checkout does not, add a `get_gui_agent.py` branch that imports
`OpenAISkillAgentV2` or `OpenAITextSkillAgent` and passes `skills_library_dir`.

## Run

```bash
export MACOSWORLD_SKILLS_LIBRARY_DIR=/path/to/mac_mmskills

python run.py \
  --gui_agent_name openai-skill-v2-mm-branch \
  --your-other-macosworld-args
```

Use `openai-skill-text-branch` for text-only skill ablations.
