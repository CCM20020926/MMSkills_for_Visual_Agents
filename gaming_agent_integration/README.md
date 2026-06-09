# GamingAgent Integration

This folder contains the minimal MMSkills agent implementation for
LMGame-Bench/GamingAgent.

## Included Files

- `gamingagent/agents/skills_agent.py`: `BaseAgent`-compatible branch-loaded
  multimodal skill agent for 2D games.

The public entrypoint in GamingAgent is:

```bash
--agent_type skills
```

## Install Into GamingAgent

Run from this repository root:

```bash
cp gaming_agent_integration/gamingagent/agents/skills_agent.py \
  /path/to/GamingAgent/gamingagent/agents/skills_agent.py
```

The local GamingAgent runner already imports `SkillsAgent` in
`lmgame-bench/single_agent_runner.py`. If your checkout does not, add:

```python
from gamingagent.agents.skills_agent import SkillsAgent
```

and instantiate `SkillsAgent` when `--agent_type skills`.

## Run

```bash
export LMGAME_SKILLS_ROOT=/path/to/lmgame_or_mario_mmskills

python lmgame-bench/single_agent_runner.py \
  --agent_type skills \
  --game_name super_mario_bros \
  --model_name gpt-4o \
  --skills_root "$LMGAME_SKILLS_ROOT" \
  --skill_cooldown_steps 5
```

`SkillsAgent` filters loaded skill packages by the current `game_name` domain.
Use `--skill_names` for comma-separated exact skills when you want to expose a
small fixed subset.
