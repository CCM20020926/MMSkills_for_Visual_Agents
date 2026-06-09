# GamingAgent Integration

This folder contains the minimal MMSkills agent implementation for
LMGame-Bench/GamingAgent.

Reference project: <https://github.com/lmgame-org/GamingAgent>.

The agent implementation is synced from the local GamingAgent source:
`/Users/zhangkangning/code_repos/GamingAgent/gamingagent/agents/skills_agent.py`.
It is intentionally kept as a thin benchmark adapter around MMSkills packages,
not as a fork of the full GamingAgent runner.

## Included Files

- `gamingagent/agents/skills_agent.py`: `BaseAgent`-compatible branch-loaded
  multimodal skill agent for 2D games.
- `gamingagent/agents/__init__.py`: optional package export update for
  `SkillsAgent`.
- `patches/register_skills_agent.patch`: minimal patch for a vanilla
  GamingAgent checkout. It registers `--agent_type skills` and passes
  MMSkills-specific CLI arguments to `SkillsAgent`.

The public entrypoint in GamingAgent is:

```bash
--agent_type skills
```

## Clone GamingAgent

```bash
git clone https://github.com/lmgame-org/GamingAgent.git
cd GamingAgent
```

Follow GamingAgent's environment setup first. The MMSkills integration only
adds a `BaseAgent`-compatible agent class and a small runner registration patch.

## Install Into GamingAgent

Run from this repository root:

```bash
cp gaming_agent_integration/gamingagent/agents/skills_agent.py \
  /path/to/GamingAgent/gamingagent/agents/skills_agent.py
```

If the target checkout does not already support `--agent_type skills`, apply
the registration patch from inside the GamingAgent checkout. The patch updates
both `gamingagent/agents/__init__.py` and
`lmgame-bench/single_agent_runner.py`:

```bash
git apply /path/to/MMSkills_for_Visual_Agents/gaming_agent_integration/patches/register_skills_agent.patch
```

If your checkout already has the runner hooks but not the package export, copy
only the export file:

```bash
cp /path/to/MMSkills_for_Visual_Agents/gaming_agent_integration/gamingagent/agents/__init__.py \
  gamingagent/agents/__init__.py
```

Confirm the runner exposes the skills arguments:

```bash
python lmgame-bench/single_agent_runner.py --help | grep -E "agent_type|skills_root|skill_cooldown"
```

The local source checkout at `/Users/zhangkangning/code_repos/GamingAgent`
already has these runner hooks.

## Adapter Behavior

`SkillsAgent` subclasses the non-harness `BaseAgent` path and keeps
GamingAgent's evaluated action interface unchanged. At each decision step:

1. The main call sees compact MMSkill hints and may emit
   `LOAD_SKILL("<exact_skill_name>")`.
2. Stage 1 decides whether skill reference images are useful and can request
   `before`, `full_frame`, and `after` views from runtime state cards.
3. Stage 2 returns a planner JSON memo with fields such as `state_pattern`,
   `subgoal`, `control_intent`, `risk_profile`, `action_constraints`,
   `state_checks`, `do_not_do`, `fallback_if_no_progress`, and
   `expected_state`.
4. The final call emits the actual game move. `action_constraints` is treated
   as guidance over legal action names, not as a scripted action sequence.

Skills are filtered by the current `game_name` domain. By default, the agent
uses `--skills_root`, `LMGAME_SKILLS_ROOT`, or the latest
`runs/lmgame_multimodal_skills/*/phase4/skills` package root.

## Run

```bash
export LMGAME_SKILLS_ROOT=/path/to/lmgame_or_mario_mmskills

python lmgame-bench/single_agent_runner.py \
  --agent_type skills \
  --no_harness \
  --observation_mode vision \
  --game_name super_mario_bros \
  --model_name gpt-4o \
  --skills_root "$LMGAME_SKILLS_ROOT" \
  --skill_cooldown_steps 5 \
  --max_stage1_selected_states 2 \
  --max_stage1_selected_views 4
```

Use `--skill_names` for comma-separated exact skills when you want to expose a
small fixed subset.
