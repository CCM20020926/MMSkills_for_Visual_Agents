# VAB-Minecraft Integration

This folder contains the MMSkills adapters for VisualAgentBench
VAB-Minecraft.

Reference project: <https://github.com/THUDM/VisualAgentBench>.

VAB-Minecraft uses the AgentBench-style flow from VisualAgentBench: start
Minecraft task workers with `src.start_task`, then launch evaluation with
`src.assigner`.

## Clone VisualAgentBench

```bash
git clone https://github.com/THUDM/VisualAgentBench.git
cd VisualAgentBench
```

Follow the official VAB-Minecraft setup first, including Docker, the Minecraft
task image, and Steve-1 weights under `data/minecraft`.

## Included Files

- `mmskills_http_agent.py`: `AgentClient`/`HTTPAgent`-compatible MMSkills
  adapter for the original THUDM checkout. It keeps the normal VAB HTTP request
  path and adds an internal text planner branch when the model emits
  `LOAD_SKILL("<skill_name>")`.
- `openai-chat-mmskills.yaml`: example VAB HTTP-agent config.
- `minecraft_mmskills_assignment.yaml`: self-contained Minecraft assignment
  config that registers `mmskills-gpt-4o` without editing VAB's shared
  `configs/agents/api_agents.yaml`.
- `gemini_minecraft_skills_agent.py`: Gemini-specific VAB-Minecraft MMSkills
  adapter based on the `GeminiAgent` host-side branch design. It supports
  task-skill assignment, `LOAD_SKILL`, gated `LOAD_STATE_VIEWS`, planner JSON,
  active planner memo, and skill logs.
- `minecraft_skill_loader.py`: filesystem loader for VAB-Minecraft MMSkill
  packages and runtime state-card image views.
- `gemini-minecraft-mmskills.yaml`: example Gemini agent config for VAB
  checkouts that already provide `gemini_agent.py`.
- `minecraft_gemini_mmskills_assignment.yaml`: self-contained assignment for
  the Gemini MMSkills agent.

The HTTP adapter is intentionally lightweight. VAB's HTTP prompter usually
sends one current image, so it uses `SKILL.md` plus runtime state cues and does
not attach additional skill reference images to branch calls. The Gemini
adapter is the multimodal path: Stage 1 can select `before`, `full_frame`,
`focus_crop`, or `after` state views, and Stage 2 returns planner guidance
before the final Minecraft action.

## Install Into Original THUDM VisualAgentBench

Use this path when your checkout has `src/client/agents/http_agent.py` and no
`gemini_agent.py`. Run these commands from the MMSkills repository root:

```bash
cp vab_minecraft_integration/mmskills_http_agent.py \
  /path/to/VisualAgentBench/src/client/agents/mmskills_http_agent.py

cp vab_minecraft_integration/openai-chat-mmskills.yaml \
  /path/to/VisualAgentBench/configs/agents/openai-chat-mmskills.yaml

cp vab_minecraft_integration/minecraft_mmskills_assignment.yaml \
  /path/to/VisualAgentBench/configs/assignments/minecraft_mmskills.yaml
```

Then edit `configs/agents/openai-chat-mmskills.yaml` with your API key or use
your existing gateway URL and headers. Edit
`configs/assignments/minecraft_mmskills.yaml` if you need a different model
name, `max_tokens`, concurrency, or output folder.

## Install Into A Gemini VAB Checkout

Use this path when your VisualAgentBench variant has a `GeminiAgent` class next
to the other agent files, usually as `src/client/agents/gemini_agent.py` or
`src/agents/gemini_agent.py`.

For the standard THUDM package layout, run these commands from the MMSkills
repository root:

```bash
cp vab_minecraft_integration/gemini_minecraft_skills_agent.py \
  /path/to/VisualAgentBench/src/client/agents/gemini_minecraft_skills_agent.py

cp vab_minecraft_integration/minecraft_skill_loader.py \
  /path/to/VisualAgentBench/src/client/agents/minecraft_skill_loader.py

cp vab_minecraft_integration/gemini-minecraft-mmskills.yaml \
  /path/to/VisualAgentBench/configs/agents/gemini-minecraft-mmskills.yaml

cp vab_minecraft_integration/minecraft_gemini_mmskills_assignment.yaml \
  /path/to/VisualAgentBench/configs/assignments/minecraft_gemini_mmskills.yaml
```

If your checkout uses `src/agents` instead of `src/client/agents`, copy the two
Python files there and update `configs/agents/gemini-minecraft-mmskills.yaml`:

```yaml
module: src.agents.gemini_minecraft_skills_agent.GeminiMinecraftSkillsAgent
```

The Gemini adapter imports `.gemini_agent.GeminiAgent`; it does not bundle that
base class because the Gemini transport differs across VAB forks.

## Configure Skills

Point the adapter at a local VAB-Minecraft MMSkills package root:

```bash
export VAB_MMSKILLS_SKILLS_ROOT=/path/to/vab_minecraft_mmskills
```

For the Gemini adapter, optionally provide a task-to-skill mapping JSON:

```bash
export VAB_MMSKILLS_TASK_SKILL_MAPPING=/path/to/vab_minecraft_task_skill_mapping_top5.json
```

If no mapping exists, the Gemini adapter falls back to available skill metadata
and the configured fallback skills. For the HTTP adapter, optionally expose a
fixed subset:

```bash
export VAB_MMSKILLS_SKILL_NAMES=skill_a,skill_b
```

## Run

Terminal 1 starts the Minecraft task workers and controller:

```bash
python -m src.start_task --config configs/start_task.yaml --auto-controller
```

Terminal 2 launches evaluation with the MMSkills assignment:

```bash
python -m src.assigner --auto-retry --config configs/assignments/minecraft_mmskills.yaml
```

For the Gemini adapter, use:

```bash
python -m src.assigner --auto-retry --config configs/assignments/minecraft_gemini_mmskills.yaml
```

The HTTP assignment copied above is equivalent to:

```yaml
import: definition.yaml

definition:
  agent:
    mmskills-gpt-4o:
      import: ../agents/openai-chat-mmskills.yaml
      parameters:
        name: "mmskills-gpt-4o"
        body:
          model: "gpt-4o-2024-05-13"
          max_tokens: 512

concurrency:
  task:
    minecraft: 4
  agent:
    mmskills-gpt-4o: 4

assignments:
  - agent:
      - mmskills-gpt-4o
    task:
      - minecraft

output: "outputs/minecraft_mmskills"
```

Keep the rest of the VAB-Minecraft setup unchanged. The adapter returns the
benchmark's normal `OBSERVATION`/`THOUGHT`/`ACTION` response after any internal
MMSkills branch consultation.
