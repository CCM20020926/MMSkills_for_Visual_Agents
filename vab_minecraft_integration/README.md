# VAB-Minecraft Integration

This folder contains a minimal MMSkills adapter for VisualAgentBench
VAB-Minecraft.

Reference project: <https://github.com/THUDM/VisualAgentBench>.

VAB-Minecraft uses the AgentBench-style flow from VisualAgentBench: start
Minecraft task workers with `src.start_task`, then launch evaluation with
`src.assigner`.

## Included Files

- `mmskills_http_agent.py`: `AgentClient`/`HTTPAgent`-compatible MMSkills
  adapter. It keeps the normal VAB HTTP request path and adds an internal
  text planner branch when the model emits `LOAD_SKILL("<skill_name>")`.
- `openai-chat-mmskills.yaml`: example VAB agent config.
- `minecraft_mmskills_assignment.yaml`: self-contained Minecraft assignment
  config that registers `mmskills-gpt-4o` without editing VAB's shared
  `configs/agents/api_agents.yaml`.

This adapter is intentionally lightweight. VAB's HTTP prompter usually sends
one current image, so this adapter uses `SKILL.md` plus runtime state cues and
does not attach additional skill reference images to branch calls.

## Install Into VisualAgentBench

Run from this repository root:

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

## Configure Skills

Set the skills root through the YAML `skills_root` field or an environment
variable:

```bash
export VAB_MMSKILLS_SKILLS_ROOT=/path/to/vab_minecraft_mmskills
```

Optionally expose a fixed subset:

```bash
export VAB_MMSKILLS_SKILL_NAMES=skill_a,skill_b
```

## Run

Complete VAB-Minecraft setup first, including Docker, the Minecraft task image,
and Steve-1 weights under `data/minecraft`.

Terminal 1 starts the Minecraft task workers and controller:

```bash
python -m src.start_task --config configs/start_task.yaml --auto-controller
```

Terminal 2 launches evaluation with the MMSkills assignment:

```bash
python -m src.assigner --auto-retry --config configs/assignments/minecraft_mmskills.yaml
```

The assignment copied above is equivalent to:

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
