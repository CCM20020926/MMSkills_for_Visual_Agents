# VAB-Minecraft Integration

This folder contains a minimal MMSkills adapter for VisualAgentBench
VAB-Minecraft.

## Included Files

- `mmskills_http_agent.py`: `AgentClient`/`HTTPAgent`-compatible MMSkills
  adapter. It keeps the normal VAB HTTP request path and adds an internal
  text planner branch when the model emits `LOAD_SKILL("<skill_name>")`.
- `openai-chat-mmskills.yaml`: example VAB agent config.

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
```

Then edit `openai-chat-mmskills.yaml` with your API key or use your existing
gateway URL and headers. Register the copied config in
`configs/agents/api_agents.yaml`:

```yaml
mmskills-gpt-4o:
    import: "./openai-chat-mmskills.yaml"
    parameters:
        name: "mmskills-gpt-4o"
        body:
            model: "gpt-4o-2024-05-13"
            max_tokens: 512
```

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

Use the registered agent name in the assignment file, for example in
`configs/assignments/default.yaml`:

```yaml
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
```

Then start the normal VAB workers:

```bash
python -m src.start_task --config configs/start_task.yaml --auto-controller
```

Keep the rest of the VAB-Minecraft setup unchanged. The adapter returns the
benchmark's normal `OBSERVATION`/`THOUGHT`/`ACTION` response after any internal
MMSkills branch consultation.
