# 🧭 Task-Skill Mapping

`task_skill_mapping.json` is a compact global mapping from OSWorld domain and
task ID to resolver-ready skill names.

The full MMSkills packages are hosted on Hugging Face rather than bundled in
this GitHub branch:

```text
https://huggingface.co/datasets/zhangkangning/mmskills
```

Within the dataset, the canonical Ubuntu task mapping is:

```text
ubuntu/task_skill_mapping.json
```

That source file uses a richer metadata schema with `summary`, `domains`, and
`task_to_skills`. The checked-in GitHub file keeps only the compact runtime
shape currently consumed by `mm_agents.task_skill_resolver`:

```json
{
  "chrome": {
    "<task_id>": ["CHROME_..."]
  },
  "vs_code": {
    "<task_id>": ["VSCODE_..."]
  }
}
```

Current converted coverage:

- 10 Ubuntu OSWorld domains.
- 360 task IDs.
- 437 task-skill assignments.
- 170 unique referenced public skills.

Per-domain task-ID coverage in this compact file:

| Domain | Mapped task IDs |
| --- | ---: |
| chrome | 45 |
| gimp | 26 |
| libreoffice_calc | 47 |
| libreoffice_impress | 47 |
| libreoffice_writer | 23 |
| multi_apps | 93 |
| os | 24 |
| thunderbird | 15 |
| vlc | 17 |
| vs_code | 23 |

For Chrome specifically, OSWorld's raw `evaluation_examples/examples/chrome/`
directory can contain 46 task JSON files, while the no-Google-Drive split used
by the README evaluation commands (`evaluation_examples/test_nogdrive.json`)
contains 45 Chrome tasks. This mapping covers all 45 Chrome tasks in that split.
The raw task `06fe7178-4491-4589-810f-2e2bc9502122` is intentionally absent
because it is not part of `test_nogdrive.json`.

The mapping may reference skills that are not present in the small local
`skills_library/` subset. For full coverage, download the corresponding packages
from the Hugging Face dataset or use the MMSkills Agent Adapter for on-demand
retrieval.

Use it with the MMSkills-aware OSWorld runner:

```bash
python run.py \
  --agent_type mm_skill \
  --skills_library_dir skills_library \
  --task_skill_mapping_root task_skill_mappings/task_skill_mapping.json \
  --skill_mode multimodal
```

The resolver also supports per-domain generated mapping files, but this release
uses one global domain-first file so the runner can load all Ubuntu task-skill
assignments through a single `--task_skill_mapping_root` path.
