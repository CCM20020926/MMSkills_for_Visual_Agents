"""Filesystem loader for VAB-Minecraft MMSkill packages.

Copy this file next to `gemini_minecraft_skills_agent.py` in a VisualAgentBench
agent package. It loads product-neutral MMSkills assets and resolves selected
runtime state-card image views for planner branches.
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    directory: str


@dataclass(frozen=True)
class SkillContent:
    name: str
    description: str
    text: str
    directory: str


@dataclass(frozen=True)
class SkillStateView:
    view_type: str = ""
    image_path: str = ""
    use_for: str = ""
    label: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillStateBundle:
    state_id: str = ""
    state_name: str = ""
    stage: str = ""
    image_role: str = ""
    when_to_use: str = ""
    when_not_to_use: str = ""
    visible_cues: List[str] = field(default_factory=list)
    verification_cue: str = ""
    visual_risk: str = ""
    preferred_view_order: List[str] = field(default_factory=list)
    available_views: List[SkillStateView] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillStateBundleSet:
    skill_name: str
    schema_version: str
    source_file: str
    bundles: List[SkillStateBundle] = field(default_factory=list)


@dataclass
class LoadedSkillStateView:
    view: SkillStateView
    image_path: str
    mime_type: str


@dataclass
class ResolvedSkillStateSelection:
    state: SkillStateBundle
    requested_view_types: List[str] = field(default_factory=list)
    reason: str = ""
    loaded_views: List[LoadedSkillStateView] = field(default_factory=list)


class MinecraftSkillLoader:
    """Load VAB-Minecraft MMSkills from a local skill package root."""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

    def __init__(self, skills_library_dir: str, max_skill_chars: int = 10000) -> None:
        self.skills_dir = Path(skills_library_dir).expanduser()
        if not self.skills_dir.is_absolute():
            self.skills_dir = Path.cwd() / self.skills_dir
        self.max_skill_chars = max(1000, int(max_skill_chars))
        self._metadata_cache: Optional[List[SkillMetadata]] = None
        self._content_cache: Dict[str, Optional[SkillContent]] = {}
        self._bundle_cache: Dict[str, Optional[SkillStateBundleSet]] = {}
        self._dir_cache: Dict[str, Path] = {}

    def discover_skills(self) -> List[SkillMetadata]:
        if self._metadata_cache is not None:
            return list(self._metadata_cache)

        if not self.skills_dir.exists():
            logger.warning("Minecraft skills directory not found: %s", self.skills_dir)
            self._metadata_cache = []
            return []

        result: List[SkillMetadata] = []
        for skill_dir in sorted(path for path in self.skills_dir.iterdir() if path.is_dir()):
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            raw = skill_md.read_text(encoding="utf-8", errors="replace")
            frontmatter = self._parse_frontmatter(raw)
            result.append(
                SkillMetadata(
                    name=frontmatter.get("name") or skill_dir.name,
                    description=frontmatter.get("description") or "",
                    directory=str(skill_dir),
                )
            )
            self._dir_cache[skill_dir.name] = skill_dir

        self._metadata_cache = result
        return list(result)

    def skill_names(self) -> List[str]:
        return [Path(meta.directory).name for meta in self.discover_skills()]

    def load_skill_content(self, skill_name: str) -> Optional[SkillContent]:
        skill_dir = self._resolve_skill_dir(skill_name)
        if skill_dir is None:
            return None
        key = skill_dir.name
        if key in self._content_cache:
            return self._content_cache[key]

        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            self._content_cache[key] = None
            return None

        raw = skill_md.read_text(encoding="utf-8", errors="replace")
        frontmatter = self._parse_frontmatter(raw)
        body = self._strip_frontmatter(raw)
        if len(body) > self.max_skill_chars:
            body = body[: self.max_skill_chars].rstrip() + "\n\n[Truncated]"
        content = SkillContent(
            name=frontmatter.get("name") or key,
            description=frontmatter.get("description") or "",
            text=body,
            directory=str(skill_dir),
        )
        self._content_cache[key] = content
        return content

    def load_state_bundles(self, skill_name: str) -> Optional[SkillStateBundleSet]:
        skill_dir = self._resolve_skill_dir(skill_name)
        if skill_dir is None:
            return None
        key = skill_dir.name
        if key in self._bundle_cache:
            return self._bundle_cache[key]

        state_path = skill_dir / "runtime_state_cards.json"
        if not state_path.exists():
            state_path = skill_dir / "state_cards.json"
        if not state_path.exists():
            self._bundle_cache[key] = None
            return None

        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", state_path, exc)
            self._bundle_cache[key] = None
            return None

        raw_states = payload.get("states") if isinstance(payload, dict) else None
        if not isinstance(raw_states, list):
            logger.warning("Unsupported state schema in %s", state_path)
            self._bundle_cache[key] = None
            return None

        bundles: List[SkillStateBundle] = []
        for item in raw_states:
            if not isinstance(item, dict):
                continue
            views: List[SkillStateView] = []
            raw_views = item.get("available_views", [])
            if isinstance(raw_views, list):
                for view in raw_views:
                    if not isinstance(view, dict):
                        continue
                    views.append(
                        SkillStateView(
                            view_type=str(view.get("view_type", "") or "").strip(),
                            image_path=str(view.get("image_path", "") or "").strip(),
                            use_for=str(view.get("use_for", "") or "").strip(),
                            label=str(view.get("label", "") or "").strip(),
                            raw=view,
                        )
                    )

            visible_cues = item.get("visible_cues", [])
            if not isinstance(visible_cues, list):
                visible_cues = []
            preferred_view_order = item.get("preferred_view_order", [])
            if not isinstance(preferred_view_order, list):
                preferred_view_order = []

            bundles.append(
                SkillStateBundle(
                    state_id=str(item.get("state_id", "") or "").strip(),
                    state_name=str(item.get("state_name", "") or "").strip(),
                    stage=str(item.get("stage", "") or "").strip(),
                    image_role=str(item.get("image_role", "") or "").strip(),
                    when_to_use=str(item.get("when_to_use", "") or "").strip(),
                    when_not_to_use=str(item.get("when_not_to_use", "") or "").strip(),
                    visible_cues=[str(cue).strip() for cue in visible_cues if str(cue).strip()],
                    verification_cue=str(item.get("verification_cue", "") or "").strip(),
                    visual_risk=str(item.get("visual_risk", "") or "").strip(),
                    preferred_view_order=[
                        str(view_type).strip()
                        for view_type in preferred_view_order
                        if str(view_type).strip()
                    ],
                    available_views=views,
                    raw=item,
                )
            )

        bundle_set = SkillStateBundleSet(
            skill_name=key,
            schema_version=str(payload.get("schema_version", "") or ""),
            source_file=str(state_path),
            bundles=bundles,
        )
        self._bundle_cache[key] = bundle_set
        return bundle_set

    def format_state_bundles_for_branch(self, skill_name: str) -> str:
        bundles = self.load_state_bundles(skill_name)
        if not bundles or not bundles.bundles:
            return "No runtime state bundles are available for this skill."

        lines = [
            f"# Runtime State Bundles ({skill_name})",
            "",
            "Use these Minecraft state bundles to decide which visual references are useful.",
            "Reference images are state-recognition aids only; do not copy coordinates, terrain layout, or example counts.",
            "",
        ]
        for idx, bundle in enumerate(bundles.bundles, start=1):
            lines.extend(
                [
                    f"{idx}. state_id: {bundle.state_id or '(missing)'}",
                    f"   state_name: {bundle.state_name or '(missing)'}",
                    f"   stage: {bundle.stage or '(unknown)'}",
                    f"   when_to_use: {bundle.when_to_use or '(missing)'}",
                    f"   when_not_to_use: {bundle.when_not_to_use or '(missing)'}",
                    "   visible_cues: "
                    + ("; ".join(bundle.visible_cues) if bundle.visible_cues else "(none listed)"),
                ]
            )
            if bundle.verification_cue:
                lines.append(f"   verification_cue: {bundle.verification_cue}")
            if bundle.visual_risk:
                lines.append(f"   visual_risk: {bundle.visual_risk}")
            lines.append("   available_views:")
            if bundle.available_views:
                for view in bundle.available_views:
                    lines.append(
                        "   - "
                        f"view_type: {view.view_type or '(unknown)'} | "
                        f"use_for: {view.use_for or '(missing)'} | "
                        f"label: {view.label or '(missing)'}"
                    )
            else:
                lines.append("   - (no views listed)")
            lines.append("")
        return "\n".join(lines).rstrip()

    def load_selected_state_views(
        self,
        skill_name: str,
        requested_items: List[Dict[str, Any]],
    ) -> Tuple[List[ResolvedSkillStateSelection], List[str]]:
        skill_dir = self._resolve_skill_dir(skill_name)
        bundles = self.load_state_bundles(skill_name)
        if skill_dir is None or not bundles or not bundles.bundles:
            return [], ["No state bundles are available for this skill."]

        by_state = {
            self._normalize_selector(bundle.state_id): bundle
            for bundle in bundles.bundles
            if self._normalize_selector(bundle.state_id)
        }
        selections: Dict[str, ResolvedSkillStateSelection] = {}
        missing: List[str] = []

        for item in requested_items or []:
            if not isinstance(item, dict):
                missing.append(str(item))
                continue
            raw_state_id = str(item.get("state_id", "") or "").strip()
            bundle = by_state.get(self._normalize_selector(raw_state_id))
            if bundle is None:
                missing.append(f"state_id:{raw_state_id or '(missing)'}")
                continue

            raw_views = item.get("views", [])
            requested_view_types = (
                [str(view_type).strip() for view_type in raw_views if str(view_type).strip()]
                if isinstance(raw_views, list)
                else []
            )
            if not requested_view_types:
                requested_view_types = self._default_views_for_state(bundle)

            selection = selections.get(bundle.state_id)
            if selection is None:
                selection = ResolvedSkillStateSelection(
                    state=bundle,
                    requested_view_types=[],
                    reason=str(item.get("reason", "") or "").strip(),
                    loaded_views=[],
                )
                selections[bundle.state_id] = selection

            available_by_view = {
                self._normalize_selector(view.view_type): view
                for view in bundle.available_views
                if self._normalize_selector(view.view_type)
            }
            for view_type in requested_view_types:
                normalized_view = self._normalize_selector(view_type)
                view = available_by_view.get(normalized_view)
                if view is None:
                    missing.append(f"{bundle.state_id}:{view_type}")
                    continue
                if view.view_type in selection.requested_view_types:
                    continue
                selection.requested_view_types.append(view.view_type)
                image_path = self._resolve_image_path(skill_dir, view.image_path)
                if image_path is None:
                    missing.append(f"{bundle.state_id}:{view.view_type}")
                    continue
                selection.loaded_views.append(
                    LoadedSkillStateView(
                        view=view,
                        image_path=str(image_path),
                        mime_type=self._mime_type(image_path),
                    )
                )

        return [selection for selection in selections.values() if selection.loaded_views], missing

    def encode_image(self, image_path: str) -> Tuple[str, str]:
        path = Path(image_path)
        return base64.b64encode(path.read_bytes()).decode("utf-8"), self._mime_type(path)

    def _resolve_skill_dir(self, skill_name: str) -> Optional[Path]:
        if not self._dir_cache:
            self.discover_skills()
        raw = str(skill_name or "").strip()
        if not raw:
            return None
        candidate = Path(raw).expanduser()
        if candidate.is_absolute() and candidate.is_dir():
            return candidate
        if raw in self._dir_cache:
            return self._dir_cache[raw]

        normalized = self._normalize_selector(raw)
        for name, path in self._dir_cache.items():
            if self._normalize_selector(name) == normalized:
                return path
        return None

    @staticmethod
    def _resolve_image_path(skill_dir: Path, image_path: str) -> Optional[Path]:
        raw = str(image_path or "").strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        if path.is_absolute() and path.exists():
            return path
        candidate = skill_dir / raw
        if candidate.exists():
            return candidate
        candidate = skill_dir / "Images" / Path(raw).name
        if candidate.exists():
            return candidate
        images_dir = skill_dir / "Images"
        if images_dir.exists():
            for item in images_dir.iterdir():
                if item.name.lower() == Path(raw).name.lower():
                    return item
        return None

    @staticmethod
    def _default_views_for_state(bundle: SkillStateBundle) -> List[str]:
        available = {view.view_type for view in bundle.available_views if view.view_type}
        result = [view_type for view_type in bundle.preferred_view_order if view_type in available]
        if result:
            return result[:2]
        for preferred in ("full_frame", "focus_crop", "before", "after"):
            if preferred in available and preferred not in result:
                result.append(preferred)
        if result:
            return result[:2]
        return [bundle.available_views[0].view_type] if bundle.available_views else []

    @staticmethod
    def _parse_frontmatter(text: str) -> Dict[str, str]:
        if not text.startswith("---"):
            return {}
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.DOTALL)
        if not match:
            return {}
        result: Dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().strip("\"'")
        return result

    @staticmethod
    def _strip_frontmatter(text: str) -> str:
        if not text.startswith("---"):
            return text.strip()
        return re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.DOTALL).strip()

    @staticmethod
    def _normalize_selector(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")

    @staticmethod
    def _mime_type(path: Path) -> str:
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".gif": "image/gif",
        }.get(path.suffix.lower(), "image/png")
