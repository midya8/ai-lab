"""File-based prompt registry: immutable, numbered, content-hashed versions plus movable tags."""
from __future__ import annotations

import difflib
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

from .template import Template

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RESERVED_TAG = re.compile(r"^(latest|v?\d+)$")


class RegistryError(Exception):
    pass


@dataclass(frozen=True)
class Version:
    name: str
    number: int
    hash: str
    text: str
    message: str
    created: str
    parent: int | None

    @property
    def template(self) -> Template:
        return Template(self.text)

    @property
    def label(self) -> str:
        return f"{self.name}@v{self.number} ({self.hash[:8]})"


class Registry:
    """Versions are stored as ``<root>/<name>/v0001.json``; tags in ``tags.json``."""

    def __init__(self, root: str | Path, clock: Callable[[], float] = time.time):
        self.root = Path(root)
        self.clock = clock

    # -- helpers ---------------------------------------------------------
    def _dir(self, name: str) -> Path:
        if not _NAME.match(name):
            raise RegistryError(f"invalid prompt name: {name!r}")
        return self.root / name

    def names(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and any(p.glob("v*.json")))

    def _load(self, path: Path) -> Version:
        return Version(**json.loads(path.read_text(encoding="utf-8")))

    def versions(self, name: str) -> list[Version]:
        d = self._dir(name)
        if not d.exists():
            return []
        return [self._load(p) for p in sorted(d.glob("v*.json"))]

    def _tags(self, name: str) -> dict[str, int]:
        p = self._dir(name) / "tags.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    # -- writes ----------------------------------------------------------
    def commit(self, name: str, text: str, message: str = "") -> tuple[Version, bool]:
        """Store ``text`` as a new version. Returns ``(version, created)``.

        Committing text identical to the latest version is a no-op, so
        re-running a deploy script never creates phantom versions.
        """
        if not text.strip():
            raise RegistryError("prompt text is empty")
        existing = self.versions(name)
        digest = Template(text).digest()
        if existing and existing[-1].hash == digest:
            return existing[-1], False
        number = existing[-1].number + 1 if existing else 1
        t = time.gmtime(self.clock())
        created = "%04d-%02d-%02dT%02d:%02d:%02dZ" % t[:6]
        v = Version(name, number, digest, text.rstrip("\n") + "\n", message, created,
                    existing[-1].number if existing else None)
        d = self._dir(name)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"v{number:04d}.json").write_text(json.dumps(asdict(v), indent=2) + "\n", encoding="utf-8")
        return v, True

    def tag(self, name: str, tag: str, ref: str = "latest") -> Version:
        if not _NAME.match(tag) or _RESERVED_TAG.match(tag):
            raise RegistryError(f"invalid tag: {tag!r} (not 'latest' or a version number)")
        v = self.resolve(name, ref)
        tags = self._tags(name)
        tags[tag] = v.number
        (self._dir(name) / "tags.json").write_text(json.dumps(tags, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return v

    # -- reads -----------------------------------------------------------
    def resolve(self, name: str, ref: str | None = "latest") -> Version:
        """``latest``, ``v3``/``3``, a tag, or a hash prefix (>= 4 chars)."""
        vs = self.versions(name)
        if not vs:
            raise RegistryError(f"no such prompt: {name}")
        ref = (ref or "latest").strip()
        if ref == "latest":
            return vs[-1]
        m = re.fullmatch(r"v?(\d+)", ref)
        if m:
            n = int(m.group(1))
            for v in vs:
                if v.number == n:
                    return v
            raise RegistryError(f"{name} has no version {n}")
        tags = self._tags(name)
        if ref in tags:
            return next(v for v in vs if v.number == tags[ref])
        if len(ref) >= 4 and re.fullmatch(r"[0-9a-f]+", ref):
            hits = [v for v in vs if v.hash.startswith(ref)]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise RegistryError(f"ambiguous hash prefix {ref!r}")
        raise RegistryError(f"cannot resolve {ref!r} for {name}")

    def tags_for(self, name: str) -> dict[str, int]:
        return self._tags(name)

    def diff(self, name: str, a: str, b: str) -> str:
        va, vb = self.resolve(name, a), self.resolve(name, b)
        lines = difflib.unified_diff(
            va.text.splitlines(), vb.text.splitlines(),
            fromfile=f"{name}@v{va.number}", tofile=f"{name}@v{vb.number}", lineterm="")
        return "\n".join(lines)
