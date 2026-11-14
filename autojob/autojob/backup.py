"""Backup and restore: the whole job search in one zip file.

``autojob backup`` packs the application log plus the two sidecar
settings files (``goals.json`` and ``questions.json``) into a single
zip, with a manifest that records what is inside and how many
applications it holds. ``autojob restore`` unpacks one into a fresh
directory, and ``backup --check`` reads a backup back and reports
what it holds without writing anything.

The honest rules, carried over from the rest of the tracker:

- A backup is a copy, never a move: creating one never touches the
  log or the sidecars, and restoring never touches the backup file.
  The application packets are not included — the log records where
  each one was written, and a packet can be rebuilt with ``apply``
  or ``batch`` — so a backup is the log and the settings, nothing
  else, and the manifest says so.
- Restore refuses to overwrite: if the target directory already
  holds an ``applications.jsonl`` (or a sidecar the backup would
  write), restore stops unless ``--force`` is given. A backup is
  fully validated before the first file is written, so a bad backup
  leaves the target directory exactly as it was.
- The manifest is checked on the way back in: the format and
  version must match, the application count must match the lines in
  the log, and every log line must parse as one JSON object. A
  backup that fails any of that is reported, not unpacked, and a
  sidecar that will not parse is reported the same way.
"""

from __future__ import annotations

import datetime
import json
import zipfile
from pathlib import Path

from . import __version__
from .goals import default_goals_path, load_goals
from .questions import default_questions_path, load_answers
from .tracker import TrackerError, read_log

BACKUP_FORMAT = "autojob-backup"
BACKUP_VERSION = 1
LOG_NAME = "applications.jsonl"
GOALS_NAME = "goals.json"
ANSWERS_NAME = "questions.json"
MANIFEST_NAME = "manifest.json"


class BackupError(ValueError):
    """A backup that cannot be created, read, or restored as stated."""


def default_backup_path(log_path, today=None):
    """The backup file that belongs with a log, dated today."""
    if today is None:
        today = datetime.date.today().isoformat()
    return str(Path(log_path).parent / f"autojob-backup-{today}.zip")


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _read_sidecar(path, loader, label):
    """Read a sidecar if it exists, else None. A corrupt one fails."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = loader(str(path))
    except ValueError as exc:
        raise BackupError(f"cannot back up {label} {path}: {exc}") from exc
    return path.read_bytes(), data


def create_backup(log_path, out_path, goals_path=None, answers_path=None,
                  today=None):
    """Pack the log and its sidecars into a zip. Returns a summary.

    The log is validated with ``read_log`` first, so a corrupt line
    fails the backup instead of being packed silently; the raw bytes
    are what go into the zip, so a round trip restores them exactly.
    The sidecars are validated the same way and packed only when
    they exist. ``today`` is the date recorded in the manifest
    (default: today); ``out_path`` is created, including parent
    directories, and an existing file at that path is replaced.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    elif _parse_date(today) is None:
        raise BackupError(f"--today must be YYYY-MM-DD, got {today!r}")
    log_path = Path(log_path)
    try:
        entries = read_log(log_path)
    except TrackerError as exc:
        raise BackupError(f"cannot back up log {log_path}: {exc}") from exc
    for i, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise BackupError(
                f"cannot back up log {log_path}: line {i} is not an "
                f"application")
    log_bytes = log_path.read_bytes() if log_path.exists() else b""
    goals = _read_sidecar(
        goals_path or default_goals_path(log_path), load_goals, "goals file")
    answers = _read_sidecar(
        answers_path or default_questions_path(log_path), load_answers,
        "answers file")
    manifest = {
        "format": BACKUP_FORMAT,
        "version": BACKUP_VERSION,
        "autojob_version": __version__,
        "created_at": today,
        "applications": len(entries),
        "files": {
            LOG_NAME: len(entries),
            GOALS_NAME: goals is not None,
            ANSWERS_NAME: answers is not None,
        },
        "note": ("The application log and the goals/questions settings. "
                 "Application packets are not included."),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME,
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        zf.writestr(LOG_NAME, log_bytes)
        if goals is not None:
            zf.writestr(GOALS_NAME, goals[0])
        if answers is not None:
            zf.writestr(ANSWERS_NAME, answers[0])
    return {
        "path": str(out_path),
        "created_at": today,
        "applications": len(entries),
        "goals": goals is not None,
        "answers": answers is not None,
        "files": [MANIFEST_NAME, LOG_NAME]
                 + ([GOALS_NAME] if goals is not None else [])
                 + ([ANSWERS_NAME] if answers is not None else []),
    }


def inspect_backup(path):
    """Read a backup back and report what it holds. Writes nothing.

    Every check runs before anything is returned: the file must be a
    zip, the manifest must be present and well formed with the right
    format and version, the log must be present with one JSON object
    per line and a count matching the manifest, and any sidecar
    present must parse as JSON. Returns the same summary shape
    ``create_backup`` returns, plus ``"valid": True``.
    """
    path = Path(path)
    if not path.exists():
        raise BackupError(f"no backup file at {path}")
    try:
        zf = zipfile.ZipFile(path, "r")
    except zipfile.BadZipFile as exc:
        raise BackupError(f"{path} is not a zip file: {exc}") from exc
    with zf:
        names = set(zf.namelist())
        if MANIFEST_NAME not in names:
            raise BackupError(f"{path} has no {MANIFEST_NAME} — not an "
                              f"autojob backup")
        try:
            manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BackupError(
                f"{path}: {MANIFEST_NAME} will not parse: {exc}") from exc
        if not isinstance(manifest, dict):
            raise BackupError(f"{path}: {MANIFEST_NAME} must hold one "
                              f"JSON object")
        if manifest.get("format") != BACKUP_FORMAT:
            raise BackupError(
                f"{path}: not an autojob backup "
                f"(format {manifest.get('format')!r})")
        if manifest.get("version") != BACKUP_VERSION:
            raise BackupError(
                f"{path}: backup version {manifest.get('version')!r} "
                f"is not version {BACKUP_VERSION}")
        if LOG_NAME not in names:
            raise BackupError(f"{path} has no {LOG_NAME} — not an "
                              f"autojob backup")
        try:
            log_text = zf.read(LOG_NAME).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BackupError(
                f"{path}: {LOG_NAME} is not UTF-8: {exc}") from exc
        count = 0
        for i, line in enumerate(log_text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BackupError(
                    f"{path}: {LOG_NAME} line {i} will not parse: "
                    f"{exc}") from exc
            if not isinstance(entry, dict):
                raise BackupError(
                    f"{path}: {LOG_NAME} line {i} is not an application")
            count += 1
        expected = manifest.get("applications")
        if expected != count:
            raise BackupError(
                f"{path}: manifest says {expected} applications but "
                f"{LOG_NAME} holds {count}")
        sidecars = {}
        for name in (GOALS_NAME, ANSWERS_NAME):
            if name in names:
                try:
                    json.loads(zf.read(name).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise BackupError(
                        f"{path}: {name} will not parse: {exc}") from exc
                sidecars[name] = True
    return {
        "path": str(path),
        "created_at": manifest.get("created_at"),
        "autojob_version": manifest.get("autojob_version"),
        "applications": count,
        "goals": sidecars.get(GOALS_NAME, False),
        "answers": sidecars.get(ANSWERS_NAME, False),
        "files": sorted(names),
        "valid": True,
    }


def restore_backup(backup_path, out_dir, force=False):
    """Unpack a backup into ``out_dir``. Returns a summary.

    The backup is fully validated first (see ``inspect_backup``), so
    a bad backup writes nothing. Then every file the target already
    holds is checked: without ``force`` an existing log or sidecar
    refuses the whole restore, so nothing is half overwritten. With
    ``force`` the backup's files replace what is there.
    """
    summary = inspect_backup(backup_path)
    out_dir = Path(out_dir)
    targets = [LOG_NAME]
    if summary["goals"]:
        targets.append(GOALS_NAME)
    if summary["answers"]:
        targets.append(ANSWERS_NAME)
    if not force:
        existing = [name for name in targets if (out_dir / name).exists()]
        if existing:
            raise BackupError(
                "refusing to restore: "
                + ", ".join(str(out_dir / name) for name in existing)
                + " already exist(s); use --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(backup_path, "r") as zf:
        for name in targets:
            (out_dir / name).write_bytes(zf.read(name))
    return {
        "path": str(backup_path),
        "restored_to": str(out_dir),
        "created_at": summary["created_at"],
        "applications": summary["applications"],
        "goals": summary["goals"],
        "answers": summary["answers"],
        "files": targets,
    }


def backup_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _sidecar_word(included, label):
    return f"included ({label})" if included else f"not present ({label})"


def render_backup_markdown(data):
    """Human-readable backup summary: what it holds, where it is."""
    if "restored_to" in data:
        lines = [f"Restored backup {data['path']} to {data['restored_to']} — "
                 f"{data['applications']} application"
                 + ("s" if data["applications"] != 1 else "")]
    elif data.get("valid"):
        lines = [f"Backup {data['path']} is valid — "
                 f"{data['applications']} application"
                 + ("s" if data["applications"] != 1 else "")]
        if data.get("created_at"):
            lines[0] += f", made {data['created_at']}"
        if data.get("autojob_version"):
            lines[0] += f" with autojob {data['autojob_version']}"
    else:
        lines = [f"Backup {data['path']} — {data['applications']} "
                 f"application"
                 + ("s" if data["applications"] != 1 else "")]
        if data.get("created_at"):
            lines[0] += f", made {data['created_at']}"
    lines.append("  goals: " + _sidecar_word(data["goals"], "goals.json"))
    lines.append("  answers: "
                 + _sidecar_word(data["answers"], "questions.json"))
    lines.append("  Application packets are not in the backup — the log "
                 "records where each one was written.")
    return "\n".join(lines) + "\n"
