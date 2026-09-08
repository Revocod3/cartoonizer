from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def effective_settings(batch: dict[str, Any], individual: dict[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(batch)
    merged.update(individual or {})
    return merged


def validate_svg(svg_text: str) -> bool:
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError:
        return False
    if root.tag.rsplit("}", 1)[-1] != "svg":
        return False
    geometry = {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}
    has_geometry = False
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "image":
            return False
        has_geometry |= tag in geometry
        for attribute, value in element.attrib.items():
            if attribute.rsplit("}", 1)[-1] == "href" and value.strip().lower().startswith(("http:", "https:", "//", "data:", "file:")):
                return False
    return has_geometry


def render_svg(source_svg: Path, destination_png: Path, output_width: int | None = None) -> Path:
    import cairosvg

    cairosvg.svg2png(url=str(source_svg), write_to=str(destination_png), output_width=output_width)
    return destination_png


def has_transparency(path: Path) -> bool:
    from PIL import Image

    with Image.open(path) as image:
        if "A" not in image.getbands():
            return False
        minimum, _ = image.getchannel("A").getextrema()
        return minimum < 255


def export_version(source_png: Path, source_svg: Path, destination: Path, stem: str, width: int, height: int, report: dict[str, Any] | None = None) -> dict[str, Any]:
    if width < 1 or height < 1:
        raise ValueError("Export dimensions must be positive")
    if not validate_svg(source_svg.read_text(encoding="utf-8")):
        raise ValueError("SVG is not self-contained vector geometry")
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow is required to export PNG files") from exc
    destination.mkdir(parents=True, exist_ok=True)
    png_target, svg_target = destination / f"{stem}.png", destination / f"{stem}.svg"
    with Image.open(source_png) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        source_width, source_height = image.size
        if source_width * height != source_height * width:
            raise ValueError("Requested dimensions do not preserve the source aspect ratio")
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        image.save(png_target, "PNG", optimize=True)
    shutil.copyfile(source_svg, svg_target)
    manifest = {"stem": stem, "width": width, "height": height, "png": png_target.name, "svg": svg_target.name, **(report or {})}
    (destination / f"{stem}.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


class Pipeline(Protocol):
    def analyze(self, original: Path, settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]: ...
    def generate(self, original: Path, instructions: dict[str, Any], destination: Path, previous: Path | None = None) -> tuple[Path, dict[str, int]]: ...
    def evaluate(self, original: Path, candidate: Path, instructions: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]: ...
    def vectorize(self, candidate: Path, destination: Path, mode: str) -> Path: ...


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init()

    def close(self) -> None:
        self.conn.close()

    def _init(self) -> None:
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, name TEXT NOT NULL, settings TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, original_path TEXT NOT NULL, individual_settings TEXT NOT NULL DEFAULT '{}', stage TEXT NOT NULL DEFAULT 'pending', analysis TEXT, candidate_path TEXT, svg_path TEXT, evaluation TEXT, decision TEXT, correction_note TEXT, error TEXT, updated_at TEXT NOT NULL, UNIQUE(batch_id, original_path));
        CREATE TABLE IF NOT EXISTS versions (id TEXT PRIMARY KEY, job_id TEXT NOT NULL, number INTEGER NOT NULL, candidate_path TEXT NOT NULL, svg_path TEXT NOT NULL, evaluation TEXT NOT NULL, correction TEXT, accepted INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, UNIQUE(job_id, number));
        CREATE TABLE IF NOT EXISTS usage (id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT NOT NULL, provider TEXT NOT NULL, input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0, image_attempts INTEGER DEFAULT 0, created_at TEXT NOT NULL);
        """)
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(jobs)")}
        if "error" not in columns:
            self.conn.execute("ALTER TABLE jobs ADD COLUMN error TEXT")
        self.conn.commit()

    def create_batch(self, name: str, settings: dict[str, Any]) -> str:
        batch_id = str(uuid.uuid4())
        self.conn.execute("INSERT INTO batches VALUES (?,?,?,?)", (batch_id, name, json.dumps(settings), utc_now()))
        self.conn.commit()
        return batch_id

    def batches(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM batches ORDER BY created_at DESC")]

    def batch(self, batch_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM batches WHERE id=?", (batch_id,)).fetchone()
        if not row:
            raise KeyError(batch_id)
        result = dict(row)
        result["settings"] = json.loads(result["settings"])
        return result

    def add_job(self, batch_id: str, original_path: str, individual_settings: dict[str, Any] | None = None) -> str:
        job_id = str(uuid.uuid4())
        self.conn.execute("INSERT OR IGNORE INTO jobs(id,batch_id,original_path,individual_settings,updated_at) VALUES (?,?,?,?,?)", (job_id, batch_id, original_path, json.dumps(individual_settings or {}), utc_now()))
        self.conn.commit()
        return self.conn.execute("SELECT id FROM jobs WHERE batch_id=? AND original_path=?", (batch_id, original_path)).fetchone()["id"]

    def update_job_settings(self, job_id: str, settings: dict[str, Any]) -> None:
        self.conn.execute("UPDATE jobs SET individual_settings=?, updated_at=? WHERE id=?", (json.dumps(settings), utc_now(), job_id))
        self.conn.commit()

    def claim_next_job(self, batch_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE batch_id=? AND stage='pending' ORDER BY rowid LIMIT 1", (batch_id,)).fetchone()
        if not row:
            return None
        self.conn.execute("UPDATE jobs SET stage='analysis', updated_at=? WHERE id=? AND stage='pending'", (utc_now(), row["id"]))
        self.conn.commit()
        return dict(row)

    def finish_stage(self, job_id: str, stage: str, payload: dict[str, Any]) -> None:
        mapping = {"analysis": ("analysis", json.dumps(payload), "analyzed"), "generated": ("candidate_path", payload.get("path"), "generated"), "evaluated": ("evaluation", json.dumps(payload), "evaluated")}
        if stage in mapping:
            column, value, next_stage = mapping[stage]
            self.conn.execute(f"UPDATE jobs SET {column}=?, stage=?, error=NULL, updated_at=? WHERE id=?", (value, next_stage, utc_now(), job_id))
        else:
            self.conn.execute("UPDATE jobs SET stage=?, error=NULL, updated_at=? WHERE id=?", (stage, utc_now(), job_id))
        self.conn.commit()

    def fail_job(self, job_id: str, message: str, uncertain: bool = False) -> None:
        self.conn.execute("UPDATE jobs SET stage=?, error=?, updated_at=? WHERE id=?", ("generation_uncertain" if uncertain else "failed", message, utc_now(), job_id))
        self.conn.commit()

    def retry_failed(self, batch_id: str) -> int:
        jobs = self.conn.execute("SELECT id,analysis,candidate_path,evaluation FROM jobs WHERE batch_id=? AND stage IN ('failed','generation_uncertain')", (batch_id,)).fetchall()
        for job in jobs:
            stage = "evaluated" if job["evaluation"] else "generated" if job["candidate_path"] else "analyzed" if job["analysis"] else "pending"
            self.conn.execute("UPDATE jobs SET stage=?,error=NULL,updated_at=? WHERE id=?", (stage, utc_now(), job["id"]))
        self.conn.commit()
        return len(jobs)

    def job(self, job_id: str) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        return dict(row)

    def jobs(self, batch_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM jobs WHERE batch_id=? ORDER BY rowid", (batch_id,))]

    def add_version(self, job_id: str, candidate_path: str, svg_path: str, evaluation: dict[str, Any], correction: str | None = None) -> str:
        number = self.conn.execute("SELECT COALESCE(MAX(number),0)+1 FROM versions WHERE job_id=?", (job_id,)).fetchone()[0]
        version_id = str(uuid.uuid4())
        self.conn.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?,?,?)", (version_id, job_id, number, candidate_path, svg_path, json.dumps(evaluation), correction, 0, utc_now()))
        self.conn.execute("UPDATE jobs SET candidate_path=?,svg_path=?,evaluation=?,stage='ready',error=NULL,updated_at=? WHERE id=?", (candidate_path, svg_path, json.dumps(evaluation), utc_now(), job_id))
        self.conn.commit()
        return version_id

    def versions(self, job_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM versions WHERE job_id=? ORDER BY number DESC", (job_id,))]

    def accept_version(self, job_id: str, version_id: str) -> None:
        if not self.conn.execute("SELECT 1 FROM versions WHERE id=? AND job_id=?", (version_id, job_id)).fetchone():
            raise ValueError("Version does not belong to job")
        self.conn.execute("UPDATE versions SET accepted=0 WHERE job_id=?", (job_id,))
        self.conn.execute("UPDATE versions SET accepted=1 WHERE id=?", (version_id,))
        self.conn.execute("UPDATE jobs SET decision='selected',updated_at=? WHERE id=?", (utc_now(), job_id))
        self.conn.commit()

    def accepted_version(self, job_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM versions WHERE job_id=? AND accepted=1", (job_id,)).fetchone()
        return dict(row) if row else None

    def decide(self, job_id: str, decision: str, note: str = "") -> None:
        self.conn.execute("UPDATE jobs SET decision=?,correction_note=?,updated_at=? WHERE id=?", (decision, note, utc_now(), job_id))
        self.conn.commit()

    def record_usage(self, batch_id: str, provider: str, input_tokens: int = 0, output_tokens: int = 0, image_attempts: int = 0) -> None:
        self.conn.execute("INSERT INTO usage(batch_id,provider,input_tokens,output_tokens,image_attempts,created_at) VALUES (?,?,?,?,?,?)", (batch_id, provider, input_tokens, output_tokens, image_attempts, utc_now()))
        self.conn.commit()

    def batch_report(self, batch_id: str) -> dict[str, Any]:
        jobs = self.jobs(batch_id)
        counts = {"total": len(jobs), "selected": sum(j["decision"] == "selected" for j in jobs), "discarded": sum(j["decision"] == "discarded" for j in jobs), "failed": sum(j["stage"] in ("failed", "generation_uncertain") for j in jobs), "ready": sum(j["stage"] == "ready" for j in jobs)}
        usage: dict[str, dict[str, int]] = defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "image_attempts": 0})
        for row in self.conn.execute("SELECT provider,input_tokens,output_tokens,image_attempts FROM usage WHERE batch_id=?", (batch_id,)):
            for key in ("input_tokens", "output_tokens", "image_attempts"):
                usage[row["provider"]][key] += row[key]
        return {"batch": self.batch(batch_id), "counts": counts, "usage": dict(usage), "jobs": jobs}


class BatchWorker:
    def __init__(self, db: Database, pipeline: Pipeline, root: Path):
        self.db, self.pipeline, self.root = db, pipeline, Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_batch(self, batch_id: str) -> int:
        completed = 0
        for row in self.db.jobs(batch_id):
            if row["stage"] in ("ready", "failed", "generation_uncertain"):
                continue
            try:
                self._resume(batch_id, row["id"])
                completed += 1
            except Exception as exc:
                self.db.fail_job(row["id"], str(exc), uncertain=self.db.job(row["id"])["stage"] == "generating")
        return completed

    def _resume(self, batch_id: str, job_id: str) -> None:
        batch, job = self.db.batch(batch_id), self.db.job(job_id)
        original = Path(job["original_path"])
        settings = effective_settings(batch["settings"], json.loads(job["individual_settings"]))
        job_root = self.root / job_id
        job_root.mkdir(parents=True, exist_ok=True)
        if job["stage"] in ("pending", "analysis"):
            if job["stage"] == "pending":
                self.db.conn.execute("UPDATE jobs SET stage='analysis',updated_at=? WHERE id=?", (utc_now(), job_id)); self.db.conn.commit()
            analysis, usage = self.pipeline.analyze(original, settings)
            analysis["effective_settings"] = settings
            self.db.record_usage(batch_id, "gpt-5.6-luna", **usage)
            self.db.finish_stage(job_id, "analysis", analysis)
            job = self.db.job(job_id)
        analysis = {**settings, **json.loads(job["analysis"])}
        if job["stage"] == "analyzed":
            destination = job_root / "candidate-1.png"
            self.db.conn.execute("UPDATE jobs SET stage='generating',updated_at=? WHERE id=?", (utc_now(), job_id)); self.db.conn.commit()
            generated, usage = self.pipeline.generate(original, analysis, destination)
            self.db.record_usage(batch_id, "gpt-image-2", **usage)
            self.db.finish_stage(job_id, "generated", {"path": str(generated)})
            job = self.db.job(job_id)
        candidate = Path(job["candidate_path"])
        if job["stage"] == "generated":
            evaluation, usage = self.pipeline.evaluate(original, candidate, analysis)
            if settings.get("background") == "transparent" and not has_transparency(candidate):
                evaluation["category"] = "requires_attention"
                evaluation.setdefault("defects", []).append({"criterion": "background", "severity": "high", "region": "background", "evidence": "Requested transparency is absent", "correction": "Remove the background and preserve alpha"})
            self.db.record_usage(batch_id, "gpt-5.6-terra", **usage)
            self.db.finish_stage(job_id, "evaluated", evaluation)
            job = self.db.job(job_id)
        if job["stage"] == "evaluated":
            svg = self.pipeline.vectorize(candidate, job_root / "candidate-1.svg", settings.get("vector_mode", "detail"))
            if not validate_svg(svg.read_text(encoding="utf-8")):
                raise ValueError("Vectorizer returned an invalid SVG")
            evaluation = json.loads(job["evaluation"])
            if hasattr(self.pipeline, "evaluate_vector"):
                rendered = render_svg(svg, job_root / "candidate-1-svg.png")
                vector_evaluation, usage = self.pipeline.evaluate_vector(candidate, rendered)
                self.db.record_usage(batch_id, "gpt-5.6-terra", **usage)
                evaluation["vector_evaluation"] = vector_evaluation
            self.db.add_version(job_id, str(candidate), str(svg), evaluation)

    def correct(self, batch_id: str, job_id: str, note: str) -> str:
        if not note.strip():
            raise ValueError("Correction note is required")
        job = self.db.job(job_id)
        settings = effective_settings(self.db.batch(batch_id)["settings"], json.loads(job["individual_settings"]))
        analysis = {**settings, **json.loads(job["analysis"]), "correction": note}
        number, job_root = len(self.db.versions(job_id)) + 1, self.root / job_id
        generated, usage = self.pipeline.generate(Path(job["original_path"]), analysis, job_root / f"candidate-{number}.png", Path(job["candidate_path"]))
        self.db.record_usage(batch_id, "gpt-image-2", **usage)
        evaluation, usage = self.pipeline.evaluate(Path(job["original_path"]), generated, analysis)
        self.db.record_usage(batch_id, "gpt-5.6-terra", **usage)
        svg = self.pipeline.vectorize(generated, job_root / f"candidate-{number}.svg", "detail")
        if not validate_svg(svg.read_text(encoding="utf-8")):
            raise ValueError("Vectorizer returned an invalid SVG")
        if hasattr(self.pipeline, "evaluate_vector"):
            rendered = render_svg(svg, job_root / f"candidate-{number}-svg.png")
            vector_evaluation, usage = self.pipeline.evaluate_vector(generated, rendered)
            self.db.record_usage(batch_id, "gpt-5.6-terra", **usage)
            evaluation["vector_evaluation"] = vector_evaluation
        return self.db.add_version(job_id, str(generated), str(svg), evaluation, note)


class SimulatedPipeline:
    def analyze(self, original: Path, settings: dict[str, Any]):
        return ({"subjects": ["anime character"], "pose": "preserve original", "ambiguities": [], "edit_instructions": settings.get("notes", "")}, {"input_tokens": 0, "output_tokens": 0})

    def generate(self, original: Path, instructions: dict[str, Any], destination: Path, previous: Path | None = None):
        shutil.copyfile(previous or original, destination)
        return destination, {"image_attempts": 0}

    def evaluate(self, original: Path, candidate: Path, instructions: dict[str, Any]):
        return ({"category": "possible_correction", "criteria": {"identity": "not_verifiable"}, "defects": [], "uncertainties": ["Simulated evaluation"]}, {"input_tokens": 0, "output_tokens": 0})

    def vectorize(self, candidate: Path, destination: Path, mode: str):
        destination.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><path d="M10 10h80v80H10z" fill="none" stroke="currentColor"/></svg>', encoding="utf-8")
        return destination
