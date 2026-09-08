import base64
import json
import struct
import tempfile
import unittest
from pathlib import Path

from cartoonizer.core import (
    BatchWorker,
    Database,
    effective_settings,
    export_version,
    render_svg,
    validate_svg,
)
from cartoonizer.providers import VTracerPipeline, choose_image_quality, choose_image_size
from cartoonizer.models import Evaluation


def assert_strict_object_schema(test_case, schema):
    if schema.get("type") == "object":
        test_case.assertEqual(set(schema.get("required", [])), set(schema.get("properties", {})))
        test_case.assertFalse(schema.get("additionalProperties", True))
    for value in schema.values():
        if isinstance(value, dict):
            assert_strict_object_schema(test_case, value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    assert_strict_object_schema(test_case, item)


PNG_2X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAYAAAD0In+KAAAAFElEQVR4nGP4z8Dwn4GBgYGJAQoAHgQCAcwB/3sAAAAASUVORK5CYII="
)


class FakePipeline:
    def analyze(self, original, settings):
        return {"subjects": ["person"], "edit_instructions": "preserve"}, {"input_tokens": 10, "output_tokens": 4}

    def generate(self, original, instructions, destination, previous=None):
        destination.write_bytes(original.read_bytes())
        return destination, {"image_attempts": 1}

    def evaluate(self, original, candidate, instructions):
        return {"category": "no_defects", "criteria": {"identity": "pass"}}, {"input_tokens": 8, "output_tokens": 3}

    def vectorize(self, candidate, destination, mode):
        destination.write_text('<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0h2v1H0z"/></svg>')
        return destination


class CartoonizerCoreTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.db = Database(self.root / "cartoonizer.sqlite")

    def tearDown(self):
        self.db.close()
        self.folder.cleanup()

    def test_individual_settings_override_batch_settings(self):
        settings = effective_settings(
            {"background": "transparent", "cleanup": "soft", "colour": "faithful"},
            {"background": "white", "cleanup": "strong"},
        )
        self.assertEqual(settings["background"], "white")
        self.assertEqual(settings["cleanup"], "strong")
        self.assertEqual(settings["colour"], "faithful")

    def test_evaluation_schema_is_strict_and_has_fixed_criteria(self):
        from openai.lib._pydantic import to_strict_json_schema

        schema = to_strict_json_schema(Evaluation)
        assert_strict_object_schema(self, schema)
        criteria = schema["$defs"]["EvaluationCriteria"]
        self.assertEqual(
            set(criteria["properties"]),
            {"identity", "expression", "pose", "details", "text", "lines", "colors", "background"},
        )

    def test_completed_stage_is_not_claimed_twice(self):
        batch_id = self.db.create_batch("Pilot", {"background": "transparent"})
        job_id = self.db.add_job(batch_id, "originals/a.png")
        first = self.db.claim_next_job(batch_id)
        self.assertEqual(first["id"], job_id)
        self.db.finish_stage(job_id, "analysis", {"subjects": ["person"]})
        self.assertIsNone(self.db.claim_next_job(batch_id))

    def test_svg_validation_rejects_embedded_raster(self):
        self.assertFalse(validate_svg('<svg><image href="data:image/png;base64,abc"/></svg>'))
        self.assertTrue(validate_svg('<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0h1v1z"/></svg>'))

    def test_svg_validation_requires_geometry_and_rejects_external_references(self):
        self.assertFalse(validate_svg('<svg xmlns="http://www.w3.org/2000/svg"/>'))
        self.assertFalse(validate_svg('<svg xmlns="http://www.w3.org/2000/svg"><use href="//example.com/a.svg#x"/></svg>'))

    def test_export_keeps_the_accepted_version_and_ratio(self):
        source = self.root / "candidate.png"
        source.write_bytes(PNG_2X1)
        svg = self.root / "candidate.svg"
        svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0h1v1z"/></svg>')
        destination = self.root / "export"
        manifest = export_version(source, svg, destination, "picked", 1200, 600)
        self.assertEqual(manifest["width"], 1200)
        self.assertEqual(manifest["height"], 600)
        self.assertTrue((destination / "picked.png").exists())
        data = (destination / "picked.png").read_bytes()
        self.assertEqual(struct.unpack(">II", data[16:24]), (1200, 600))

    def test_worker_resumes_after_analysis_without_repeating_it(self):
        source = self.root / "source.png"
        source.write_bytes(PNG_2X1)
        batch_id = self.db.create_batch("Pilot", {"background": "transparent"})
        job_id = self.db.add_job(batch_id, str(source))
        self.db.claim_next_job(batch_id)
        self.db.finish_stage(job_id, "analysis", {"subjects": ["person"], "edit_instructions": "preserve"})
        worker = BatchWorker(self.db, FakePipeline(), self.root / "artifacts")
        worker.run_batch(batch_id)
        job = self.db.job(job_id)
        self.assertEqual(job["stage"], "ready")
        self.assertTrue(Path(job["candidate_path"]).exists())
        self.assertTrue(Path(job["svg_path"]).exists())

    def test_acceptance_fixes_one_version_and_export_manifest_contains_history(self):
        source = self.root / "source.png"
        source.write_bytes(PNG_2X1)
        batch_id = self.db.create_batch("Pilot", {})
        job_id = self.db.add_job(batch_id, str(source))
        BatchWorker(self.db, FakePipeline(), self.root / "artifacts").run_batch(batch_id)
        version = self.db.versions(job_id)[0]
        self.db.accept_version(job_id, version["id"])
        accepted = self.db.accepted_version(job_id)
        self.assertEqual(accepted["id"], version["id"])
        report = self.db.batch_report(batch_id)
        self.assertEqual(report["counts"]["selected"], 1)
        self.assertEqual(report["usage"]["gpt-5.6-luna"]["input_tokens"], 10)

    def test_failed_job_does_not_block_remaining_jobs(self):
        class FailFirst(FakePipeline):
            def analyze(self, original, settings):
                if original.name == "bad.png":
                    raise ValueError("bad input")
                return super().analyze(original, settings)

        batch_id = self.db.create_batch("Pilot", {})
        for name in ("bad.png", "good.png"):
            path = self.root / name
            path.write_bytes(PNG_2X1)
            self.db.add_job(batch_id, str(path))
        BatchWorker(self.db, FailFirst(), self.root / "artifacts").run_batch(batch_id)
        stages = {Path(j["original_path"]).name: j["stage"] for j in self.db.jobs(batch_id)}
        self.assertEqual(stages, {"bad.png": "failed", "good.png": "ready"})

    def test_retry_failed_resumes_from_last_completed_stage(self):
        batch_id = self.db.create_batch("Pilot", {})
        job_id = self.db.add_job(batch_id, "source.png")
        self.db.conn.execute(
            "UPDATE jobs SET stage='failed',analysis=?,candidate_path=?,error='schema error' WHERE id=?",
            (json.dumps({"subjects": ["person"]}), "candidate.png", job_id),
        )
        self.db.conn.commit()

        self.assertEqual(self.db.retry_failed(batch_id), 1)
        job = self.db.job(job_id)
        self.assertEqual(job["stage"], "generated")
        self.assertIsNone(job["error"])

    def test_manual_retry_can_resume_uncertain_generation(self):
        batch_id = self.db.create_batch("Pilot", {})
        job_id = self.db.add_job(batch_id, "source.png")
        self.db.conn.execute(
            "UPDATE jobs SET stage='generation_uncertain',analysis=?,error='invalid parameter' WHERE id=?",
            (json.dumps({"subjects": ["person"]}), job_id),
        )
        self.db.conn.commit()

        self.assertEqual(self.db.retry_failed(batch_id), 1)
        self.assertEqual(self.db.job(job_id)["stage"], "analyzed")

    def test_missing_requested_transparency_requires_attention(self):
        from PIL import Image

        source = self.root / "opaque.png"
        Image.new("RGB", (32, 32), "red").save(source)
        batch_id = self.db.create_batch("Transparent", {"background": "transparent"})
        job_id = self.db.add_job(batch_id, str(source))
        BatchWorker(self.db, FakePipeline(), self.root / "artifacts").run_batch(batch_id)
        evaluation = json.loads(self.db.job(job_id)["evaluation"])
        self.assertEqual(evaluation["category"], "requires_attention")
        self.assertEqual(evaluation["defects"][0]["criterion"], "background")

    def test_image_size_tracks_orientation_without_deforming(self):
        self.assertEqual(choose_image_size(1920, 1080), "1536x1024")
        self.assertEqual(choose_image_size(1080, 1920), "1024x1536")
        self.assertEqual(choose_image_size(1200, 1100), "1024x1024")

    def test_image_quality_allows_only_low_or_medium(self):
        self.assertEqual(choose_image_quality({"image_quality": "low"}), "low")
        self.assertEqual(choose_image_quality({"image_quality": "medium"}), "medium")
        self.assertEqual(choose_image_quality({}), "medium")
        with self.assertRaises(ValueError):
            choose_image_quality({"image_quality": "high"})

    def test_vtracer_produces_valid_vector_geometry(self):
        from PIL import Image, ImageDraw

        source = self.root / "source.png"
        image = Image.new("RGB", (32, 32), "white")
        ImageDraw.Draw(image).ellipse((5, 5, 27, 27), fill="red", outline="black")
        image.save(source)
        target = self.root / "result.svg"
        VTracerPipeline().vectorize(source, target, "detail")
        self.assertTrue(validate_svg(target.read_text()))
        rendered = self.root / "rendered.png"
        render_svg(target, rendered)
        self.assertTrue(rendered.read_bytes().startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
