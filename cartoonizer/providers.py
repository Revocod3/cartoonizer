from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from PIL import Image

from .models import Analysis, Evaluation
from .prompts import ANALYSIS_SYSTEM, EVALUATION_SYSTEM


def choose_image_size(width: int, height: int) -> str:
    ratio = width / height
    if ratio > 1.2:
        return "1536x1024"
    if ratio < 1 / 1.2:
        return "1024x1536"
    return "1024x1024"


def choose_image_quality(instructions: dict[str, Any]) -> str:
    quality = instructions.get("image_quality", "medium")
    if quality not in ("low", "medium"):
        raise ValueError("Image quality must be low or medium")
    return quality


def image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode()}"


def usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }


class VTracerPipeline:
    def vectorize(self, candidate: Path, destination: Path, mode: str) -> Path:
        import vtracer

        options = {
            "detail": dict(filter_speckle=2, color_precision=8, layer_difference=8, corner_threshold=45, length_threshold=3.5, path_precision=3),
            "curves": dict(filter_speckle=6, color_precision=6, layer_difference=16, corner_threshold=70, length_threshold=8.0, path_precision=2),
        }["curves" if mode == "curves" else "detail"]
        vtracer.convert_image_to_svg_py(str(candidate), str(destination), colormode="color", hierarchical="stacked", mode="spline", **options)
        return destination


class OpenAIPipeline(VTracerPipeline):
    def __init__(self, api_key: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)

    def analyze(self, original: Path, settings: dict[str, Any]):
        response = self.client.responses.parse(
            model="gpt-5.6-luna",
            reasoning={"effort": "none"},
            instructions=ANALYSIS_SYSTEM,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": "Analyse this original and prepare precise edit instructions. Effective settings:\n" + json.dumps(settings, ensure_ascii=False)},
                {"type": "input_image", "image_url": image_data_url(original), "detail": "high"},
            ]}],
            text_format=Analysis,
        )
        if response.output_parsed is None:
            raise RuntimeError("Luna returned no structured analysis")
        return response.output_parsed.model_dump(), usage_dict(response)

    def generate(self, original: Path, instructions: dict[str, Any], destination: Path, previous: Path | None = None):
        with original.open("rb") as source:
            files: list[Any] = [source]
            previous_handle = previous.open("rb") if previous else None
            if previous_handle:
                files.append(previous_handle)
            try:
                with Image.open(original) as image:
                    size = choose_image_size(*image.size)
                background = "transparent" if instructions.get("background") == "transparent" else "auto"
                response = self.client.images.edit(
                    model="gpt-image-2",
                    image=files,
                    prompt=self._generation_prompt(instructions, previous is not None),
                    n=1,
                    size=size,
                    quality=choose_image_quality(instructions),
                    output_format="png",
                    background=background,
                )
            finally:
                if previous_handle:
                    previous_handle.close()
        if not response.data or not response.data[0].b64_json:
            raise RuntimeError("GPT Image 2 returned no image")
        destination.write_bytes(base64.b64decode(response.data[0].b64_json))
        return destination, {"image_attempts": 1}

    def evaluate(self, original: Path, candidate: Path, instructions: dict[str, Any]):
        response = self.client.responses.parse(
            model="gpt-5.6-terra",
            reasoning={"effort": "none"},
            instructions=EVALUATION_SYSTEM,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": "Evaluate candidate against original and instructions:\n" + json.dumps(instructions, ensure_ascii=False)},
                {"type": "input_image", "image_url": image_data_url(original), "detail": "high"},
                {"type": "input_image", "image_url": image_data_url(candidate), "detail": "high"},
            ]}],
            text_format=Evaluation,
        )
        if response.output_parsed is None:
            raise RuntimeError("Terra returned no structured evaluation")
        return response.output_parsed.model_dump(), usage_dict(response)

    def evaluate_vector(self, candidate: Path, rendered: Path):
        response = self.client.responses.parse(
            model="gpt-5.6-terra",
            reasoning={"effort": "none"},
            instructions="Compare the SVG render to the raster candidate. Report only loss or deformation introduced by vectorization.",
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": "First image is the raster candidate; second is the rendered SVG."},
                {"type": "input_image", "image_url": image_data_url(candidate), "detail": "high"},
                {"type": "input_image", "image_url": image_data_url(rendered), "detail": "high"},
            ]}],
            text_format=Evaluation,
        )
        if response.output_parsed is None:
            raise RuntimeError("Terra returned no SVG evaluation")
        return response.output_parsed.model_dump(), usage_dict(response)

    @staticmethod
    def _generation_prompt(instructions: dict[str, Any], correction: bool) -> str:
        prefix = "Correct the supplied candidate using the original as the fidelity reference." if correction else "Redraw the supplied original."
        return prefix + " Preserve visible identity, expression, pose, composition, colours, text and required details. Do not invent cropped content. Return a clean PNG. Instructions:\n" + json.dumps(instructions, ensure_ascii=False)
