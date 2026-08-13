from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3_worker.schemas import GenerationInput  # noqa: E402
from h3_worker.utils import aspect_dimensions, h3_num_frames, validate_total_durations  # noqa: E402
from pydantic import ValidationError  # noqa: E402


def valid_payload() -> dict:
    return {
        "schema_version": "1.0",
        "client_job_id": "job_test",
        "prompt": "A fox crosses a snowy clearing.",
        "resolved_prompt": "integrated_multimodal_description: A fox crosses a snowy clearing.",
        "mode": "t2va",
        "references": [],
        "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 8, "fps": 24},
        "seed": 42,
        "inference_steps": 30,
        "generate_audio": True,
    }


class ContractTests(unittest.TestCase):
    def test_valid_text_job(self) -> None:
        request = GenerationInput.model_validate(valid_payload())
        self.assertEqual(request.mode, "t2va")
        self.assertEqual(request.target.fps, 24)

    def test_mode_must_match_keyframe(self) -> None:
        payload = valid_payload()
        payload["references"] = [{
            "id": "first", "kind": "image", "name": "first.png",
            "url": "https://example.com/first.png", "role": "first_frame",
        }]
        with self.assertRaises(ValidationError):
            GenerationInput.model_validate(payload)

    def test_reference_limits(self) -> None:
        payload = valid_payload()
        payload["mode"] = "ref2va"
        payload["references"] = [
            {"id": f"image-{index}", "kind": "image", "name": f"{index}.png", "url": f"https://example.com/{index}.png", "role": "reference"}
            for index in range(10)
        ]
        with self.assertRaises(ValidationError):
            GenerationInput.model_validate(payload)

    def test_h3_frame_lattice(self) -> None:
        for duration in range(5, 16):
            frames = h3_num_frames(duration)
            self.assertEqual((frames - 5) % 17, 0)
            self.assertGreaterEqual(frames, duration * 24)

    def test_dimensions_are_pipeline_aligned(self) -> None:
        for ratio in ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"):
            width, height = aspect_dimensions(ratio)
            self.assertEqual(width % 32, 0)
            self.assertEqual(height % 32, 0)
            self.assertEqual(min(width, height), 768)

    def test_total_media_duration_is_limited(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not exceed 15 seconds"):
            validate_total_durations([("audio", 8.0), ("audio", 8.0), ("video", 12.0)])


if __name__ == "__main__":
    unittest.main()
