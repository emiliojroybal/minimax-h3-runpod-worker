from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from h3_worker.schemas import GenerationInput  # noqa: E402
from h3_worker.model_cache import (  # noqa: E402
    missing_components,
    repository_cache_name,
    required_components,
    resolve_model_snapshot,
    snapshot_from_hub_cache,
)
from h3_worker.lora import parse_hugging_face_lora_url  # noqa: E402
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
    def test_worker_advertises_native_and_draft_canvases(self) -> None:
        handler_source = (ROOT / "handler.py").read_text(encoding="utf-8")
        self.assertIn('"short_edges": [544, 768]', handler_source)

    def test_production_image_includes_qwen_video_runtime(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertRegex(
            dockerfile,
            r"torch==2\.11\.0\s+torchvision==0\.26\.0\s+torchaudio==2\.11\.0",
        )
        self.assertIn("Qwen3VLVideoProcessor", dockerfile)
        requirements = (ROOT / "requirements-production.txt").read_text(encoding="utf-8")
        self.assertIn("kernels==0.16.0", requirements)
        adapter = (ROOT / "h3_worker" / "adapters" / "diffusers_adapter.py").read_text(encoding="utf-8")
        self.assertIn('"_flash_3_hub"', adapter)
        self.assertIn('"flash_4_hub"', adapter)
        self.assertIn('request.inference_steps + 1', adapter)
        self.assertIn('with torch.inference_mode()', adapter)

    def test_turbo_lora_contract(self) -> None:
        payload = valid_payload()
        payload["target"]["short_edge"] = 544
        payload["inference_steps"] = 4
        payload["loras"] = [{
            "id": "turbo", "name": "Turbo 4-step",
            "url": "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
            "weight": 1.0, "alpha": 8, "video_shift": 12, "audio_shift": 3,
            "compatible_mode": "base", "short_edge": 544, "recommended_steps": 4,
        }]
        request = GenerationInput.model_validate(payload)
        self.assertEqual(request.loras[0].recommended_steps, 4)

    def test_turbo_lora_rejects_wrong_resolution(self) -> None:
        payload = valid_payload()
        payload["inference_steps"] = 4
        payload["loras"] = [{
            "id": "turbo", "name": "Turbo 4-step",
            "url": "https://huggingface.co/lightx2v/Minimax-h3-Turbo/resolve/main/minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
            "weight": 1.0, "alpha": 8, "video_shift": 12, "audio_shift": 3,
            "compatible_mode": "base", "short_edge": 544, "recommended_steps": 4,
        }]
        with self.assertRaises(ValidationError):
            GenerationInput.model_validate(payload)

    def test_linked_lora_is_restricted_to_supported_hugging_face_files(self) -> None:
        repo, revision, filename = parse_hugging_face_lora_url(
            "https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/"
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
        )
        self.assertEqual(repo, "lightx2v/Minimax-h3-Turbo")
        self.assertEqual(revision, "main")
        self.assertTrue(filename.endswith(".safetensors"))
        with self.assertRaises(ValueError):
            parse_hugging_face_lora_url(
                "https://example.com/lightx2v/Minimax-h3-Turbo/blob/main/adapter.safetensors"
            )

    def test_valid_text_job(self) -> None:
        request = GenerationInput.model_validate(valid_payload())
        self.assertEqual(request.mode, "t2va")
        self.assertEqual(request.target.fps, 24)

    def test_studio_output_handoff_contract(self) -> None:
        payload = valid_payload()
        payload["output_handoff"] = {
            "object_key": "studio-handoff/job_test.mp4",
            "upload_url": "https://example.com/signed-upload",
            "content_type": "video/mp4",
        }
        request = GenerationInput.model_validate(payload)
        self.assertEqual(request.output_handoff.object_key, "studio-handoff/job_test.mp4")
        payload["output_handoff"]["object_key"] = "../../unexpected.mp4"
        with self.assertRaises(ValidationError):
            GenerationInput.model_validate(payload)

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

        width, height = aspect_dimensions("16:9", short_edge=544)
        self.assertEqual((width, height), (960, 544))
        self.assertEqual(aspect_dimensions("16:9", short_edge=768), (1344, 768))

    def test_total_media_duration_is_limited(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not exceed 15 seconds"):
            validate_total_durations([("audio", 8.0), ("audio", 8.0), ("video", 12.0)])

    def test_model_cache_resolves_main_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            hub = Path(temp)
            model_root = hub / repository_cache_name("MiniMaxAI/MiniMax-H3")
            snapshot = model_root / "snapshots" / "revision-1"
            snapshot.mkdir(parents=True)
            (snapshot / "modular_model_index.json").write_text("{}", encoding="utf-8")
            (model_root / "refs").mkdir()
            (model_root / "refs" / "main").write_text("revision-1", encoding="utf-8")
            self.assertEqual(snapshot_from_hub_cache("MiniMaxAI/MiniMax-H3", hub), snapshot)
            self.assertEqual(resolve_model_snapshot("MiniMaxAI/MiniMax-H3", hub), snapshot)

    def test_model_cache_detects_workflow_specific_transformers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            snapshot = Path(temp)
            (snapshot / "modular_model_index.json").write_text("{}", encoding="utf-8")
            for component in (
                "processor", "tokenizer", "text_encoder", "vae", "audio_vae",
                "scheduler", "audio_scheduler", "transformer",
            ):
                (snapshot / component).mkdir()
            self.assertEqual(missing_components(snapshot, "t2va"), [])
            self.assertEqual(missing_components(snapshot, "fl2va"), [])
            self.assertEqual(missing_components(snapshot, "ref2va"), ["transformer_ref"])

    def test_required_components_selects_reference_transformer(self) -> None:
        self.assertIn("transformer", required_components("t2va"))
        self.assertNotIn("transformer_ref", required_components("t2va"))
        self.assertIn("transformer_ref", required_components("ref2va"))
        self.assertNotIn("transformer", required_components("ref2va"))


if __name__ == "__main__":
    unittest.main()
