from __future__ import annotations

from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator


class ReferenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=160)
    kind: Literal["image", "video", "audio"]
    name: str = Field(min_length=1, max_length=200)
    url: AnyHttpUrl
    role: Literal["reference", "first_frame", "last_frame"] = "reference"
    elementId: str | None = Field(default=None, max_length=160)
    variantId: str | None = Field(default=None, max_length=160)


class GenerationTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    short_edge: Literal[768] = 768
    aspect_ratio: Literal["21:9", "16:9", "4:3", "1:1", "3:4", "9:16", "auto"] = "16:9"
    duration_seconds: int = Field(default=8, ge=5, le=15)
    fps: Literal[24] = 24


class GenerationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    operation: Literal["generate", "health", "capabilities"] = "generate"
    schema_version: Literal["1.0"] = "1.0"
    client_job_id: str = Field(default="health", min_length=1, max_length=160)
    prompt: str = Field(default="health", min_length=1, max_length=12_000)
    resolved_prompt: str = Field(default="health", min_length=1, max_length=24_000)
    mode: Literal["t2va", "fl2va", "ref2va"] = "t2va"
    references: list[ReferenceInput] = Field(default_factory=list, max_length=12)
    target: GenerationTarget = Field(default_factory=GenerationTarget)
    seed: int = Field(default=0, ge=0, le=9_223_372_036_854_775_807)
    inference_steps: int = Field(default=30, ge=2, le=80)
    generate_audio: Literal[True] = True

    @model_validator(mode="after")
    def validate_reference_contract(self) -> "GenerationInput":
        if self.operation != "generate":
            return self

        images = sum(reference.kind == "image" for reference in self.references)
        videos = sum(reference.kind == "video" for reference in self.references)
        audio = sum(reference.kind == "audio" for reference in self.references)
        if images > 9 or videos > 3 or audio > 3:
            raise ValueError("Reference limits are 9 images, 3 videos, and 3 audio clips.")

        keyframes = [reference for reference in self.references if reference.role != "reference"]
        omni = [reference for reference in self.references if reference.role == "reference"]
        if any(reference.kind != "image" for reference in keyframes):
            raise ValueError("First and last frames must be images.")
        if len([reference for reference in keyframes if reference.role == "first_frame"]) > 1:
            raise ValueError("Only one first frame is allowed.")
        if len([reference for reference in keyframes if reference.role == "last_frame"]) > 1:
            raise ValueError("Only one last frame is allowed.")
        expected_mode = "fl2va" if keyframes else "ref2va" if omni else "t2va"
        if self.mode != expected_mode:
            raise ValueError(f"Mode {self.mode} does not match the supplied references; expected {expected_mode}.")
        if keyframes and omni:
            raise ValueError("Keyframes and omni-references use different H3 checkpoints and cannot be mixed.")
        return self


class ErrorBody(BaseModel):
    code: str
    message: str
    details: object | None = None


class GenerationResult(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    client_job_id: str
    status: Literal["completed", "failed"]
    mode: str
    seed: int
    output_url: str | None = None
    output_path: str | None = None
    storage: str | None = None
    duration_seconds: float | None = None
    elapsed_seconds: float
    error: ErrorBody | None = None
