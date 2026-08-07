import hashlib
import json
import logging
import math
import os

import comfy.utils
import folder_paths
import node_helpers
import numpy as np
import torch
from PIL import Image, ImageOps

from comfy_api.latest import io

from .ltx_looping_media import (
    first_video_frame,
    media_fingerprint,
    validate_media_timeline,
)

log = logging.getLogger(__name__)

LoopingDirectorPlan = io.Custom("LTX_LOOPING_DIRECTOR_PLAN")

LTX_TIME_SCALE = 8
DEFAULT_FRAME_RATE = 24.0
DEFAULT_TOTAL_DURATION = 48.0
DEFAULT_TILE_DURATION = 10.0
DEFAULT_OVERLAP_DURATION = 2.0
DEFAULT_TARGET_HEIGHT = 1088
DEFAULT_TILE_PROMPT = (
    "The couple continues the choreography at a regular pace while the camera makes "
    "a slow orbit toward the next tile's end reference image."
)


def _parse_timeline(timeline_data):
    if isinstance(timeline_data, dict):
        data = timeline_data
    elif timeline_data:
        try:
            data = json.loads(timeline_data)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"LTXLoopingDirector: invalid timeline_data JSON: {exc}") from exc
    else:
        data = {}

    if not isinstance(data, dict):
        raise ValueError("LTXLoopingDirector: timeline_data must contain a JSON object")

    legacy_retake_mode = bool(data.get("retakeMode"))
    legacy_retake = data.get("retake") or data.get("retakeVideo")

    tile_prompts = data.get("tile_prompts", []) or []
    keyframes = data.get("keyframes", []) or []
    if not isinstance(tile_prompts, list):
        raise ValueError("LTXLoopingDirector: tile_prompts must be a JSON array")
    if not isinstance(keyframes, list):
        raise ValueError("LTXLoopingDirector: keyframes must be a JSON array")

    for index, prompt in enumerate(tile_prompts):
        if not isinstance(prompt, str):
            raise ValueError(f"LTXLoopingDirector: tile_prompts[{index}] must be a string")

    # Version 1 had no media lanes. Keep old saved workflows executable while the
    # editor upgrades them to version 2 on the next edit.
    if int(data.get("version", 1) or 1) < 2:
        data = {
            **data,
            "version": 2,
            "video_segments": [],
            "ic_segments": [],
            "audio_segments": [],
            "use_custom_audio": False,
            "inpaint_audio": True,
            "use_ic_video_audio": False,
            "retake_mode": legacy_retake_mode,
            "retake": legacy_retake,
            "ic_settings": {},
        }

    if data.get("retakeMode") or data.get("retakeVideo"):
        data = {
            **data,
            "retake_mode": bool(data.get("retakeMode")),
            "retake": data.get("retake") or data.get("retakeVideo"),
        }

    return data, tile_prompts, keyframes


def _aligned_frames(seconds, frame_rate, minimum):
    frames = round(float(seconds) * float(frame_rate) / LTX_TIME_SCALE) * LTX_TIME_SCALE
    return max(int(minimum), frames)


def _temporal_chunks(frame_count, temporal_tile_size, temporal_overlap):
    latent_frames = (frame_count - 1) // LTX_TIME_SCALE + 1
    latent_tile_size = temporal_tile_size // LTX_TIME_SCALE
    latent_overlap = temporal_overlap // LTX_TIME_SCALE
    latent_stride = latent_tile_size - latent_overlap
    tile_count = max(1, math.ceil((latent_frames - latent_overlap) / latent_stride))
    return [
        (
            tile_index * latent_stride,
            min(tile_index * latent_stride + latent_tile_size, latent_frames),
        )
        for tile_index in range(tile_count)
    ]


def _default_reference_frames(frame_count, temporal_tile_size, temporal_overlap):
    chunks = _temporal_chunks(frame_count, temporal_tile_size, temporal_overlap)
    final_index = ((frame_count - 1) // LTX_TIME_SCALE) * LTX_TIME_SCALE
    margin = temporal_overlap // 2
    indices = [0]
    tile_stride = temporal_tile_size - temporal_overlap
    for tile_index in range(len(chunks)):
        reference_index = min(
            tile_index * tile_stride + temporal_tile_size - margin,
            final_index,
        )
        reference_index -= reference_index % LTX_TIME_SCALE
        if reference_index not in indices:
            indices.append(reference_index)
    return indices


def _calculate_schedule(frame_rate, total_duration, tile_duration, overlap_duration):
    frame_rate = float(frame_rate)
    total_duration = float(total_duration)
    tile_duration = float(tile_duration)
    overlap_duration = float(overlap_duration)
    if frame_rate <= 0 or total_duration <= 0 or tile_duration <= 0 or overlap_duration < 0:
        raise ValueError("LTXLoopingDirector: timing values must be positive")

    frame_count = max(
        LTX_TIME_SCALE + 1,
        math.floor((total_duration * frame_rate - 1) / LTX_TIME_SCALE)
        * LTX_TIME_SCALE
        + 1,
    )
    tile_size = min(_aligned_frames(tile_duration, frame_rate, 24), 1000)
    overlap = _aligned_frames(overlap_duration, frame_rate, 16)
    overlap = min(overlap, 80, tile_size - LTX_TIME_SCALE)
    if frame_count > 10001:
        raise ValueError("LTXLoopingDirector: the calculated frame_count exceeds 10001")

    chunks = _temporal_chunks(frame_count, tile_size, overlap)
    return (
        frame_count,
        tile_size,
        overlap,
        chunks,
        _default_reference_frames(frame_count, tile_size, overlap),
    )


def _validate_timing(frame_rate, total_duration, tile_duration, overlap_duration):
    return _calculate_schedule(
        frame_rate,
        total_duration,
        tile_duration,
        overlap_duration,
    )


def _parse_keyframes(keyframes, frame_count):
    parsed = []
    for order, keyframe in enumerate(keyframes):
        if not isinstance(keyframe, dict):
            raise ValueError(f"LTXLoopingDirector: keyframes[{order}] must be a JSON object")

        frame = keyframe.get("frame")
        image_file = keyframe.get("imageFile")
        if isinstance(frame, bool) or not isinstance(frame, (int, float)):
            raise ValueError(f"LTXLoopingDirector: keyframes[{order}].frame must be an integer")
        if int(frame) != frame:
            raise ValueError(f"LTXLoopingDirector: keyframes[{order}].frame must be an integer")
        frame = int(frame)
        if frame < 0 or frame >= frame_count:
            raise ValueError(
                f"LTXLoopingDirector: keyframe {order} is outside frame_count "
                f"({frame} not in 0..{frame_count - 1})"
            )
        if frame % LTX_TIME_SCALE:
            raise ValueError(
                f"LTXLoopingDirector: keyframe {order} is not aligned to an 8-frame grid"
            )
        if not isinstance(image_file, str) or not image_file:
            raise ValueError(f"LTXLoopingDirector: keyframes[{order}].imageFile is required")

        parsed.append((frame, order, image_file))

    parsed.sort(key=lambda item: (item[0], item[1]))
    return parsed


def _resolve_keyframe(image_file):
    try:
        path = folder_paths.get_annotated_filepath(image_file)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LTXLoopingDirector: invalid keyframe path {image_file!r}") from exc
    if not os.path.isfile(path):
        raise ValueError(f"LTXLoopingDirector: keyframe file not found: {image_file}")
    return path


def _load_keyframe(image_file):
    path = _resolve_keyframe(image_file)
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, ValueError) as exc:
        raise ValueError(f"LTXLoopingDirector: could not read keyframe {image_file}: {exc}") from exc

    if array.ndim != 3 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"LTXLoopingDirector: keyframe {image_file} has no pixels")
    return torch.from_numpy(array).unsqueeze(0)


def _aligned_dimension(value, multiple=64, minimum=64):
    return max(minimum, round(float(value) / multiple) * multiple)


def _resize_reference(image, width, height):
    return (
        comfy.utils.common_upscale(
            image[:1].movedim(-1, 1),
            width,
            height,
            "lanczos",
            crop="disabled",
        )
        .movedim(1, -1)
        .clamp(0, 1)
    )


def _normalize_keyframe(image, width, height):
    if image.shape[1] == height and image.shape[2] == width:
        return image
    return (
        comfy.utils.common_upscale(
            image.movedim(-1, 1), width, height, "bilinear", crop="center"
        )
        .movedim(1, -1)
        .clamp(0, 1)
    )


class LTXLoopingDirector(io.ComfyNode):
    """Prepare per-tile conditioning, reference images, and looping dimensions."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLoopingDirector",
            display_name="LTX Looping Director",
            category="WhatDreamsCost",
            description=(
                "Creates one prompt conditioning per looping sampler tile and a batch "
                "of static image keyframes. Timing is entered in seconds; the frame "
                "schedule and output dimensions are derived automatically."
            ),
            inputs=[
                io.Clip.Input("clip", tooltip="CLIP used to encode the global and tile prompts."),
                io.String.Input(
                    "global_prompt",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="Common prompt prepended to every temporal tile.",
                ),
                io.Float.Input(
                    "frame_rate",
                    default=DEFAULT_FRAME_RATE,
                    min=1.0,
                    max=240.0,
                    step=0.01,
                    optional=True,
                    tooltip="Video frame rate used to convert the duration controls to LTX frame counts.",
                ),
                io.Float.Input(
                    "total_duration",
                    default=DEFAULT_TOTAL_DURATION,
                    min=0.1,
                    max=3600.0,
                    step=0.1,
                    optional=True,
                    tooltip="Total output duration in seconds.",
                ),
                io.Float.Input(
                    "tile_duration",
                    default=DEFAULT_TILE_DURATION,
                    min=0.1,
                    max=3600.0,
                    step=0.1,
                    optional=True,
                    tooltip="Temporal tile duration in seconds.",
                ),
                io.Float.Input(
                    "overlap_duration",
                    default=DEFAULT_OVERLAP_DURATION,
                    min=0.1,
                    max=3600.0,
                    step=0.1,
                    optional=True,
                    tooltip="Temporal overlap between adjacent tiles in seconds.",
                ),
                io.String.Input(
                    "timeline_data",
                    default=json.dumps({
                        "version": 2,
                        "tile_prompts": [DEFAULT_TILE_PROMPT] * 6,
                        "keyframes": [],
                        "video_segments": [],
                        "ic_segments": [],
                        "audio_segments": [],
                        "use_custom_audio": False,
                        "inpaint_audio": True,
                        "use_ic_video_audio": False,
                        "retake_mode": False,
                        "retake": None,
                        "ic_settings": {},
                    }),
                    optional=True,
                    tooltip="Frontend-managed prompts, keyframes, and optional media.",
                ),
                io.Int.Input(
                    "target_height",
                    default=DEFAULT_TARGET_HEIGHT,
                    min=32,
                    max=16384,
                    step=32,
                    optional=True,
                    tooltip=(
                        "Final output height. The frame-0 start image supplies the aspect "
                        "ratio; width is calculated and aligned automatically."
                    ),
                ),
                io.Int.Input(
                    "reference_keyframe_index",
                    default=0,
                    min=0,
                    max=10000,
                    step=1,
                    optional=True,
                    tooltip=(
                        "Timeline slot used for the external reference image. The editor "
                        "shows this as a readable keyframe selector."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="per_tile_conditionings"),
                io.Image.Output(display_name="cond_images"),
                io.String.Output(display_name="cond_image_indices"),
                io.Int.Output(display_name="temporal_tile_size"),
                io.Int.Output(display_name="temporal_overlap"),
                io.Int.Output(display_name="frame_count"),
                io.Float.Output(display_name="frame_rate"),
                io.String.Output(display_name="global_prompt"),
                io.Image.Output(display_name="start_image"),
                io.Int.Output(display_name="stage_1_width"),
                io.Int.Output(display_name="stage_1_height"),
                io.Image.Output(
                    display_name="reference_image",
                    tooltip="Image selected by reference_keyframe_index for external use.",
                ),
                LoopingDirectorPlan.Output(
                    display_name="director_plan",
                    tooltip="Execution plan consumed by LTX Looping Director Sampler.",
                ),
            ],
        )

    @classmethod
    def _validate(cls, timeline_data, frame_rate, total_duration, tile_duration, overlap_duration):
        frame_count, tile_size, overlap, chunks, _ = _validate_timing(
            frame_rate,
            total_duration,
            tile_duration,
            overlap_duration,
        )
        data, tile_prompts, keyframes = _parse_timeline(timeline_data)
        if len(tile_prompts) != len(chunks):
            raise ValueError(
                "LTXLoopingDirector: expected "
                f"{len(chunks)} tile prompts, received {len(tile_prompts)}"
            )
        parsed = _parse_keyframes(keyframes, frame_count)
        if parsed and parsed[0][0] != 0:
            raise ValueError("LTXLoopingDirector: the first keyframe must be at frame 0")
        validate_media_timeline(data, frame_count, chunks)
        return data, tile_prompts, keyframes, chunks, frame_count, tile_size, overlap

    @classmethod
    def validate_inputs(cls, **kwargs):
        try:
            timeline_data = kwargs.get("timeline_data", "")
            cls._validate(
                timeline_data,
                kwargs.get("frame_rate", DEFAULT_FRAME_RATE),
                kwargs.get("total_duration", DEFAULT_TOTAL_DURATION),
                kwargs.get("tile_duration", DEFAULT_TILE_DURATION),
                kwargs.get("overlap_duration", DEFAULT_OVERLAP_DURATION),
            )
            frame_count = _calculate_schedule(
                kwargs.get("frame_rate", DEFAULT_FRAME_RATE),
                kwargs.get("total_duration", DEFAULT_TOTAL_DURATION),
                kwargs.get("tile_duration", DEFAULT_TILE_DURATION),
                kwargs.get("overlap_duration", DEFAULT_OVERLAP_DURATION),
            )[0]
            parsed = _parse_keyframes(_parse_timeline(timeline_data)[2], frame_count)
            if parsed and not 0 <= int(kwargs.get("reference_keyframe_index", 0)) < len(parsed):
                raise ValueError(
                    "LTXLoopingDirector: reference_keyframe_index must select an existing "
                    f"timeline keyframe slot (0..{len(parsed) - 1})"
                )
            for _, _, image_file in parsed:
                _resolve_keyframe(image_file)
        except (TypeError, ValueError) as exc:
            return str(exc)
        return True

    @classmethod
    def fingerprint_inputs(cls, timeline_data="", **kwargs):
        digest = hashlib.sha256()
        digest.update(str(timeline_data or "").encode("utf-8"))
        digest.update(str(kwargs.get("global_prompt", "")).encode("utf-8"))
        digest.update(str(kwargs.get("frame_rate", DEFAULT_FRAME_RATE)).encode("utf-8"))
        digest.update(str(kwargs.get("total_duration", DEFAULT_TOTAL_DURATION)).encode("utf-8"))
        digest.update(str(kwargs.get("tile_duration", DEFAULT_TILE_DURATION)).encode("utf-8"))
        digest.update(str(kwargs.get("overlap_duration", DEFAULT_OVERLAP_DURATION)).encode("utf-8"))
        digest.update(str(kwargs.get("target_height", DEFAULT_TARGET_HEIGHT)).encode("utf-8"))
        digest.update(str(kwargs.get("reference_keyframe_index", 0)).encode("utf-8"))
        try:
            frame_count = _calculate_schedule(
                kwargs.get("frame_rate", DEFAULT_FRAME_RATE),
                kwargs.get("total_duration", DEFAULT_TOTAL_DURATION),
                kwargs.get("tile_duration", DEFAULT_TILE_DURATION),
                kwargs.get("overlap_duration", DEFAULT_OVERLAP_DURATION),
            )[0]
            parsed = _parse_keyframes(_parse_timeline(timeline_data)[2], frame_count)
        except (TypeError, ValueError):
            return digest.hexdigest()

        for _, _, image_file in parsed:
            digest.update(image_file.encode("utf-8"))
            try:
                path = _resolve_keyframe(image_file)
                with open(path, "rb") as image:
                    for block in iter(lambda: image.read(1024 * 1024), b""):
                        digest.update(block)
            except (OSError, TypeError, ValueError):
                digest.update(b"<missing>")
        try:
            for entry in media_fingerprint(_parse_timeline(timeline_data)[0]):
                digest.update(repr(entry).encode("utf-8"))
        except (OSError, TypeError, ValueError):
            digest.update(b"<missing-media>")
        return digest.hexdigest()

    @staticmethod
    def _encode(clip, text, frame_rate):
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        return node_helpers.conditioning_set_values(
            conditioning, {"frame_rate": float(frame_rate)}
        )

    @classmethod
    def _build_conditionings(cls, clip, global_prompt, tile_prompts, chunks, frame_rate):
        global_prompt = (global_prompt or "").strip()
        cache = {}
        conditionings = []
        for tile_index in range(len(chunks)):
            parts = [global_prompt, tile_prompts[tile_index].strip()]
            text = "\n\n".join(part for part in parts if part) or " "
            if text not in cache:
                cache[text] = cls._encode(clip, text, frame_rate)
            conditionings.append(cache[text])
        return conditionings

    @classmethod
    def _prepare_keyframes(cls, keyframes, frame_count):
        parsed = _parse_keyframes(keyframes, frame_count)
        if not parsed:
            return parsed, [], None, None
        if parsed[0][0] != 0:
            raise ValueError("LTXLoopingDirector: the first keyframe must be at frame 0")

        raw_images = [_load_keyframe(image_file) for _, _, image_file in parsed]
        height, width = raw_images[0].shape[1:3]
        cond_images = torch.cat(
            [_normalize_keyframe(image, width, height) for image in raw_images], dim=0
        )
        start_image = raw_images[0][:1]
        return parsed, raw_images, cond_images, start_image

    @classmethod
    def _build_keyframes(cls, keyframes, frame_count):
        parsed, _, cond_images, start_image = cls._prepare_keyframes(keyframes, frame_count)
        indices = ",".join(str(frame) for frame, _, _ in parsed)
        if not parsed:
            return None, "", None
        return cond_images, indices, start_image

    @classmethod
    def _build_keyframes_and_reference(
        cls, keyframes, frame_count, target_height, reference_keyframe_index
    ):
        parsed, raw_images, cond_images, start_image = cls._prepare_keyframes(keyframes, frame_count)
        if not parsed:
            return None, "", None, 0, 0, None

        reference_keyframe_index = int(reference_keyframe_index)
        selected = next(
            (
                image
                for (_, order, _), image in zip(parsed, raw_images)
                if order == reference_keyframe_index
            ),
            None,
        )
        if selected is None:
            raise ValueError(
                "LTXLoopingDirector: reference_keyframe_index must select an existing "
                f"timeline keyframe slot (0..{len(keyframes) - 1})"
            )

        source_height, source_width = start_image.shape[1:3]
        final_height = _aligned_dimension(target_height, 64, 64)
        final_width = _aligned_dimension(
            final_height * source_width / source_height,
            64,
            64,
        )
        reference_image = _resize_reference(selected, final_width, final_height)
        return (
            cond_images,
            ",".join(str(frame) for frame, _, _ in parsed),
            start_image,
            final_width // 2,
            final_height // 2,
            reference_image,
        )

    @classmethod
    def execute(
        cls,
        clip,
        global_prompt="",
        frame_rate=DEFAULT_FRAME_RATE,
        total_duration=DEFAULT_TOTAL_DURATION,
        tile_duration=DEFAULT_TILE_DURATION,
        overlap_duration=DEFAULT_OVERLAP_DURATION,
        timeline_data="",
        target_height=DEFAULT_TARGET_HEIGHT,
        reference_keyframe_index=0,
    ):
        (
            data,
            tile_prompts,
            keyframes,
            chunks,
            frame_count,
            temporal_tile_size,
            temporal_overlap,
        ) = cls._validate(
            timeline_data,
            frame_rate,
            total_duration,
            tile_duration,
            overlap_duration,
        )
        output_global_prompt = global_prompt or ""
        global_prompt = output_global_prompt.strip()
        positive = cls._encode(clip, global_prompt or " ", frame_rate)
        conditionings = cls._build_conditionings(
            clip,
            global_prompt,
            tile_prompts,
            chunks,
            frame_rate,
        )
        (
            cond_images,
            cond_indices,
            start_image,
            stage_1_width,
            stage_1_height,
            reference_image,
        ) = cls._build_keyframes_and_reference(
            keyframes,
            frame_count,
            int(target_height or DEFAULT_TARGET_HEIGHT),
            int(reference_keyframe_index or 0),
        )

        if start_image is None:
            aspect_segment = next(
                iter(
                    (data.get("video_segments", []) or [])
                    + (data.get("ic_segments", []) or [])
                    + ([data.get("retake")] if data.get("retake_mode") and data.get("retake") else [])
                ),
                None,
            )
            if aspect_segment is not None:
                filename = (
                    aspect_segment.get("imageFile")
                    or aspect_segment.get("videoFile")
                    or aspect_segment.get("fileName")
                )
                media_start = first_video_frame(filename)
                source_height, source_width = media_start.shape[1:3]
                final_height = _aligned_dimension(target_height or DEFAULT_TARGET_HEIGHT, 64, 64)
                final_width = _aligned_dimension(
                    final_height * source_width / source_height,
                    64,
                    64,
                )
                start_image = media_start
                stage_1_width = final_width // 2
                stage_1_height = final_height // 2
                reference_image = _resize_reference(media_start, final_width, final_height)
            else:
                final_height = _aligned_dimension(target_height or DEFAULT_TARGET_HEIGHT, 64, 64)
                final_width = _aligned_dimension(final_height * 16 / 9, 64, 64)
                stage_1_width = final_width // 2
                stage_1_height = final_height // 2

        plan = {
            "version": 1,
            "frame_rate": float(frame_rate),
            "frame_count": frame_count,
            "temporal_tile_size": temporal_tile_size,
            "temporal_overlap": temporal_overlap,
            "chunks": [
                {"tile": index, "start": start, "end": end}
                for index, (start, end) in enumerate(chunks)
            ],
            "stage_1_width": stage_1_width,
            "stage_1_height": stage_1_height,
            "target_width": stage_1_width * 2 if stage_1_width else 0,
            "target_height": stage_1_height * 2 if stage_1_height else 0,
            "media": data,
            "reference_keyframe_index": int(reference_keyframe_index or 0),
        }
        log.info(
            "[LTXLoopingDirector] %d tile conditionings, %d keyframes, frame_count=%d",
            len(conditionings),
            0 if cond_images is None else cond_images.shape[0],
            frame_count,
        )
        return io.NodeOutput(
            positive,
            conditionings,
            cond_images,
            cond_indices,
            temporal_tile_size,
            temporal_overlap,
            frame_count,
            float(frame_rate),
            output_global_prompt,
            start_image,
            stage_1_width,
            stage_1_height,
            reference_image,
            plan,
        )


NODE_CLASS_MAPPINGS = {
    "LTXLoopingDirector": LTXLoopingDirector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLoopingDirector": "LTX Looping Director",
}
