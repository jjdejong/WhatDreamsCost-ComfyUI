import hashlib
import json
import logging
import os

import comfy.utils
import folder_paths
import node_helpers
import numpy as np
import torch
from PIL import Image, ImageOps

from comfy_api.latest import io

from .ltx_director import _compress_image, _resize_image

log = logging.getLogger(__name__)

LTX_TIME_SCALE = 8


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

    tile_prompts = data.get("tile_prompts", [])
    keyframes = data.get("keyframes", [])
    if tile_prompts is None:
        tile_prompts = []
    if keyframes is None:
        keyframes = []
    if not isinstance(tile_prompts, list):
        raise ValueError("LTXLoopingDirector: tile_prompts must be a JSON array")
    if not isinstance(keyframes, list):
        raise ValueError("LTXLoopingDirector: keyframes must be a JSON array")

    for index, prompt in enumerate(tile_prompts):
        if not isinstance(prompt, str):
            raise ValueError(
                f"LTXLoopingDirector: tile_prompts[{index}] must be a string"
            )

    return data, tile_prompts, keyframes


def _temporal_chunks(frame_count, temporal_tile_size, temporal_overlap):
    latent_frames = (frame_count - 1) // LTX_TIME_SCALE + 1
    tile_frames = temporal_tile_size // LTX_TIME_SCALE
    overlap_frames = temporal_overlap // LTX_TIME_SCALE
    step = tile_frames - overlap_frames

    chunks = []
    for start, end in zip(
        range(0, latent_frames + tile_frames - overlap_frames, step),
        range(tile_frames, latent_frames + tile_frames - overlap_frames, step),
    ):
        chunks.append((start, min(end, latent_frames)))
    return chunks


def _validate_timing(frame_count, temporal_tile_size, temporal_overlap):
    frame_count = int(frame_count)
    temporal_tile_size = int(temporal_tile_size)
    temporal_overlap = int(temporal_overlap)

    if frame_count < 9 or (frame_count - 1) % LTX_TIME_SCALE:
        raise ValueError(
            "LTXLoopingDirector: frame_count must be 8n+1 and at least 9"
        )
    if temporal_tile_size < LTX_TIME_SCALE or temporal_tile_size % LTX_TIME_SCALE:
        raise ValueError(
            "LTXLoopingDirector: temporal_tile_size must be a positive multiple of 8"
        )
    if temporal_overlap < 0 or temporal_overlap % LTX_TIME_SCALE:
        raise ValueError(
            "LTXLoopingDirector: temporal_overlap must be a non-negative multiple of 8"
        )
    if temporal_overlap >= temporal_tile_size:
        raise ValueError(
            "LTXLoopingDirector: temporal_overlap must be smaller than temporal_tile_size"
        )

    chunks = _temporal_chunks(frame_count, temporal_tile_size, temporal_overlap)
    if not chunks:
        raise ValueError(
            "LTXLoopingDirector: the clip is too short for the selected tile size and overlap"
        )
    return chunks


def _parse_keyframes(keyframes, frame_count):
    parsed = []
    for order, keyframe in enumerate(keyframes):
        if not isinstance(keyframe, dict):
            raise ValueError(
                f"LTXLoopingDirector: keyframes[{order}] must be a JSON object"
            )

        frame = keyframe.get("frame")
        image_file = keyframe.get("imageFile")
        if isinstance(frame, bool) or not isinstance(frame, (int, float)):
            raise ValueError(
                f"LTXLoopingDirector: keyframes[{order}].frame must be an integer"
            )
        if int(frame) != frame:
            raise ValueError(
                f"LTXLoopingDirector: keyframes[{order}].frame must be an integer"
            )
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
            raise ValueError(
                f"LTXLoopingDirector: keyframes[{order}].imageFile is required"
            )

        parsed.append((frame, order, image_file))

    parsed.sort(key=lambda item: (item[0], item[1]))
    return parsed


def _resolve_keyframe(image_file):
    try:
        path = folder_paths.get_annotated_filepath(image_file)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"LTXLoopingDirector: invalid keyframe path {image_file!r}"
        ) from exc
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
        raise ValueError(
            f"LTXLoopingDirector: could not read keyframe {image_file}: {exc}"
        ) from exc

    if array.ndim != 3 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"LTXLoopingDirector: keyframe {image_file} has no pixels")
    return torch.from_numpy(array).unsqueeze(0)


def _snap(value, divisible_by):
    return max(divisible_by, (int(value) // divisible_by) * divisible_by)


def _process_keyframe(image, custom_width, custom_height, resize_method, divisible_by, img_compression):
    source_h, source_w = image.shape[1:3]
    if custom_width > 0 and custom_height > 0:
        image = _resize_image(
            image, custom_width, custom_height, resize_method, divisible_by
        )
    elif custom_width > 0:
        target_w = _snap(custom_width, divisible_by)
        target_h = _snap(int(source_h * target_w / source_w), divisible_by)
        image = _resize_image(image, target_w, target_h, "stretch to fit", divisible_by)
    elif custom_height > 0:
        target_h = _snap(custom_height, divisible_by)
        target_w = _snap(int(source_w * target_h / source_h), divisible_by)
        image = _resize_image(image, target_w, target_h, "stretch to fit", divisible_by)
    else:
        image = _resize_image(
            image, source_w, source_h, "maintain aspect ratio", divisible_by
        )

    if img_compression > 0:
        image = _compress_image(image, img_compression)
    return image


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
    """Prepare native per-tile conditioning and keyframes for LTXVLoopingSampler."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLoopingDirector",
            display_name="LTX Looping Director",
            category="WhatDreamsCost",
            description=(
                "Creates one prompt conditioning per looping sampler tile and a batch "
                "of static image keyframes."
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
                io.String.Input(
                    "first_tile_prompt",
                    multiline=True,
                    default="",
                    optional=True,
                    tooltip="Additional prompt text used only for the first temporal tile.",
                ),
                io.Int.Input(
                    "frame_count",
                    default=241,
                    min=9,
                    max=10001,
                    step=8,
                    tooltip="Pixel-frame clip length. Must be 8n+1 and match the latent shells.",
                ),
                io.Int.Input(
                    "temporal_tile_size",
                    default=240,
                    min=8,
                    max=1000,
                    step=8,
                    tooltip="Temporal tile size in pixel frames. Match both looping samplers.",
                ),
                io.Int.Input(
                    "temporal_overlap",
                    default=64,
                    min=0,
                    max=992,
                    step=8,
                    tooltip="Temporal overlap in pixel frames. Match both looping samplers.",
                ),
                io.Float.Input(
                    "frame_rate",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=0.01,
                    tooltip="Frame rate stored on each conditioning.",
                ),
                io.String.Input(
                    "timeline_data",
                    default='{"version":1,"tile_prompts":["",""],"keyframes":[]}',
                    tooltip="Frontend-managed tile prompts and image keyframes.",
                ),
                io.Int.Input(
                    "custom_width",
                    default=0,
                    min=0,
                    max=8192,
                    step=1,
                    optional=True,
                    advanced=True,
                    tooltip="Target keyframe width. Zero keeps the first image aspect/size.",
                ),
                io.Int.Input(
                    "custom_height",
                    default=0,
                    min=0,
                    max=8192,
                    step=1,
                    optional=True,
                    advanced=True,
                    tooltip="Target keyframe height. Zero keeps the first image aspect/size.",
                ),
                io.Combo.Input(
                    "resize_method",
                    options=[
                        "maintain aspect ratio",
                        "stretch to fit",
                        "pad",
                        "pad green",
                        "crop",
                    ],
                    default="maintain aspect ratio",
                    optional=True,
                    advanced=True,
                    tooltip="How keyframes are resized before batching.",
                ),
                io.Int.Input(
                    "divisible_by",
                    default=32,
                    min=1,
                    max=256,
                    step=1,
                    optional=True,
                    advanced=True,
                    tooltip="Final keyframe dimensions are snapped to this divisor.",
                ),
                io.Int.Input(
                    "img_compression",
                    default=18,
                    min=0,
                    max=100,
                    step=1,
                    optional=True,
                    advanced=True,
                    tooltip="H.264 CRF compression applied to keyframes; zero disables it.",
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
            ],
        )

    @classmethod
    def _validate(cls, timeline_data, frame_count, temporal_tile_size, temporal_overlap):
        chunks = _validate_timing(frame_count, temporal_tile_size, temporal_overlap)
        data, tile_prompts, keyframes = _parse_timeline(timeline_data)
        if len(tile_prompts) != len(chunks):
            raise ValueError(
                "LTXLoopingDirector: expected "
                f"{len(chunks)} tile prompts, received {len(tile_prompts)}"
            )
        _parse_keyframes(keyframes, int(frame_count))
        return data, tile_prompts, keyframes, chunks

    @classmethod
    def validate_inputs(cls, **kwargs):
        try:
            cls._validate(
                kwargs.get("timeline_data", ""),
                kwargs.get("frame_count", 241),
                kwargs.get("temporal_tile_size", 240),
                kwargs.get("temporal_overlap", 64),
            )
            for keyframe in _parse_keyframes(
                _parse_timeline(kwargs.get("timeline_data", ""))[2],
                int(kwargs.get("frame_count", 241)),
            ):
                _resolve_keyframe(keyframe[2])
        except (TypeError, ValueError) as exc:
            return str(exc)
        return True

    @classmethod
    def fingerprint_inputs(cls, timeline_data="", **kwargs):
        digest = hashlib.sha256()
        digest.update(str(timeline_data or "").encode("utf-8"))
        try:
            keyframes = _parse_timeline(timeline_data)[2]
        except (TypeError, ValueError):
            return digest.hexdigest()

        try:
            parsed = _parse_keyframes(keyframes, int(kwargs.get("frame_count", 241)))
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
        return digest.hexdigest()

    @staticmethod
    def _encode(clip, text, frame_rate):
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        return node_helpers.conditioning_set_values(
            conditioning, {"frame_rate": float(frame_rate)}
        )

    @classmethod
    def _build_conditionings(
        cls, clip, global_prompt, first_tile_prompt, tile_prompts, chunks, frame_rate
    ):
        global_prompt = (global_prompt or "").strip()
        first_tile_prompt = (first_tile_prompt or "").strip()
        cache = {}
        conditionings = []

        for tile_index in range(len(chunks)):
            parts = [global_prompt]
            if tile_index == 0:
                parts.append(first_tile_prompt)
            parts.append(tile_prompts[tile_index].strip())
            text = "\n\n".join(part for part in parts if part) or " "
            if text not in cache:
                cache[text] = cls._encode(clip, text, frame_rate)
            conditionings.append(cache[text])
        return conditionings

    @classmethod
    def _build_keyframes(
        cls,
        keyframes,
        frame_count,
        custom_width,
        custom_height,
        resize_method,
        divisible_by,
        img_compression,
    ):
        parsed = _parse_keyframes(keyframes, frame_count)
        if not parsed:
            return None, "", None

        images = [
            _process_keyframe(
                _load_keyframe(image_file),
                int(custom_width),
                int(custom_height),
                resize_method,
                int(divisible_by),
                int(img_compression),
            )
            for _, _, image_file in parsed
        ]
        height, width = images[0].shape[1:3]
        images = [_normalize_keyframe(image, width, height) for image in images]
        cond_images = torch.cat(images, dim=0)
        indices = ",".join(str(frame) for frame, _, _ in parsed)
        start_image = next(
            (image for (frame, _, _), image in zip(parsed, images) if frame == 0),
            None,
        )
        if start_image is not None:
            start_image = start_image[:1]
        return cond_images, indices, start_image

    @classmethod
    def execute(
        cls,
        clip,
        global_prompt="",
        first_tile_prompt="",
        frame_count=241,
        temporal_tile_size=240,
        temporal_overlap=64,
        frame_rate=24.0,
        timeline_data="",
        custom_width=0,
        custom_height=0,
        resize_method="maintain aspect ratio",
        divisible_by=32,
        img_compression=18,
    ):
        custom_width = int(custom_width or 0)
        custom_height = int(custom_height or 0)
        resize_method = resize_method or "maintain aspect ratio"
        divisible_by = int(divisible_by or 32)
        img_compression = int(img_compression if img_compression is not None else 18)
        _, tile_prompts, keyframes, chunks = cls._validate(
            timeline_data, frame_count, temporal_tile_size, temporal_overlap
        )
        output_global_prompt = global_prompt or ""
        global_prompt = output_global_prompt.strip()
        positive = cls._encode(clip, global_prompt or " ", frame_rate)
        conditionings = cls._build_conditionings(
            clip,
            global_prompt,
            first_tile_prompt,
            tile_prompts,
            chunks,
            frame_rate,
        )
        cond_images, cond_indices, start_image = cls._build_keyframes(
            keyframes,
            int(frame_count),
            custom_width,
            custom_height,
            resize_method,
            divisible_by,
            img_compression,
        )
        log.info(
            "[LTXLoopingDirector] %d tile conditionings, %d keyframes, frame_count=%d",
            len(conditionings),
            0 if cond_images is None else cond_images.shape[0],
            int(frame_count),
        )
        return io.NodeOutput(
            positive,
            conditionings,
            cond_images,
            cond_indices,
            int(temporal_tile_size),
            int(temporal_overlap),
            int(frame_count),
            float(frame_rate),
            output_global_prompt,
            start_image,
        )


NODE_CLASS_MAPPINGS = {
    "LTXLoopingDirector": LTXLoopingDirector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLoopingDirector": "LTX Looping Director",
}
