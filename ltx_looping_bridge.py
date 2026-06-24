"""Bridge from the LTX Director timeline to Lightricks' LTXVLoopingSampler.

LTX Director is a single-pass preparation node: its Prompt-Relay scheduling is baked
into a cross-attention mask sized to the *whole* clip (absolute full-timeline latent
coords, see prompt_relay.create_mask_fn). The Lightricks LTXVLoopingSampler re-runs the
model per temporal tile in tile-local coords, so that baked mask silently mis-stretches
(it falls into the "scaled" branch and replays the entire prompt schedule inside every
tile). Director and the looping sampler both want to own temporal conditioning, in
incompatible coordinate frames — they don't compose.

This bridge sidesteps the mask by feeding the looping sampler's *native* per-tile prompt
path instead ("one prompt per tile, by timeline position"):

  * Each looping temporal tile is assigned the timeline text segment that overlaps that
    tile's frame window the most (global_prompt prepended to every tile). The result is a
    per-tile CONDITIONING list — exactly LTXVLoopingSampler.optional_positive_conditionings'
    contract (same as Lightricks' LTXVMultiPromptProvider).
  * Director keyframes (guide_data) are converted to an IMAGE batch + comma-separated
    index string for optional_cond_images / optional_cond_image_indices. Indices are
    snapped to multiples of 8 (looping-sampler constraint); zero-strength dummy guides
    (Director's t2v placeholder) are dropped.

Wire, into LTXVLoopingSampler: the *plain* model (NOT Director's patched model output),
an STG guider from the global prompt, Director's video_latent (concat audio for AV), and
this node's outputs. For Director 2.0, connect Director's guide_data and the bridge will
derive tile prompts from timeline_data. The older local_prompts / segment_lengths widgets
remain as a fallback for legacy workflows.
"""

import json

import comfy.utils
import node_helpers
import torch

from comfy_api.latest import io

from .ltx_director import GuideData

# LTX video VAE temporal compression: pixel_frames = 8 * (latent_frames - 1) + 1.
LTX_TIME_SCALE = 8


def _parse_segments(local_prompts, segment_lengths):
    """Reconstruct (start_px, end_px, prompt) for each timeline text segment.

    The Director JS emits segment_lengths as gap-absorbed, sorted-by-start, contiguous
    pixel-frame durations parallel to local_prompts (joined with " | "). cumsum(lengths)
    therefore yields each segment's exact start. Empty-prompt segments (image-only
    blocks) are dropped.
    """
    prompts = [p.strip() for p in local_prompts.split("|")]
    lengths = []
    for tok in segment_lengths.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            lengths.append(int(round(float(tok))))
        except ValueError:
            lengths.append(0)

    segments = []
    cursor = 0
    for i, length in enumerate(lengths):
        start, end = cursor, cursor + length
        cursor = end
        prompt = prompts[i] if i < len(prompts) else ""
        if prompt:
            segments.append((start, end, prompt))
    return segments, cursor


def _segments_from_parts(prompts, lengths):
    segments = []
    cursor = 0
    for prompt, length in zip(prompts, lengths):
        length = int(length)
        start, end = cursor, cursor + length
        cursor = end
        prompt = (prompt or "").strip()
        if prompt:
            segments.append((start, end, prompt))
    return segments, cursor


def _parse_timeline_segments(timeline_data, start_frame=0, duration_frames=None):
    """Mirror LTX Director 2.0's commitChanges prompt segmentation.

    Director 2.0 stores richer timeline_data and only derives local_prompts /
    segment_lengths as hidden compatibility widgets. Prefer this source so the bridge is
    based on the current Director model instead of copy/pasted hidden widget values.
    """
    if isinstance(timeline_data, str):
        data = json.loads(timeline_data) if timeline_data.strip() else {}
    elif isinstance(timeline_data, dict):
        data = timeline_data
    else:
        data = {}

    if not data:
        return [], 0

    start_frame = int(data.get("start_frame", start_frame) or 0)
    if duration_frames is None:
        duration_frames = data.get("duration_frames")
    if duration_frames is None:
        duration_frames = data.get("normalDurationFrames")
    duration_frames = int(duration_frames or 0)
    if duration_frames <= 0:
        return [], 0

    end_frame = start_frame + duration_frames
    prompts = []
    lengths = []

    if data.get("retakeMode", False):
        total_frames = int(data.get("normalDurationFrames") or duration_frames)
        retake_start = int(data.get("retakeStart") or 0)
        retake_length = int(data.get("retakeLength") or total_frames)
        retake_end = retake_start + retake_length
        retake_prompt = data.get("retakePrompt") or ""
        global_prompt = data.get("retake_global_prompt") or data.get("global_prompt") or ""

        pieces = [
            (start_frame, min(end_frame, retake_start), global_prompt or "video"),
            (max(start_frame, retake_start), min(end_frame, retake_end), retake_prompt or "video"),
            (max(start_frame, retake_end), end_frame, global_prompt or "video"),
        ]
        for piece_start, piece_end, prompt in pieces:
            length = piece_end - piece_start
            if length > 0:
                lengths.append(length)
                prompts.append(prompt)
        return _segments_from_parts(prompts, lengths)

    sorted_segments = sorted(data.get("segments", []) or [], key=lambda s: int(s.get("start", 0)))
    current_cursor = start_frame
    pending_gap = 0

    for seg in sorted_segments:
        seg_start = int(seg.get("start", 0))
        seg_len = int(seg.get("length", 1))
        seg_end = seg_start + seg_len
        if seg_end <= start_frame:
            continue
        if seg_start >= end_frame:
            break

        effective_start = max(seg_start, start_frame)
        if effective_start > current_cursor:
            gap_length = min(effective_start, end_frame) - current_cursor
            if lengths:
                lengths[-1] += gap_length
            else:
                pending_gap += gap_length

        clipped_end = min(seg_end, end_frame)
        clipped_length = clipped_end - effective_start
        if clipped_length > 0:
            lengths.append(clipped_length + pending_gap)
            prompts.append(seg.get("prompt") or "")
            pending_gap = 0
            current_cursor = max(current_cursor, seg_end)

    clamped_cursor = min(current_cursor, end_frame)
    if lengths and clamped_cursor < end_frame:
        lengths[-1] += end_frame - clamped_cursor

    return _segments_from_parts(prompts, lengths)


def _frame_count(total_px):
    """Snap the timeline's total pixel length to the nearest valid LTX clip length (8n+1)."""
    n = max(1, round((total_px - 1) / LTX_TIME_SCALE))
    return n * LTX_TIME_SCALE + 1


def _tile_windows(total_px, tile_size_px, overlap_px):
    """Replicate LTXVLoopingSampler's temporal tiling, returning (start_px, end_px)
    windows. Mirrors looping_sampler._process_temporal_chunks: step in latent frames
    (//8), then report pixel-frame windows."""
    latent_t = max(1, (total_px - 1) // LTX_TIME_SCALE + 1)
    tile_lat = max(1, tile_size_px // LTX_TIME_SCALE)
    ov_lat = max(0, overlap_px // LTX_TIME_SCALE)
    step_lat = max(1, tile_lat - ov_lat)

    windows = []
    i = 0
    while True:
        start = i * step_lat
        if start >= latent_t:
            break
        end = min(start + tile_lat, latent_t)
        windows.append((start * LTX_TIME_SCALE, end * LTX_TIME_SCALE))
        if end >= latent_t:
            break
        i += 1
    return windows


def _segment_for_window(segments, win_start, win_end):
    """Text segment with the largest overlap with [win_start, win_end); ties -> earliest.
    Returns the prompt string, or None when no text segment overlaps (global only)."""
    best_prompt = None
    best_overlap = 0
    for seg_start, seg_end, prompt in segments:
        overlap = min(win_end, seg_end) - max(win_start, seg_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_prompt = prompt
    return best_prompt


class LTXLoopingBridge(io.ComfyNode):
    """Adapt an LTX Director timeline into per-tile prompts + keyframes for the
    Lightricks LTXVLoopingSampler (one prompt per tile, assigned by timeline position)."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LTXLoopingBridge",
            display_name="LTX Looping Bridge",
            category="WhatDreamsCost",
            description=(
                "Drives Lightricks' LTXVLoopingSampler from an LTX Director timeline: one "
                "prompt per temporal tile (assigned by timeline position) plus keyframe "
                "image batch + indices. Sidesteps Prompt Relay's whole-clip attention mask, "
                "which cannot be tiled."
            ),
            inputs=[
                io.Clip.Input("clip", tooltip="CLIP used to encode each per-tile prompt."),
                io.String.Input(
                    "local_prompts", multiline=True, default="", optional=True,
                    tooltip="The Director node's auto-populated 'local_prompts' widget value "
                    "(segment prompts joined with ' | '). Legacy fallback; Director 2.0 "
                    "guide_data/timeline_data is preferred.",
                ),
                io.String.Input(
                    "segment_lengths", default="", optional=True,
                    tooltip="The Director node's auto-populated 'segment_lengths' widget value. "
                    "Legacy fallback when timeline_data/guide_data is not connected.",
                ),
                io.String.Input(
                    "timeline_data", default="", optional=True,
                    tooltip="Optional Director 2.0 timeline JSON. Usually not needed when guide_data "
                    "is connected because Director embeds the same data there.",
                ),
                io.Int.Input(
                    "temporal_tile_size", default=80, min=24, max=1000, step=8,
                    tooltip="MUST match temporal_tile_size on the LTXVLoopingSampler (pixel "
                    "frames). Used to align prompts to tiles.",
                ),
                io.Int.Input(
                    "temporal_overlap", default=24, min=0, max=80, step=8,
                    tooltip="MUST match temporal_overlap on the LTXVLoopingSampler (pixel frames).",
                ),
                io.String.Input(
                    "global_prompt", multiline=True, default="", optional=True,
                    tooltip="Prepended to every tile's prompt (the Director 'global_prompt'). "
                    "Wire the same text node that feeds the Director to avoid drift.",
                ),
                io.Float.Input(
                    "frame_rate", default=24.0, min=1.0, max=240.0, step=0.01, optional=True,
                    tooltip="Embedded into each conditioning (LTX needs it for temporal/audio). "
                    "Use the same frame_rate as the Director.",
                ),
                GuideData.Input(
                    "guide_data", optional=True,
                    tooltip="Optional. The Director 'guide_data' output. Converted to a keyframe "
                    "image batch + index string for the looping sampler's optional_cond_images.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="per_tile_conditionings",
                    tooltip="Per-tile prompt list -> LTXVLoopingSampler optional_positive_conditionings.",
                ),
                io.Image.Output(
                    display_name="cond_images",
                    tooltip="Keyframe image batch -> LTXVLoopingSampler optional_cond_images.",
                ),
                io.String.Output(
                    display_name="cond_image_indices",
                    tooltip="Comma-separated frame indices -> LTXVLoopingSampler optional_cond_image_indices.",
                ),
                io.Int.Output(
                    display_name="temporal_tile_size",
                    tooltip="Pass-through of the tile size -> both LTXVLoopingSampler stages (single source).",
                ),
                io.Int.Output(
                    display_name="temporal_overlap",
                    tooltip="Pass-through of the overlap -> both LTXVLoopingSampler stages.",
                ),
                io.Int.Output(
                    display_name="frame_count",
                    tooltip="Clip length (8n+1) derived from the Director timeline -> the empty video/audio latent shells.",
                ),
                io.Image.Output(
                    display_name="start_image",
                    tooltip="First keyframe of the batch (earliest by timeline position) -> use for output "
                    "dimensions (GetImageSize) and the negative-index identity anchor (VAEEncode), so the "
                    "reference image is entered only once (in the Director timeline). None if no keyframes.",
                ),
            ],
        )

    @classmethod
    def _encode(cls, clip, text, frame_rate):
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
        return node_helpers.conditioning_set_values(cond, {"frame_rate": frame_rate})

    @classmethod
    def _build_conditionings(cls, clip, segments, total_px, tile_size_px, overlap_px,
                             global_prompt, frame_rate):
        global_prompt = (global_prompt or "").strip()
        windows = _tile_windows(total_px, tile_size_px, overlap_px)

        cache = {}
        conditionings = []
        for win_start, win_end in windows:
            seg_prompt = _segment_for_window(segments, win_start, win_end)
            parts = [p for p in (global_prompt, seg_prompt) if p]
            text = " ".join(parts) if parts else " "
            if text not in cache:
                cache[text] = cls._encode(clip, text, frame_rate)
            conditionings.append(cache[text])

        print(
            f"[LTXLoopingBridge] {len(windows)} tile(s) over {total_px} px frames; "
            f"{len(segments)} text segment(s). "
            + ", ".join(
                f"[{ws}-{we})->'{(_segment_for_window(segments, ws, we) or '(global)')[:24]}'"
                for ws, we in windows
            )
        )
        return conditionings

    @classmethod
    def _build_keyframes(cls, guide_data):
        if not guide_data:
            return None, ""
        images = guide_data.get("images", []) or []
        insert_frames = guide_data.get("insert_frames", []) or []
        strengths = guide_data.get("strengths", []) or []

        kept_imgs, kept_idx, kept_strengths = [], [], []
        for i, img in enumerate(images):
            strength = strengths[i] if i < len(strengths) else 1.0
            if strength <= 0.0:
                # Director inserts a zero-strength dummy for pure t2v; the looping sampler
                # applies a single global cond_image_strength, so a 0-strength guide would
                # otherwise be injected at full strength. Drop it.
                continue
            frame = insert_frames[i] if i < len(insert_frames) else 0
            snapped = 0 if frame == 0 else int(round(frame / 8.0)) * 8
            kept_imgs.append(img)
            kept_idx.append(snapped)
            kept_strengths.append(strength)

        if not kept_imgs:
            return None, ""

        ref = kept_imgs[0]
        h, w = ref.shape[1], ref.shape[2]
        norm = []
        for img in kept_imgs:
            if img.shape[1] != h or img.shape[2] != w:
                img = (
                    comfy.utils.common_upscale(img.movedim(-1, 1), w, h, "bilinear", "center")
                    .movedim(1, -1)
                    .clamp(0, 1)
                )
            norm.append(img)
        cond_images = torch.cat(norm, dim=0)

        if len(set(round(s, 3) for s in kept_strengths)) > 1:
            print(
                f"[LTXLoopingBridge] WARNING: per-keyframe guide strengths {kept_strengths} "
                "vary, but LTXVLoopingSampler applies a single cond_image_strength to all "
                "keyframes. Set cond_image_strength on the sampler to your preferred value."
            )

        indices = ",".join(str(i) for i in kept_idx)
        print(f"[LTXLoopingBridge] {len(norm)} keyframe(s) at indices [{indices}] ({h}x{w}).")
        return cond_images, indices

    @classmethod
    def execute(cls, clip, local_prompts="", segment_lengths="", timeline_data="",
                temporal_tile_size=80, temporal_overlap=24, global_prompt="", frame_rate=24.0,
                guide_data=None) -> io.NodeOutput:
        director_timeline = ""
        director_start = 0
        director_duration = None
        if guide_data:
            director_timeline = guide_data.get("timeline_data", "") or ""
            director_start = int(guide_data.get("start_frame", 0) or 0)
            director_duration = guide_data.get("duration_frames")

        segments, total_px = _parse_timeline_segments(
            director_timeline or timeline_data,
            start_frame=director_start,
            duration_frames=director_duration,
        )
        if total_px <= 0:
            segments, total_px = _parse_segments(local_prompts or "", segment_lengths or "")
        if total_px <= 0:
            raise ValueError(
                "LTXLoopingBridge: no usable Director 2.0 timeline_data/guide_data or legacy "
                "segment_lengths was provided."
            )

        # Clip length (8n+1) and tile windows are derived from the Director timeline, so the
        # bridge is the single source of looping timing (no separate schedule node needed).
        frame_count = _frame_count(total_px)
        conditionings = cls._build_conditionings(
            clip, segments, frame_count, temporal_tile_size, temporal_overlap,
            global_prompt, frame_rate,
        )
        cond_images, cond_indices = cls._build_keyframes(guide_data)
        # The first keyframe (earliest by timeline position) doubles as the I2V start frame,
        # so dims + identity anchor can be taken from it instead of a separate LoadImage.
        start_image = cond_images[:1] if cond_images is not None else None
        return io.NodeOutput(
            conditionings, cond_images, cond_indices,
            int(temporal_tile_size), int(temporal_overlap), int(frame_count),
            start_image,
        )


NODE_CLASS_MAPPINGS = {
    "LTXLoopingBridge": LTXLoopingBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLoopingBridge": "LTX Looping Bridge",
}
