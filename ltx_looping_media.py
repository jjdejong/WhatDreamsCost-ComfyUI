import math
import os

import av
import comfy.utils
import folder_paths
import numpy as np
import torch


def parse_media_timeline(data):
    """Return the validated v2 media portions of a Looping Director timeline."""
    if not isinstance(data, dict):
        raise ValueError("LTXLoopingDirector: timeline data must be an object")

    def media_list(name):
        value = data.get(name, []) or []
        if not isinstance(value, list):
            raise ValueError(f"LTXLoopingDirector: {name} must be a JSON array")
        return value

    video_segments = media_list("video_segments")
    ic_segments = media_list("ic_segments")
    audio_segments = media_list("audio_segments")
    for name, segments in (
        ("video_segments", video_segments),
        ("ic_segments", ic_segments),
        ("audio_segments", audio_segments),
    ):
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                raise ValueError(f"LTXLoopingDirector: {name}[{index}] must be an object")

    return video_segments, ic_segments, audio_segments


_UPSCALE_METHODS = ("nearest-exact", "bilinear", "area", "bicubic", "bislerp")


def resolve_ic_settings(data):
    """Validated guide encoding settings shared by the Video, IC, and Retake guides."""
    raw = data.get("ic_settings")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("LTXLoopingDirector: ic_settings must be an object")

    crop = raw.get("crop", "center")
    if crop not in {"center", "disabled"}:
        raise ValueError("LTXLoopingDirector: ic_settings.crop must be 'center' or 'disabled'")
    upscale_method = raw.get("upscale_method", "bilinear")
    if upscale_method not in _UPSCALE_METHODS:
        raise ValueError(
            "LTXLoopingDirector: ic_settings.upscale_method must be one of "
            + ", ".join(_UPSCALE_METHODS)
        )
    use_tiled_encode = raw.get("use_tiled_encode", False)
    if not isinstance(use_tiled_encode, bool):
        raise ValueError("LTXLoopingDirector: ic_settings.use_tiled_encode must be a boolean")

    def integer(name, default, minimum, maximum):
        value = raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"LTXLoopingDirector: ic_settings.{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"LTXLoopingDirector: ic_settings.{name} must be finite")
        value = int(value)
        if not minimum <= value <= maximum:
            raise ValueError(
                f"LTXLoopingDirector: ic_settings.{name} must be in {minimum}..{maximum}"
            )
        return value

    return {
        "crop": crop,
        "upscale_method": upscale_method,
        "use_tiled_encode": use_tiled_encode,
        "tile_size": integer("tile_size", 256, 64, 512),
        "tile_overlap": integer("tile_overlap", 64, 16, 256),
    }


def _encode_guide_pixels(vae, frames, width, height, settings):
    """Resize and VAE-encode guide frames using the configured IC settings."""
    pixels = comfy.utils.common_upscale(
        frames.movedim(-1, 1),
        int(width),
        int(height),
        settings["upscale_method"],
        crop=settings["crop"],
    ).movedim(1, -1)
    encode_pixels = pixels[:, :, :, :3]
    if settings["use_tiled_encode"]:
        return vae.encode_tiled(
            encode_pixels,
            tile_x=settings["tile_size"],
            tile_y=settings["tile_size"],
            overlap=settings["tile_overlap"],
        )
    return vae.encode(encode_pixels)


def retake_tiles(retake, chunk_count=None):
    """Selected Retake tiles, accepting the legacy single-``tile`` form."""
    if not isinstance(retake, dict):
        return []
    raw = retake.get("tiles")
    if raw is None:
        raw = [retake["tile"]] if "tile" in retake else []
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = [raw]
    if not isinstance(raw, list):
        raise ValueError("LTXLoopingDirector: retake.tiles must be an array of tile indices")
    tiles = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise ValueError("LTXLoopingDirector: retake.tiles entries must be integers")
        value = int(value)
        if chunk_count is not None and not 0 <= value < chunk_count:
            raise ValueError("LTXLoopingDirector: retake.tiles is outside the tile schedule")
        if value not in tiles:
            tiles.append(value)
    return sorted(tiles)


def _resolve_input_file(filename):
    if not isinstance(filename, str) or not filename:
        raise ValueError("LTXLoopingDirector: media file is required")
    try:
        path = folder_paths.get_annotated_filepath(filename)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LTXLoopingDirector: invalid media path {filename!r}") from exc
    if not os.path.isfile(path):
        raise ValueError(f"LTXLoopingDirector: media file not found: {filename}")
    return path


def _media_filename(segment, key):
    filename = segment.get(key)
    if filename:
        return filename
    if key in {"imageFile", "videoFile", "audioFile"}:
        return segment.get("imageFile") or segment.get("videoFile")
    return None


def _validate_segment(segment, name, index, frame_count, chunks=None):
    trim_start = segment.get("trimStart", 0)
    for field, value in (("trimStart", trim_start),):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"LTXLoopingDirector: {name}[{index}].{field} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"LTXLoopingDirector: {name}[{index}].{field} must be finite")
    trim_start = int(trim_start)
    if trim_start < 0:
        raise ValueError(f"LTXLoopingDirector: {name}[{index}] has invalid trimStart")

    if "tile" in segment:
        tile = segment["tile"]
        if isinstance(tile, bool) or not isinstance(tile, (int, float)) or int(tile) != tile:
            raise ValueError(f"LTXLoopingDirector: {name}[{index}].tile must be an integer")
        tile = int(tile)
        if chunks is not None and not 0 <= tile < len(chunks):
            raise ValueError(
                f"LTXLoopingDirector: {name}[{index}].tile is outside the tile schedule"
            )
        segment["tile"] = tile
    else:
        start = segment.get("start", 0)
        length = segment.get("length", 0)
        for field, value in (("start", start), ("length", length)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"LTXLoopingDirector: {name}[{index}].{field} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"LTXLoopingDirector: {name}[{index}].{field} must be finite")
        start = int(start)
        length = int(length)
        if start < 0 or length <= 0 or trim_start < 0:
            raise ValueError(f"LTXLoopingDirector: {name}[{index}] has invalid frame range")
        if start >= frame_count or start + length <= 0:
            raise ValueError(f"LTXLoopingDirector: {name}[{index}] is outside the clip")
        if start + length > frame_count:
            raise ValueError(f"LTXLoopingDirector: {name}[{index}] extends beyond the clip")
        if chunks:
            latent_start = start // 8
            tile = next(
                (tile_index for tile_index, (chunk_start, chunk_end) in enumerate(chunks)
                 if chunk_start <= latent_start < chunk_end),
                len(chunks) - 1,
            )
            segment["tile"] = tile
        segment["start"] = start
        segment["length"] = length
    segment["trimStart"] = trim_start


def validate_media_timeline(data, frame_count, chunks=None):
    video_segments, ic_segments, audio_segments = parse_media_timeline(data)
    standard_video_tiles = set()
    for name, segments, key in (
        ("video_segments", video_segments, "imageFile"),
        ("ic_segments", ic_segments, "imageFile"),
        ("audio_segments", audio_segments, "audioFile"),
    ):
        for index, segment in enumerate(segments):
            _validate_segment(segment, name, index, frame_count, chunks)
            if name == "video_segments" and "tile" in segment:
                if segment["tile"] in standard_video_tiles:
                    raise ValueError(
                        f"LTXLoopingDirector: more than one standard Video guide is assigned to tile {segment['tile']}"
                    )
                standard_video_tiles.add(segment["tile"])
            filename = _media_filename(segment, key)
            if not filename and segment.get("linkedVideoId") is not None:
                filename = next(
                    (
                        item.get("imageFile")
                        for item in video_segments
                        if str(item.get("id")) == str(segment.get("linkedVideoId"))
                    ),
                    None,
                )
            if not filename:
                raise ValueError(f"LTXLoopingDirector: {name}[{index}] has no media file")
            _resolve_input_file(filename)
            strength = segment.get("strength", segment.get("videoStrength", 1.0))
            if isinstance(strength, bool) or not isinstance(strength, (int, float)):
                raise ValueError(f"LTXLoopingDirector: {name}[{index}].strength must be numeric")
            if not math.isfinite(float(strength)) or not 0 <= float(strength) <= 1:
                raise ValueError(f"LTXLoopingDirector: {name}[{index}].strength must be in 0..1")
            segment["strength"] = float(strength)
            if name in {"video_segments", "ic_segments"}:
                attention_strength = segment.get(
                    "attentionStrength",
                    segment.get("videoAttentionStrength", 0.65 if name == "ic_segments" else 1.0),
                )
                if (
                    isinstance(attention_strength, bool)
                    or not isinstance(attention_strength, (int, float))
                    or not math.isfinite(float(attention_strength))
                    or not 0 <= float(attention_strength) <= 1
                ):
                    raise ValueError(
                        f"LTXLoopingDirector: {name}[{index}].attentionStrength must be in 0..1"
                    )
                segment["attentionStrength"] = float(attention_strength)

    resolve_ic_settings(data)

    retake = data.get("retake") or data.get("retakeVideo")
    if data.get("retake_mode") and retake:
        if not isinstance(retake, dict):
            raise ValueError("LTXLoopingDirector: retake must be an object")
        filename = retake.get("imageFile") or retake.get("videoFile") or retake.get("fileName")
        if not filename:
            raise ValueError("LTXLoopingDirector: retake has no media file")
        _resolve_input_file(filename)
        tiles = retake_tiles(retake, len(chunks) if chunks else None)
        if not tiles and chunks:
            start = retake.get("start", retake.get("retakeStart", 0))
            if isinstance(start, bool) or not isinstance(start, (int, float)):
                raise ValueError("LTXLoopingDirector: retake.start must be numeric")
            latent_start = int(start) // 8
            tiles = [next(
                (tile_index for tile_index, (chunk_start, chunk_end) in enumerate(chunks)
                 if chunk_start <= latent_start < chunk_end),
                len(chunks) - 1,
            )]
        for tile in tiles:
            _validate_segment({
                "tile": tile,
                "trimStart": retake.get("trimStart", 0),
            }, "retake", 0, frame_count, chunks)
        retake["tiles"] = tiles
        # Keep the legacy single-tile key in sync for older readers.
        retake["tile"] = tiles[0] if tiles else retake.get("tile", 0)
        strength = retake.get("strength", data.get("retakeStrength", 1.0))
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            raise ValueError("LTXLoopingDirector: retake.strength must be numeric")
        if not math.isfinite(float(strength)) or not 0 <= float(strength) <= 1:
            raise ValueError("LTXLoopingDirector: retake.strength must be in 0..1")
        retake["strength"] = float(strength)
    return video_segments, ic_segments, audio_segments


def media_fingerprint(data, frame_count=None):
    entries = []
    video_segments, ic_segments, audio_segments = parse_media_timeline(data)
    for segments, key in (
        (video_segments, "imageFile"),
        (ic_segments, "imageFile"),
        (audio_segments, "audioFile"),
    ):
        for segment in segments:
            filename = _media_filename(segment, key)
            if not filename:
                continue
            try:
                path = _resolve_input_file(filename)
                stat = os.stat(path)
                entries.append((filename, stat.st_size, stat.st_mtime_ns))
            except (OSError, TypeError, ValueError):
                entries.append((filename, "missing"))
    retake = data.get("retake") or {}
    if isinstance(retake, dict):
        filename = retake.get("imageFile") or retake.get("videoFile") or retake.get("fileName")
        if filename:
            try:
                path = _resolve_input_file(filename)
                stat = os.stat(path)
                entries.append((filename, stat.st_size, stat.st_mtime_ns))
            except (OSError, TypeError, ValueError):
                entries.append((filename, "missing"))
    return entries


def _resample_video_frames(frames, target_count, mode="nearest"):
    if frames.shape[0] == target_count:
        return frames
    if frames.shape[0] == 1:
        return frames.repeat(target_count, 1, 1, 1)
    positions = torch.linspace(0, frames.shape[0] - 1, target_count, dtype=torch.float32)
    if mode in {"linear", "bilinear"}:
        index_0 = positions.floor().long().clamp(0, frames.shape[0] - 1)
        index_1 = positions.ceil().long().clamp(0, frames.shape[0] - 1)
        alpha = (positions - index_0.to(positions.dtype)).reshape(-1, 1, 1, 1)
        frame_0 = frames.index_select(0, index_0).to(torch.float32)
        frame_1 = frames.index_select(0, index_1).to(torch.float32)
        return (frame_0 * (1.0 - alpha) + frame_1 * alpha).to(frames.dtype)
    return frames.index_select(0, positions.round().long())


def _decode_video_frames(filename, start_frame, length, frame_rate, resample_mode="nearest"):
    path = _resolve_input_file(filename)
    extension = os.path.splitext(path)[1].lower()
    if extension in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        from PIL import Image

        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0).repeat(max(1, length), 1, 1, 1)

    target_fps = max(0.001, float(frame_rate))
    start_time = max(0.0, int(start_frame) / target_fps)
    end_time = start_time + max(1, int(length)) / target_fps
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        source_fps = float(stream.average_rate) if stream.average_rate else target_fps
        if source_fps <= 0:
            source_fps = target_fps
        if start_time and stream.time_base:
            container.seek(int(max(0.0, start_time - 0.5) / float(stream.time_base)), stream=stream, backward=True)
        decoded_count = 0
        for frame in container.decode(stream):
            timestamp = frame.time
            if timestamp is None and frame.pts is not None and stream.time_base:
                timestamp = float(frame.pts * stream.time_base)
            if timestamp is None:
                timestamp = decoded_count / source_fps
            decoded_count += 1
            if timestamp < start_time - 0.01:
                continue
            if timestamp >= end_time:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"LTXLoopingDirector: no video frames decoded from {filename}")
    return _resample_video_frames(
        torch.from_numpy(np.asarray(frames, dtype=np.float32) / 255.0),
        max(1, int(length)),
        resample_mode,
    )


def first_video_frame(filename):
    return _decode_video_frames(filename, 0, 1, 24.0)[:1]


def _tile_pixel_range(chunks, tile_index):
    latent_start, latent_end = chunks[tile_index]
    if tile_index:
        latent_start = chunks[tile_index - 1][1]
    pixel_start = latent_start * 8
    pixel_length = max(1, (latent_end - latent_start - 1) * 8 + 1)
    return pixel_start, pixel_length


def build_tile_guides(
    data,
    chunks,
    frame_rate,
    vae,
    width,
    height,
    latent_downscale_factor=1.0,
):
    """Encode tile-owned Video/IC guides at the pass resolution."""
    video_segments, ic_segments, _ = parse_media_timeline(data)
    settings = resolve_ic_settings(data)
    guide_segments = [("video", segment) for segment in video_segments]
    guide_segments.extend(("ic", segment) for segment in ic_segments)
    guides = [[] for _ in chunks]
    if not guide_segments:
        return guides
    if vae is None:
        raise ValueError(
            "LTXLoopingDirector: connect the video VAE when a Video or IC Video guide is used"
        )

    for kind, segment in guide_segments:
        tile_index = int(segment["tile"])
        if kind == "video" and any(guide["kind"] == "video" for guide in guides[tile_index]):
            raise ValueError(
                f"LTXLoopingDirector: more than one standard Video guide is assigned to tile {tile_index}"
            )
        pixel_start, pixel_length = _tile_pixel_range(chunks, tile_index)
        filename = _media_filename(segment, "imageFile")
        frames = _decode_video_frames(
            filename,
            int(segment.get("trimStart", 0)),
            pixel_length,
            float(frame_rate),
            segment.get("resampleMode", "nearest"),
        )
        downscale = max(1, int(round(float(latent_downscale_factor))))
        target_width = max(8, int(width) // downscale)
        target_height = max(8, int(height) // downscale)
        guides[tile_index].append({
            "samples": _encode_guide_pixels(vae, frames, target_width, target_height, settings),
            "kind": kind,
            "strength": float(segment.get("strength", segment.get("videoStrength", 1.0))),
            "attention_strength": float(
                segment.get(
                    "attentionStrength",
                    segment.get("videoAttentionStrength", 0.65 if kind == "ic" else 1.0),
                )
            ),
            "tile": tile_index,
            "pixel_start": pixel_start,
            "source_trim": int(segment.get("trimStart", 0)),
        })
    return guides


def build_retake_latent(data, chunks, frame_count, frame_rate, vae, width, height):
    retake = data.get("retake") or data.get("retakeVideo")
    if not data.get("retake_mode") or not isinstance(retake, dict):
        return None

    filename = retake.get("imageFile") or retake.get("videoFile") or retake.get("fileName")
    trim_start = int(retake.get("trimStart", 0))
    settings = resolve_ic_settings(data)
    latent_parts = []
    for tile_index in range(len(chunks)):
        pixel_start, pixel_length = _tile_pixel_range(chunks, tile_index)
        frames = _decode_video_frames(
            filename,
            trim_start + pixel_start,
            pixel_length,
            float(frame_rate),
        )
        latent_parts.append(_encode_guide_pixels(vae, frames, width, height, settings))
    if not latent_parts:
        return None
    latent = torch.cat(latent_parts, dim=2)
    expected_frames = (int(frame_count) - 1) // 8 + 1
    if latent.shape[2] > expected_frames:
        latent = latent[:, :, :expected_frames]
    elif latent.shape[2] < expected_frames:
        latent = torch.cat(
            [latent, latent[:, :, -1:].repeat(1, 1, expected_frames - latent.shape[2], 1, 1)],
            dim=2,
        )

    mask = torch.zeros(
        latent.shape[0], 1, latent.shape[2], 1, 1,
        device=latent.device,
        dtype=latent.dtype,
    )
    strength = float(retake.get("strength", 1.0))
    # Each selected tile regenerates only the region it owns; the earlier tile keeps
    # the overlap, so adjacent selections join without regenerating it twice.
    for tile in retake_tiles(retake, len(chunks)) or [0]:
        start, end = chunks[tile]
        if tile:
            start = chunks[tile - 1][1]
        start = max(0, min(start, latent.shape[2]))
        end = max(start, min(end, latent.shape[2]))
        mask[:, :, start:end] = strength
    return {"samples": latent, "noise_mask": mask}


def _materialize_tile_range(segment, chunks, frame_count):
    if "tile" in segment:
        tile = int(segment["tile"])
        latent_start, latent_end = chunks[tile]
        if tile:
            latent_start += max(0, chunks[tile - 1][1] - latent_start)
        pixel_start = latent_start * 8
        pixel_length = max(1, (latent_end - latent_start - 1) * 8 + 1)
        segment = dict(segment)
        segment["start"] = pixel_start
        segment["length"] = min(pixel_length, frame_count - pixel_start)
    return segment


def _decode_audio(filename, target_sample_rate):
    path = _resolve_input_file(filename)
    if os.path.splitext(path)[1].lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return None
    with av.open(path) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(
            format="fltp", layout="stereo", rate=int(target_sample_rate)
        )
        chunks = []
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                array = converted.to_ndarray()
                if array.ndim == 1:
                    array = array[None, :]
                chunks.append(torch.from_numpy(np.asarray(array, dtype=np.float32)))
        for converted in resampler.resample(None):
            array = converted.to_ndarray()
            if array.ndim == 1:
                array = array[None, :]
            chunks.append(torch.from_numpy(np.asarray(array, dtype=np.float32)))
    if not chunks:
        return None
    return torch.cat(chunks, dim=1)


def _audio_sample_rate(audio_vae):
    if audio_vae is None:
        return 44100
    if hasattr(audio_vae, "audio_sample_rate"):
        return int(audio_vae.audio_sample_rate)
    model = getattr(audio_vae, "first_stage_model", audio_vae)
    return int(getattr(model, "sample_rate", getattr(model, "sampling_rate", 44100)))


def _audio_latent_info(audio_vae, frame_count, frame_rate):
    model = getattr(audio_vae, "first_stage_model", audio_vae)
    if hasattr(model, "num_of_latents_from_frames"):
        length = int(model.num_of_latents_from_frames(frame_count, frame_rate))
    else:
        per_second = float(getattr(model, "latents_per_second", 1.0))
        length = max(1, math.ceil(frame_count / float(frame_rate) * per_second))
    channels = int(getattr(model, "latent_channels", 1))
    frequency = int(getattr(model, "latent_frequency_bins", 1))
    return channels, length, frequency


def _segment_waveform(segment, filename, frame_rate, target_sample_rate):
    source = _decode_audio(filename, target_sample_rate)
    if source is None:
        return None
    trim_start = max(0, int(segment.get("trimStart", 0)))
    length = max(1, int(segment.get("length", 1)))
    start = round(trim_start / float(frame_rate) * target_sample_rate)
    end = start + round(length / float(frame_rate) * target_sample_rate)
    source = source[:, start:end]
    expected = max(1, round(length / float(frame_rate) * target_sample_rate))
    if source.shape[1] < expected:
        source = torch.nn.functional.pad(source, (0, expected - source.shape[1]))
    return source[:, :expected]


def build_audio(data, frame_count, frame_rate, audio_vae, chunks=None):
    """Mix the v2 audio lane and return an AUDIO object plus its latent mask."""
    sample_rate = _audio_sample_rate(audio_vae)
    sample_count = max(1, round(frame_count / float(frame_rate) * sample_rate))
    waveform = torch.zeros((1, 2, sample_count), dtype=torch.float32)
    mask_segments = []
    video_by_id = {str(item.get("id")): item for item in data.get("video_segments", [])}
    use_ic_audio = bool(data.get("use_ic_video_audio"))
    segments = list(data.get("audio_segments", []) or [])
    if chunks is not None:
        segments = [_materialize_tile_range(segment, chunks, frame_count) for segment in segments]
    if use_ic_audio:
        segments = []
        for segment in data.get("ic_segments", []) or []:
            if segment.get("audioFile") or segment.get("imageFile"):
                segment = _materialize_tile_range(segment, chunks, frame_count) if chunks is not None else segment
                segments.append({**segment, "audioFile": segment.get("audioFile") or segment.get("imageFile")})

    retake = data.get("retake") or {}
    if data.get("retake_mode") and retake:
        retake_segment = {
            "audioFile": retake.get("imageFile") or retake.get("videoFile") or retake.get("fileName"),
            "start": 0,
            "length": frame_count,
            "trimStart": int(retake.get("trimStart", 0)),
        }
        segments = [retake_segment]
    for segment in segments:
        filename = segment.get("audioFile")
        if not filename and segment.get("linkedVideoId") is not None:
            linked = video_by_id.get(str(segment.get("linkedVideoId")))
            filename = linked.get("imageFile") if linked else None
        if not filename:
            continue
        audio = _segment_waveform(segment, filename, frame_rate, sample_rate)
        if audio is None:
            continue
        start = max(0, round(int(segment.get("start", 0)) / float(frame_rate) * sample_rate))
        end = min(sample_count, start + audio.shape[1])
        if start >= end:
            continue
        strength = float(segment.get("strength", 1.0))
        audio = audio * strength
        waveform[0, :, start:end] += audio[:, : end - start]
        mask_segments.append((int(segment.get("start", 0)), int(segment.get("length", 0))))

    peak = waveform.abs().amax()
    if peak > 1:
        waveform /= peak

    _, latent_length, _ = _audio_latent_info(audio_vae, frame_count, frame_rate)
    inpaint_audio = bool(data.get("inpaint_audio", True))
    audio_mask = torch.ones((1, 1, latent_length, 1), dtype=torch.float32)
    if not inpaint_audio:
        audio_mask.zero_()
    for start, length in mask_segments:
        a_start = max(0, round(start / float(frame_rate) * latent_length * frame_rate / frame_count))
        a_end = min(latent_length, round((start + length) / float(frame_rate) * latent_length * frame_rate / frame_count) + 1)
        audio_mask[:, :, a_start:a_end] = 0

    retake_range = None
    if data.get("retake_mode") and retake:
        tiles = retake_tiles(retake, len(chunks) if chunks is not None else None)
        strength = float(retake.get("strength", 1.0))
        if chunks is not None and tiles:
            ranges = [
                ((chunks[tile - 1][1] if tile else chunks[tile][0]) * 8, chunks[tile][1] * 8)
                for tile in tiles
            ]
        else:
            start = int(retake.get("start", 0))
            ranges = [(start, start + int(retake.get("length", frame_count)))]
        ranges = [(max(0, start), min(frame_count, end)) for start, end in ranges]
        retake_range = (ranges[0][0], ranges[-1][1], strength)
        audio_mask.zero_()
        # Base-video audio follows the same selected-tile mask as the video latent.
        for start, end in ranges:
            a_start = max(0, round(start / float(frame_rate) * latent_length * frame_rate / frame_count))
            a_end = min(latent_length, round(end / float(frame_rate) * latent_length * frame_rate / frame_count) + 1)
            audio_mask[:, :, a_start:a_end] = strength

    audio_latent = None
    if audio_vae is not None:
        if mask_segments or retake_range:
            audio_latent_tensor = audio_vae.encode(waveform.movedim(1, -1))
        else:
            channels, length, frequency = _audio_latent_info(audio_vae, frame_count, frame_rate)
            audio_latent_tensor = torch.zeros((1, channels, length, frequency), dtype=torch.float32)
        audio_mask = audio_mask.to(device=audio_latent_tensor.device)
        audio_latent = {"samples": audio_latent_tensor, "type": "audio", "noise_mask": audio_mask}

    return {"waveform": waveform, "sample_rate": sample_rate}, audio_latent
