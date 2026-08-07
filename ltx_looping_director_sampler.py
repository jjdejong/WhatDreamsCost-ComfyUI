import copy
import hashlib
import json
import os
import re
import sys
import uuid
from types import MethodType

import comfy.utils
import folder_paths
import safetensors.torch
import torch
from comfy.nested_tensor import NestedTensor

from comfy_api.latest import io

from .ltx_director_guide import (
    _get_guide_attention_entries,
    _load_lora_model_only,
    _set_guide_attention_entries,
)
from .ltx_looping_media import (
    build_audio,
    build_retake_latent,
    build_tile_guides,
    media_fingerprint,
    validate_media_timeline,
)
from .patches import get_current_node_id


def _loaded_ltx_module(relative_name):
    try:
        import nodes
        root = nodes.LOADED_MODULE_DIRS.get("ComfyUI-LTXVideo")
    except (ImportError, AttributeError):
        root = None
    if not root:
        raise RuntimeError(
            "LTXLoopingDirectorSampler requires the ComfyUI-LTXVideo custom node to be loaded"
        )

    target = os.path.realpath(os.path.join(root, f"{relative_name}.py"))
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and os.path.realpath(filename) == target:
            return module
    raise RuntimeError(
        f"ComfyUI-LTXVideo is missing its loaded {relative_name}.py runtime module"
    )


def _ltx_looping_module():
    return _loaded_ltx_module("looping_sampler")


def _ltx_easy_samplers():
    return _loaded_ltx_module("easy_samplers")


def _ltx_latents():
    return _loaded_ltx_module("latents")


def _ltx_latent_nodes():
    from comfy_extras.nodes_lt import EmptyLTXVLatentVideo
    from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio

    return EmptyLTXVLatentVideo, LTXVEmptyLatentAudio


def _unwrap_node_output(value):
    if hasattr(value, "result"):
        value = value.result
    if isinstance(value, tuple):
        return value[0]
    return value


def _split_av(latent):
    if latent is None:
        return None, None
    samples = latent.get("samples")
    if isinstance(samples, NestedTensor) and len(samples.tensors) >= 2:
        masks = latent.get("noise_mask")
        video = latent.copy()
        audio = latent.copy()
        video["samples"] = samples.tensors[0]
        audio["samples"] = samples.tensors[1]
        if isinstance(masks, NestedTensor):
            video["noise_mask"] = masks.tensors[0]
            audio["noise_mask"] = masks.tensors[1]
        else:
            video.pop("noise_mask", None)
            audio.pop("noise_mask", None)
        return video, audio
    return latent, None


def _concat_av(video, audio):
    if video is None:
        return audio
    if audio is None:
        return video
    result = video.copy()
    result.update(audio)
    result["samples"] = NestedTensor((video["samples"], audio["samples"]))
    video_mask = video.get("noise_mask")
    audio_mask = audio.get("noise_mask")
    if video_mask is not None or audio_mask is not None:
        if video_mask is None:
            video_mask = torch.ones(
                video["samples"].shape[0], 1, video["samples"].shape[2],
                video["samples"].shape[3], video["samples"].shape[4],
                device=video["samples"].device, dtype=video["samples"].dtype,
            )
        if audio_mask is None:
            audio_mask = torch.ones(
                audio["samples"].shape[0], 1, audio["samples"].shape[2],
                audio["samples"].shape[3],
                device=audio["samples"].device, dtype=audio["samples"].dtype,
            )
        result["noise_mask"] = NestedTensor((video_mask, audio_mask))
    return result


def _fit_guide_frames(guide, frames, device, leading_frames=0):
    if guide is None:
        return None
    result = guide.copy()
    samples = result["samples"]
    if leading_frames:
        samples = torch.cat(
            [samples[:, :, :1].repeat(1, 1, leading_frames, 1, 1), samples],
            dim=2,
        )
    if samples.shape[2] < frames:
        samples = torch.cat([samples, samples[:, :, -1:].repeat(1, 1, frames - samples.shape[2], 1, 1)], dim=2)
    result["samples"] = samples[:, :, :frames].to(device=device)
    if "noise_mask" in result and result["noise_mask"] is not None:
        mask = result["noise_mask"]
        if leading_frames:
            mask = torch.cat(
                [mask[:, :, :1].repeat(1, 1, leading_frames, 1, 1), mask],
                dim=2,
            )
        if mask.shape[2] < frames:
            mask = torch.cat([mask, mask[:, :, -1:].repeat(1, 1, frames - mask.shape[2], 1, 1)], dim=2)
        result["noise_mask"] = mask[:, :, :frames].to(device=device)
    return result


def _dilate_guides(guides, latent_downscale_factor):
    scale = int(round(float(latent_downscale_factor)))
    if scale <= 1:
        return guides

    ltx = _ltx_looping_module()
    dilated_guides = []
    for tile_guides in guides:
        dilated_tile_guides = []
        for guide in tile_guides:
            dilated = ltx.LTXVDilateLatent().dilate_latent(guide, scale, scale)[0]
            for key in ("kind", "strength", "attention_strength", "tile", "pixel_start", "source_trim"):
                if key in guide:
                    dilated[key] = guide[key]
            dilated_tile_guides.append(dilated)
        dilated_guides.append(dilated_tile_guides)
    return dilated_guides


def _extract_spatial_tile(self, latents, optional_guiding_latents, optional_negative_index_latents,
                          optional_normalizing_latents, optional_keyframes, v_start, v_end,
                          h_start, h_end, height_scale_factor, width_scale_factor):
    result = self._director_original_extract(
        latents,
        optional_guiding_latents,
        optional_negative_index_latents,
        optional_normalizing_latents,
        optional_keyframes,
        v_start,
        v_end,
        h_start,
        h_end,
        height_scale_factor,
        width_scale_factor,
    )
    director_tile_guides = getattr(self, "_director_tile_guides", None)
    if director_tile_guides is not None:
        (
            tile_latents,
            tile_guiding_latents,
            tile_negative_index_latents,
            tile_keyframes,
            tile_normalizing_latents,
        ) = result
        tile_guiding_latents = tile_guiding_latents.copy()
        tile_guiding_latents["_director_tile_guides"] = [
            [
                {
                    **self._extract_latent_spatial_tile(
                        guide, v_start, v_end, h_start, h_end
                    ),
                    **{
                        key: guide[key]
                        for key in (
                            "kind",
                            "strength",
                            "attention_strength",
                            "tile",
                            "pixel_start",
                            "source_trim",
                        )
                        if key in guide
                    },
                }
                for guide in tile_guides
            ]
            for tile_guides in director_tile_guides
        ]
        return (
            tile_latents,
            tile_guiding_latents,
            tile_negative_index_latents,
            tile_keyframes,
            tile_normalizing_latents,
        )
    return result


def _fit_audio_tensor(tensor, frames, audio_info):
    if tensor is None:
        return torch.zeros(
            1,
            audio_info["channels"],
            frames,
            audio_info["freq_bins"],
            device=audio_info["device"],
            dtype=audio_info["dtype"],
        )
    tensor = tensor[:, :, :frames]
    if tensor.shape[2] < frames:
        tensor = torch.cat(
            [
                tensor,
                torch.zeros(
                    tensor.shape[0], tensor.shape[1], frames - tensor.shape[2], tensor.shape[3],
                    device=tensor.device, dtype=tensor.dtype,
                ),
            ],
            dim=2,
        )
    return tensor.to(device=audio_info["device"], dtype=audio_info["dtype"])


def _audio_mask_slice(audio_info, start, frames):
    mask = audio_info.get("mask")
    if mask is None:
        return torch.ones(
            1, 1, frames, 1,
            device=audio_info["device"], dtype=audio_info["dtype"],
        )
    mask = mask[:, :, start:start + frames].to(
        device=audio_info["device"], dtype=audio_info["dtype"]
    )
    if mask.shape[2] < frames:
        mask = torch.cat(
            [
                mask,
                torch.ones(
                    mask.shape[0], mask.shape[1], frames - mask.shape[2], mask.shape[3],
                    device=mask.device, dtype=mask.dtype,
                ),
            ],
            dim=2,
        )
    return mask


def _apply_audio_mask(output, initial, mask, prefix_frames=0):
    if output is None or initial is None or mask is None:
        return output
    prefix_frames = max(0, min(int(prefix_frames), output.shape[2]))
    tail_frames = min(
        output.shape[2] - prefix_frames,
        initial.shape[2],
        mask.shape[2],
    )
    if tail_frames <= 0:
        return output
    output_tail = output[:, :, prefix_frames:prefix_frames + tail_frames]
    initial = initial[:, :, :tail_frames].to(device=output.device, dtype=output.dtype)
    mask = mask[:, :, :tail_frames].to(device=output.device, dtype=output.dtype)
    blended = initial + (output_tail - initial) * mask
    return torch.cat(
        [
            output[:, :, :prefix_frames],
            blended,
            output[:, :, prefix_frames + tail_frames:],
        ],
        dim=2,
    )


def _move_latent(latent, device, dtype):
    if latent is None:
        return None
    result = latent.copy()
    if isinstance(result.get("samples"), torch.Tensor):
        result["samples"] = result["samples"].to(device=device, dtype=dtype)
    if isinstance(result.get("noise_mask"), torch.Tensor):
        result["noise_mask"] = result["noise_mask"].to(device=device, dtype=dtype)
    return result


def _fit_video_mask(mask, samples):
    if mask is None:
        return None
    mask = mask.to(device=samples.device, dtype=samples.dtype)
    if mask.ndim != 5:
        return mask
    frames = samples.shape[2]
    if mask.shape[2] < frames:
        mask = torch.cat(
            [
                mask,
                torch.ones(
                    mask.shape[0], mask.shape[1], frames - mask.shape[2],
                    mask.shape[3], mask.shape[4],
                    device=mask.device, dtype=mask.dtype,
                ),
            ],
            dim=2,
        )
    else:
        mask = mask[:, :, :frames]
    if mask.shape[3:] == (1, 1) or mask.shape[3:] == samples.shape[3:]:
        return mask
    resized = torch.nn.functional.interpolate(
        mask.permute(0, 2, 1, 3, 4).reshape(-1, 1, mask.shape[3], mask.shape[4]),
        size=samples.shape[3:],
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(mask.shape[0], frames, mask.shape[1], samples.shape[3], samples.shape[4]).permute(0, 2, 1, 3, 4)


def _fit_audio_mask(mask, samples):
    if mask is None:
        return None
    mask = mask.to(device=samples.device, dtype=samples.dtype)
    if mask.ndim != 4:
        return mask
    frames = samples.shape[2]
    if mask.shape[2] < frames:
        mask = torch.cat(
            [
                mask,
                torch.ones(
                    mask.shape[0], mask.shape[1], frames - mask.shape[2], mask.shape[3],
                    device=mask.device, dtype=mask.dtype,
                ),
            ],
            dim=2,
        )
    else:
        mask = mask[:, :, :frames]
    if mask.shape[3] == 1 or mask.shape[3] == samples.shape[3]:
        return mask
    resized = torch.nn.functional.interpolate(
        mask.permute(0, 2, 1, 3).reshape(-1, 1, 1, mask.shape[3]),
        size=(1, samples.shape[3]),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(mask.shape[0], frames, mask.shape[1], samples.shape[3]).permute(0, 2, 1, 3)


def _set_last_guide_attention(positive, negative, strength):
    updated = []
    for conditioning in (positive, negative):
        entries = [dict(entry) for entry in _get_guide_attention_entries(conditioning)]
        if entries:
            entries[-1]["strength"] = float(strength)
            conditioning = _set_guide_attention_entries(conditioning, entries)
        updated.append(conditioning)
    return tuple(updated)


def _process_temporal_chunks(self, tile_config, sampling_config, model_config,
                             audio_info=None, save_checkpoints=False):
    """Run the temporal loop with Director-owned tile guides and AV masks."""
    ltx = _ltx_looping_module()
    select_latents = ltx.LTXVSelectLatents()
    tile_guides = tile_config.tile_guiding_latents.get("_director_tile_guides", []) if tile_config.tile_guiding_latents else []
    # ``tile_guiding_latents`` doubles as the carrier for the Director tile guides, so
    # only treat its samples as a legacy guiding latent when one was actually connected
    # for this pass. Anything else is the fabricated carrier built in ``_run_pass``.
    legacy_guiding_latents = None
    if getattr(self, "_director_legacy_guiding", False) and tile_config.tile_guiding_latents:
        legacy_guiding_latents = {
            key: value
            for key, value in tile_config.tile_guiding_latents.items()
            if key != "_director_tile_guides"
        }
    if audio_info is not None:
        audio_info = dict(audio_info)
        audio_info["mask"] = getattr(self, "_director_audio_mask", None)
    chunk_index = 0
    tile_out_latents = None
    first_tile_out_latents = None
    accumulated_audio = None
    accumulated_audio_mask = None
    temporal_size = sampling_config.temporal_tile_size
    temporal_overlap = sampling_config.temporal_overlap
    stride = temporal_size - temporal_overlap
    total_frames = tile_config.tile_latents["samples"].shape[2]
    start_temporal_tile = 0
    resume_entry = self._director_checkpoint_context.get("resume_chunks", {}).get(
        f"{tile_config.v}:{tile_config.h}"
    )
    if resume_entry:
        resumed_video = _load_saved_latent(resume_entry.get("video"))
        resumed_audio = _load_saved_latent(resume_entry.get("audio"))
        if resumed_video is None:
            raise ValueError("LTXLoopingDirectorSampler: saved video tile state is missing")
        if resume_entry.get("audio") and resumed_audio is None:
            raise ValueError("LTXLoopingDirectorSampler: saved audio tile state is missing")
        resumed_video = _move_latent(
            resumed_video,
            tile_config.tile_latents["samples"].device,
            tile_config.tile_latents["samples"].dtype,
        )
        if resumed_audio is not None:
            resumed_audio = _move_latent(
                resumed_audio,
                tile_config.tile_latents["samples"].device,
                tile_config.tile_latents["samples"].dtype,
            )
        tile_out_latents = resumed_video
        reference = _load_saved_latent(resume_entry.get("reference"))
        if resume_entry.get("reference") and reference is None:
            raise ValueError("LTXLoopingDirectorSampler: saved reference tile state is missing")
        first_tile_out_latents = _move_latent(
            reference or copy.deepcopy(resumed_video),
            tile_config.tile_latents["samples"].device,
            tile_config.tile_latents["samples"].dtype,
        )
        accumulated_audio = resumed_audio.get("samples") if resumed_audio is not None else None
        accumulated_audio_mask = resumed_audio.get("noise_mask") if resumed_audio is not None else None
        start_temporal_tile = int(resume_entry.get("chunk", -1)) + 1
        chunk_index = start_temporal_tile

    for i_temporal_tile, (start_index, end_index) in enumerate(
        zip(
            range(0, total_frames + temporal_size - temporal_overlap, stride),
            range(temporal_size, total_frames + temporal_size - temporal_overlap, stride),
        )
    ):
        if i_temporal_tile < start_temporal_tile:
            continue
        latent_frames = min(end_index, total_frames) - start_index
        latent_chunk = select_latents.select_latents(
            tile_config.tile_latents,
            start_index,
            min(end_index - 1, total_frames - 1),
        )[0]
        guides = tile_guides[i_temporal_tile] if i_temporal_tile < len(tile_guides) else []
        guides = [
            _fit_guide_frames(
                guide,
                latent_frames,
                latent_chunk["samples"].device,
                temporal_overlap if i_temporal_tile > 0 else 0,
            )
            for guide in guides
        ]
        guide = guides[0] if guides else None
        guide_strength = float(sampling_config.guiding_strength)

        if legacy_guiding_latents is not None:
            legacy_guiding_chunk = select_latents.select_latents(
                legacy_guiding_latents,
                start_index,
                min(end_index - 1, legacy_guiding_latents["samples"].shape[2] - 1),
            )[0]
        else:
            legacy_guiding_chunk = None

        if tile_config.tile_normalizing_latents is not None:
            normalizing_latent_chunk = select_latents.select_latents(
                tile_config.tile_normalizing_latents,
                start_index,
                min(end_index - 1, tile_config.tile_normalizing_latents["samples"].shape[2] - 1),
            )[0]
            normalize_per_frame = True
        else:
            normalizing_latent_chunk = first_tile_out_latents
            normalize_per_frame = False

        seed_offset = self._get_per_tile_value(sampling_config.per_tile_seed_offsets, i_temporal_tile)
        model_config.noise.seed = self._calculate_tile_seed(
            tile_config.first_seed,
            start_index,
            tile_config.vertical_tiles,
            tile_config.horizontal_tiles,
            tile_config.v,
            tile_config.h,
            seed_offset,
        )
        new_guider = self._prepare_guider_for_chunk(
            model_config.guider,
            sampling_config.optional_positive_conditionings,
            chunk_index,
        )

        this_chunk_keyframe_indices = [
            in_tile_index
            for tile_index, in_tile_index in tile_config.keyframe_per_tile_indices
            if tile_index == i_temporal_tile
        ]
        if this_chunk_keyframe_indices and tile_config.tile_keyframes is not None:
            this_chunk_keyframes = torch.cat([
                tile_config.tile_keyframes[index].unsqueeze(0)
                for index, (tile_index, _) in enumerate(tile_config.keyframe_per_tile_indices)
                if tile_index == i_temporal_tile
            ])
        else:
            this_chunk_keyframes = None
        keyframe_indices = ",".join(str(index) for index in this_chunk_keyframe_indices)

        if start_index == 0:
            audio_tile = None
            audio_initial = None
            audio_mask = None
            if audio_info is not None:
                video_tile_frames = min(temporal_size, total_frames)
                audio_tile_frames = max(1, round(video_tile_frames * audio_info["total_audio_frames"] / max(audio_info["total_video_frames"], 1)))
                audio_initial = _fit_audio_tensor(audio_info.get("tensor"), audio_tile_frames, audio_info)
                audio_tile = audio_initial.clone()
                audio_mask = _audio_mask_slice(audio_info, 0, audio_tile_frames)

            base_latents = latent_chunk
            if guides:
                easy = _ltx_easy_samplers()
                positive, negative = easy._get_raw_conds_from_guider(new_guider)
                for tile_guide in guides:
                    positive, negative, base_latents = _ltx_latents().LTXVAddLatentGuide().generate(
                        vae=model_config.vae,
                        positive=positive,
                        negative=negative,
                        latent=base_latents,
                        guiding_latent=tile_guide,
                        latent_idx=0,
                        strength=float(tile_guide.get("strength", guide_strength)),
                    )
                    if tile_guide.get("kind") == "ic":
                        positive, negative = _set_last_guide_attention(
                            positive,
                            negative,
                            tile_guide.get("attention_strength", 1.0),
                        )
                new_guider.set_conds(positive, negative)
                new_guider.raw_conds = (positive, negative)

            if legacy_guiding_chunk is not None:
                tile_out_latents = ltx.LTXVInContextSampler().sample(
                    vae=model_config.vae,
                    guider=new_guider,
                    sampler=model_config.sampler,
                    sigmas=model_config.sigmas,
                    noise=model_config.noise,
                    guiding_latents=legacy_guiding_chunk,
                    optional_cond_images=this_chunk_keyframes,
                    optional_cond_indices=keyframe_indices,
                    num_frames=-1,
                    optional_negative_index_latents=tile_config.tile_negative_index_latents,
                    optional_negative_index=sampling_config.optional_negative_index,
                    optional_negative_index_strength=sampling_config.optional_negative_index_strength,
                    optional_initialization_latents=base_latents,
                    cond_image_strength=sampling_config.cond_image_strength,
                    guiding_strength=guide_strength,
                    guiding_start_step=sampling_config.guiding_start_step,
                    guiding_end_step=sampling_config.guiding_end_step,
                    _audio_tile=audio_tile,
                )[0]
            else:
                tile_out_latents = ltx.LTXVBaseSampler().sample(
                    model=model_config.model,
                    vae=model_config.vae,
                    noise=model_config.noise,
                    sampler=model_config.sampler,
                    sigmas=model_config.sigmas,
                    guider=new_guider,
                    num_frames=(min(temporal_size, total_frames) - 1) * sampling_config.time_scale_factor + 1,
                    width=tile_config.tile_width * sampling_config.width_scale_factor,
                    height=tile_config.tile_height * sampling_config.height_scale_factor,
                    optional_cond_images=this_chunk_keyframes,
                    optional_cond_indices=keyframe_indices,
                    crop="center",
                    crf=30,
                    strength=sampling_config.cond_image_strength,
                    optional_negative_index_latents=tile_config.tile_negative_index_latents,
                    optional_negative_index=sampling_config.optional_negative_index,
                    optional_negative_index_strength=sampling_config.optional_negative_index_strength,
                    optional_initialization_latents=base_latents,
                    guiding_start_step=sampling_config.guiding_start_step,
                    guiding_end_step=sampling_config.guiding_end_step,
                    _audio_tile=audio_tile,
                )[0]
            accumulated_audio = tile_out_latents.pop("_audio", None)
            accumulated_audio = _apply_audio_mask(accumulated_audio, audio_initial, audio_mask)
            accumulated_audio_mask = audio_mask
            first_tile_out_latents = copy.deepcopy(tile_out_latents)
            if (
                first_tile_out_latents.get("noise_mask") is None
                and tile_config.tile_latents.get("noise_mask") is not None
            ):
                first_tile_out_latents["noise_mask"] = tile_config.tile_latents["noise_mask"]
        else:
            audio_new_init = None
            audio_initial = None
            audio_mask = None
            audio_new_mask = None
            audio_prefix_frames = 0
            src_audio = audio_info.get("tensor") if audio_info else None
            if src_audio is not None and accumulated_audio is not None:
                audio_ratio = audio_info["total_audio_frames"] / max(audio_info["total_video_frames"], 1)
                audio_new_frames = max(1, round((latent_chunk["samples"].shape[2] - temporal_overlap) * audio_ratio))
                audio_start = accumulated_audio.shape[2]
                audio_new_init = _fit_audio_tensor(
                    src_audio[:, :, audio_start:min(audio_start + audio_new_frames, src_audio.shape[2])],
                    audio_new_frames,
                    audio_info,
                )
            if audio_info is not None and accumulated_audio is not None:
                audio_ratio = audio_info["total_audio_frames"] / max(audio_info["total_video_frames"], 1)
                audio_overlap = max(1, round(temporal_overlap * audio_ratio))
                audio_prefix_frames = max(0, accumulated_audio.shape[2] - audio_overlap)
                audio_initial = torch.cat(
                    [accumulated_audio[:, :, -audio_overlap:], audio_new_init],
                    dim=2,
                ) if audio_new_init is not None else accumulated_audio[:, :, -audio_overlap:]
                audio_new_mask = _audio_mask_slice(
                    audio_info,
                    accumulated_audio.shape[2],
                    audio_initial.shape[2] - audio_overlap,
                )
                audio_mask = torch.cat(
                    [torch.zeros_like(audio_initial[:, :, :audio_overlap, :1]), audio_new_mask],
                    dim=2,
                )

            extend_latents = latent_chunk
            if guides:
                easy = _ltx_easy_samplers()
                positive, negative = easy._get_raw_conds_from_guider(new_guider)
                extend_latents = latent_chunk
                for tile_guide in guides:
                    owned_guide = ltx.LTXVSelectLatents().select_latents(
                        tile_guide,
                        temporal_overlap,
                        -1,
                    )[0]
                    positive, negative, extend_latents = _ltx_latents().LTXVAddLatentGuide().generate(
                        vae=model_config.vae,
                        positive=positive,
                        negative=negative,
                        latent=extend_latents,
                        guiding_latent=owned_guide,
                        latent_idx=temporal_overlap,
                        strength=float(tile_guide.get("strength", guide_strength)),
                    )
                    if tile_guide.get("kind") == "ic":
                        positive, negative = _set_last_guide_attention(
                            positive,
                            negative,
                            tile_guide.get("attention_strength", 1.0),
                        )
                new_guider.set_conds(positive, negative)
                new_guider.raw_conds = (positive, negative)

            tile_out_latents = ltx.LTXVExtendSampler().sample(
                model=model_config.model,
                vae=model_config.vae,
                sampler=model_config.sampler,
                sigmas=model_config.sigmas,
                noise=model_config.noise,
                latents=tile_out_latents,
                num_new_frames=(latent_chunk["samples"].shape[2] - temporal_overlap) * sampling_config.time_scale_factor,
                frame_overlap=temporal_overlap * sampling_config.time_scale_factor,
                guider=new_guider,
                strength=sampling_config.temporal_overlap_cond_strength,
                guiding_strength=guide_strength,
                cond_image_strength=sampling_config.cond_image_strength,
                optional_guiding_latents=legacy_guiding_chunk,
                optional_cond_images=this_chunk_keyframes,
                optional_cond_indices=keyframe_indices,
                optional_reference_latents=normalizing_latent_chunk,
                normalize_per_frame=normalize_per_frame,
                adain_factor=sampling_config.adain_factor,
                optional_negative_index_latents=tile_config.tile_negative_index_latents,
                optional_negative_index=sampling_config.optional_negative_index,
                optional_negative_index_strength=sampling_config.optional_negative_index_strength,
                optional_initialization_latents=extend_latents,
                guiding_start_step=sampling_config.guiding_start_step,
                guiding_end_step=sampling_config.guiding_end_step,
                _audio_tile=accumulated_audio,
                _audio_new_init=audio_new_init,
            )[0]
            accumulated_audio = tile_out_latents.pop("_audio", accumulated_audio)
            accumulated_audio = _apply_audio_mask(
                accumulated_audio,
                audio_initial,
                audio_mask,
                audio_prefix_frames,
            )
            if accumulated_audio_mask is None:
                accumulated_audio_mask = audio_new_mask
            elif audio_new_mask is not None:
                accumulated_audio_mask = torch.cat(
                    [accumulated_audio_mask, audio_new_mask],
                    dim=2,
                )

        if save_checkpoints:
            self._save_chunk_checkpoint(
                tile_out_latents,
                accumulated_audio,
                tile_config,
                chunk_index,
                accumulated_audio_mask,
                first_tile_out_latents,
            )
        chunk_index += 1

    if accumulated_audio is not None:
        tile_out_latents["_audio"] = accumulated_audio
    return tile_out_latents


CHECKPOINT_FORMAT_VERSION = 2

_GENERATION_FILE = re.compile(r"^(?P<prefix>.+)_g(?P<generation>[0-9a-f]{6,32})_.*\.latent$")


def _safe_checkpoint_prefix(prefix):
    prefix = os.path.basename(str(prefix or "ltx_looping_director"))
    prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", prefix)
    return prefix[:100] or "ltx_looping_director"


def _new_generation():
    """Identifier for one checkpoint transaction.

    Every tensor written during a run carries its generation in the filename, so a
    manifest can only ever name files that belong to the run that wrote it.
    """
    return uuid.uuid4().hex[:12]


def _checkpoint_path(name):
    return os.path.join(folder_paths.get_input_directory(), os.path.basename(name))


def _manifest_path(prefix):
    return _checkpoint_path(f"{prefix}_manifest.json")


def _write_latent_file(name, tensor, noise_mask=None):
    """Write one immutable checkpoint tensor and return its basename.

    The name must be unique within its generation: it is written through a ``.tmp``
    sibling so a partial write never appears at the final path, and it is never
    overwritten afterwards.
    """
    if isinstance(tensor, NestedTensor):
        tensor = tensor.tensors[0]
    if isinstance(noise_mask, NestedTensor):
        noise_mask = noise_mask.tensors[0]
    path = _checkpoint_path(name)
    payload = {
        "latent_tensor": tensor.detach().to("cpu", torch.float32),
        "latent_format_version_0": torch.tensor([]),
    }
    if noise_mask is not None:
        payload["noise_mask"] = noise_mask.detach().to("cpu", torch.float32)
    tmp = path + ".tmp"
    comfy.utils.save_torch_file(payload, tmp)
    os.replace(tmp, path)
    return os.path.basename(path)


def _remove_chunk_files(entry):
    """Delete tensors from a chunk entry the committed manifest no longer names."""
    if not entry:
        return
    for key in ("video", "audio", "reference"):
        name = entry.get(key)
        if not name:
            continue
        try:
            os.remove(_checkpoint_path(name))
        except OSError:
            pass


def _prune_generations(prefix, generation):
    """Remove checkpoint tensors left behind by superseded generations."""
    directory = folder_paths.get_input_directory()
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        match = _GENERATION_FILE.match(name)
        if not match or match.group("prefix") != prefix:
            continue
        if match.group("generation") == generation:
            continue
        try:
            os.remove(os.path.join(directory, name))
        except OSError:
            pass


def _save_checkpoint(
    self,
    tile_out_latents,
    accumulated_audio,
    tile_config,
    chunk_index,
    accumulated_audio_mask=None,
    first_tile_out_latents=None,
):
    context = self._director_checkpoint_context
    prefix = context["prefix"]
    generation = context["generation"]
    pass_index = context["pass_index"]
    samples = tile_out_latents["samples"]
    if isinstance(samples, NestedTensor):
        video, audio = samples.tensors[0], samples.tensors[1]
    else:
        video, audio = samples, accumulated_audio

    stem = (
        f"{prefix}_g{generation}_pass{pass_index}"
        f"_v{tile_config.v}_h{tile_config.h}_c{chunk_index}"
    )

    def write_tensor(tensor, suffix, noise_mask=None):
        if tensor is None:
            return None
        return _write_latent_file(f"{stem}_{suffix}.latent", tensor, noise_mask)

    video_mask = tile_out_latents.get("noise_mask")
    if video_mask is None:
        video_mask = tile_config.tile_latents.get("noise_mask")
    written = {
        "video": write_tensor(video, "video", video_mask),
        "audio": write_tensor(audio, "audio", accumulated_audio_mask),
    }
    if first_tile_out_latents is not None:
        written["reference"] = write_tensor(
            first_tile_out_latents.get("samples"),
            "reference",
            first_tile_out_latents.get("noise_mask"),
        )
    key = f"{tile_config.v}:{tile_config.h}"
    completed_chunks = context.setdefault("completed_chunks", {})
    superseded = completed_chunks.get(key)
    completed_chunks[key] = {"chunk": chunk_index, **written}
    context["last_chunk"] = {"v": tile_config.v, "h": tile_config.h, "chunk": chunk_index, **written}
    # The tensors above are durable and under names nothing references yet, so the
    # manifest commit below is what makes this chunk current.
    _write_manifest(prefix, context)
    # Only now is the previous chunk unreachable from the committed manifest.
    _remove_chunk_files(superseded)


def _write_manifest(prefix, context):
    path = _manifest_path(prefix)
    previous = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as manifest:
                previous = json.load(manifest)
        except (OSError, json.JSONDecodeError):
            previous = {}
    generation = context["generation"]
    if previous.get("generation") != generation:
        # A different generation owns those files; none of its keys describe this run.
        previous = {}
    payload = {
        key: value
        for key, value in previous.items()
        if key not in {"completed_chunks", "last_chunk", "pass_index"}
    }
    payload.update({
        key: value
        for key, value in context.items()
        if key not in {"model", "vae", "resume_chunks"}
    })
    pass_index = int(context["pass_index"])
    completed_chunks = context.get("completed_chunks", {})
    payload["completed_chunks"] = completed_chunks
    payload[f"pass{pass_index}_chunks"] = completed_chunks
    payload.pop("completed_pass", None)
    # New progress in this pass invalidates any completed snapshot of the same pass,
    # and every later pass derived from it.
    for stale_pass in range(pass_index, 3):
        for suffix in ("av", "video", "audio"):
            payload.pop(f"pass{stale_pass}_{suffix}", None)
        if stale_pass != pass_index:
            payload.pop(f"pass{stale_pass}_chunks", None)
    payload["format_version"] = CHECKPOINT_FORMAT_VERSION
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as manifest:
        json.dump(payload, manifest, indent=2)
    os.replace(tmp, path)
    _prune_generations(prefix, generation)


def _load_saved_latent(filename):
    if not filename:
        return None
    path = os.path.join(folder_paths.get_input_directory(), os.path.basename(filename))
    if not os.path.isfile(path):
        return None
    # ``comfy.utils.save_torch_file`` always writes safetensors, but its loader
    # dispatches on the file extension and would try ``torch.load`` on ``.latent``.
    # Read it the way the stock LoadLatent node does.
    try:
        payload = safetensors.torch.load_file(path, device="cpu")
    except Exception:
        return None
    tensor = payload.get("latent_tensor")
    if tensor is None:
        return None
    noise_mask = payload.get("noise_mask")
    audio = payload.get("audio_tensor")
    if audio is not None:
        audio_mask = payload.get("audio_noise_mask")
        if audio_mask is None and noise_mask is not None and noise_mask.ndim == 4:
            audio_mask = noise_mask
            noise_mask = None
        if noise_mask is None and audio_mask is not None:
            noise_mask = torch.ones(
                tensor.shape[0], 1, tensor.shape[2], tensor.shape[3], tensor.shape[4],
                dtype=tensor.dtype,
            )
        if audio_mask is None and noise_mask is not None:
            audio_mask = torch.ones(
                audio.shape[0], 1, audio.shape[2], audio.shape[3],
                dtype=audio.dtype,
            )
        result = {"samples": NestedTensor((tensor, audio))}
        if noise_mask is not None or audio_mask is not None:
            result["noise_mask"] = NestedTensor((noise_mask, audio_mask))
        return result
    return {"samples": tensor, **({"noise_mask": noise_mask} if noise_mask is not None else {})}


def _require_saved_latent(pass_index, name):
    """Load a manifest-referenced stream, failing loudly when it cannot be read."""
    latent = _load_saved_latent(name)
    if latent is None:
        raise ValueError(
            f"LTXLoopingDirectorSampler: pass {pass_index} checkpoint references "
            f"'{name}', which is missing or unreadable"
        )
    return latent


def _load_pass_snapshot(prefix, pass_index, generation=None):
    manifest_path = _manifest_path(prefix)
    if not os.path.isfile(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    if generation is not None and manifest.get("generation") != generation:
        return None

    video_name = manifest.get(f"pass{pass_index}_video")
    audio_name = manifest.get(f"pass{pass_index}_audio")
    av_name = manifest.get(f"pass{pass_index}_av")
    if not video_name:
        if not av_name:
            return None
        return _require_saved_latent(pass_index, av_name)
    # Every stream the manifest names must be present, otherwise a "completed" pass
    # could silently come back without its audio.
    video = _require_saved_latent(pass_index, video_name)
    audio = _require_saved_latent(pass_index, audio_name) if audio_name else None
    return _concat_av(video, audio)


def _attach_pass_masks(result, source):
    source_video, source_audio = _split_av(source)
    if result is None or source_video is None:
        return result
    result = result.copy()
    samples = result["samples"]
    if isinstance(samples, NestedTensor) and len(samples.tensors) >= 2:
        video_mask = _fit_video_mask(source_video.get("noise_mask"), samples.tensors[0])
        audio_mask = (
            _fit_audio_mask(source_audio.get("noise_mask"), samples.tensors[1])
            if source_audio is not None
            else None
        )
        if source_audio is not None and audio_mask is None:
            audio_tensor = samples.tensors[1]
            audio_mask = torch.ones(
                audio_tensor.shape[0], 1, audio_tensor.shape[2], audio_tensor.shape[3],
                device=audio_tensor.device, dtype=audio_tensor.dtype,
            )
        if video_mask is not None or audio_mask is not None:
            if video_mask is None:
                video_tensor = samples.tensors[0]
                video_mask = torch.ones(
                    video_tensor.shape[0], 1, video_tensor.shape[2], video_tensor.shape[3], video_tensor.shape[4],
                    device=video_tensor.device, dtype=video_tensor.dtype,
                )
            result["noise_mask"] = NestedTensor((video_mask, audio_mask))
    elif source_video.get("noise_mask") is not None:
        result["noise_mask"] = _fit_video_mask(source_video["noise_mask"], samples)
    return result


def _ensure_audio_mask(latent, media, plan, audio_vae, chunks):
    video, audio = _split_av(latent)
    if audio is None or audio.get("noise_mask") is not None:
        return latent
    if audio_vae is not None:
        _, built_audio = build_audio(
            media,
            int(plan["frame_count"]),
            float(plan["frame_rate"]),
            audio_vae,
            chunks,
        )
        if built_audio is not None and built_audio.get("noise_mask") is not None:
            audio["noise_mask"] = built_audio["noise_mask"]
    if audio.get("noise_mask") is None:
        audio_tensor = audio["samples"]
        audio["noise_mask"] = torch.ones(
            audio_tensor.shape[0], 1, audio_tensor.shape[2], audio_tensor.shape[3],
            device=audio_tensor.device, dtype=audio_tensor.dtype,
        )
    return _concat_av(video, audio)


def _fingerprint_value(digest, value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        digest.update(b"tensor:")
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode("utf-8"))
        flat = tensor.to(device="cpu", dtype=torch.float32).reshape(-1)
        if flat.numel():
            sample = torch.cat((flat[:32], flat[-32:]))
            digest.update(sample.numpy().tobytes())
            digest.update(str((float(flat.sum()), float(flat.mean()), float(flat.std(unbiased=False)))).encode("utf-8"))
        return
    if isinstance(value, dict):
        digest.update(b"dict{")
        for key in sorted(value):
            digest.update(str(key).encode("utf-8"))
            _fingerprint_value(digest, value[key])
        digest.update(b"}")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"list[")
        for item in value:
            _fingerprint_value(digest, item)
        digest.update(b"]")
        return
    if isinstance(value, (str, int, float, bool)) or value is None:
        digest.update(repr(value).encode("utf-8"))
        return
    digest.update(f"{type(value).__module__}.{type(value).__qualname__}".encode("utf-8"))


def _callable_identity(value):
    if value is None:
        return None
    return {
        "module": getattr(value, "__module__", None),
        "qualname": getattr(value, "__qualname__", getattr(value, "__name__", None)),
    }


def _tensor_edge_signature(value):
    tensor = value.detach().reshape(-1)
    if not tensor.numel():
        return "empty"
    edge = torch.cat((tensor[:32], tensor[-32:])).to(device="cpu", dtype=torch.float32)
    return hashlib.sha256(edge.numpy().tobytes()).hexdigest()


def _module_edge_identity(value):
    state = value.state_dict()
    entries = list(state.items())
    selected = entries[:2] + entries[-2:]
    return {
        "key_count": len(entries),
        "edges": [
            {
                "key": key,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "signature": _tensor_edge_signature(tensor),
            }
            for key, tensor in selected
        ],
    }


def _component_identity(value):
    if value is None:
        return None
    identity = {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
    }
    patcher = value if hasattr(value, "cached_patcher_init") else getattr(value, "patcher", None)
    cached_init = getattr(patcher, "cached_patcher_init", None)
    if cached_init:
        identity["loader"] = _callable_identity(cached_init[0])
        identity["loader_args"] = cached_init[1:]
    for name in ("model_hash", "filename", "model_name"):
        if hasattr(value, name):
            identity[name] = getattr(value, name)
    if hasattr(value, "sampler_function"):
        identity["sampler_function"] = _callable_identity(value.sampler_function)
        identity["extra_options"] = value.extra_options
        identity["inpaint_options"] = value.inpaint_options
    if hasattr(value, "cfg"):
        identity["cfg"] = value.cfg
    inner_model = getattr(value, "model", None)
    if inner_model is not None:
        identity["inner_type"] = f"{type(inner_model).__module__}.{type(inner_model).__qualname__}"
        if "upsampler" in type(inner_model).__name__.lower() and not cached_init:
            identity["state"] = _module_edge_identity(inner_model)
    return identity


def _checkpoint_fingerprint(
    plan,
    per_tile_conditionings,
    cond_image_indices,
    pass_count,
    pass1_noise,
    pass1_sampler,
    pass1_sigmas,
    pass1_guider,
    pass2_noise,
    pass2_sampler,
    pass2_sigmas,
    pass2_guider,
    controls,
    runtime_inputs,
):
    digest = hashlib.sha256()
    _fingerprint_value(digest, plan)
    _fingerprint_value(digest, media_fingerprint(plan.get("media", {})))
    _fingerprint_value(digest, per_tile_conditionings)
    _fingerprint_value(digest, cond_image_indices)
    _fingerprint_value(digest, pass_count)
    _fingerprint_value(digest, runtime_inputs)
    for component in (
        pass1_noise,
        pass1_sampler,
        pass1_sigmas,
        pass1_guider,
        pass2_noise,
        pass2_sampler,
        pass2_sigmas,
        pass2_guider,
        controls,
    ):
        if hasattr(component, "seed"):
            _fingerprint_value(digest, getattr(component, "seed"))
        if hasattr(component, "raw_conds"):
            _fingerprint_value(digest, getattr(component, "raw_conds"))
        _fingerprint_value(digest, _component_identity(component))
        _fingerprint_value(digest, component)
    return digest.hexdigest()


class LTXLoopingDirectorSampler(io.ComfyNode):
    """Run one or two looping passes from the plan emitted by LTXLoopingDirector."""

    @classmethod
    def define_schema(cls):
        float_input = lambda name, default: io.Float.Input(name, default=default, min=0.0, max=1.0, step=0.01, optional=True)
        return io.Schema(
            node_id="LTXLoopingDirectorSampler",
            display_name="LTX Looping Director Sampler",
            category="WhatDreamsCost",
            description=(
                "Samples the Director plan tile by tile. Video, IC Video, audio, AV concatenation, "
                "and the optional latent upscale pass stay inside this node; VAE decoding remains external."
            ),
            inputs=[
                io.Custom("LTX_LOOPING_DIRECTOR_PLAN").Input("director_plan"),
                io.Conditioning.Input("per_tile_conditionings"),
                io.Image.Input("cond_images", optional=True),
                io.String.Input("cond_image_indices", default="", optional=True),
                io.Model.Input("model"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.LatentUpscaleModel.Input("latent_upscale_model", optional=True),
                io.Noise.Input("pass1_noise"),
                io.Sampler.Input("pass1_sampler"),
                io.Sigmas.Input("pass1_sigmas"),
                io.Guider.Input("pass1_guider"),
                io.Noise.Input("pass2_noise", optional=True),
                io.Sampler.Input("pass2_sampler", optional=True),
                io.Sigmas.Input("pass2_sigmas", optional=True),
                io.Guider.Input("pass2_guider", optional=True),
                io.Combo.Input("pass_count", options=[1, 2], default=2),
                io.Combo.Input(
                    "execution_mode",
                    options=["full", "pass_1_only", "pass_2_from_inputs", "resume_latest"],
                    default="full",
                ),
                io.Combo.Input("ic_lora_name", options=["None"] + folder_paths.get_filename_list("loras"), default="None", optional=True),
                io.Float.Input("ic_lora_strength", default=1.0, min=-100.0, max=100.0, step=0.01, optional=True),
                float_input("pass1_guiding_strength", 1.0),
                float_input("pass1_overlap_cond_strength", 0.5),
                float_input("pass1_cond_image_strength", 1.0),
                io.Float.Input("pass1_adain_factor", default=0.0, min=0.0, max=1.0, step=0.01, optional=True),
                io.Int.Input("pass1_guiding_start_step", default=0, min=0, max=1000, step=1, optional=True),
                io.Int.Input("pass1_guiding_end_step", default=1000, min=0, max=1000, step=1, optional=True),
                float_input("pass2_guiding_strength", 1.0),
                float_input("pass2_overlap_cond_strength", 0.5),
                float_input("pass2_cond_image_strength", 1.0),
                io.Float.Input("pass2_adain_factor", default=0.0, min=0.0, max=1.0, step=0.01, optional=True),
                io.Int.Input("pass2_guiding_start_step", default=0, min=0, max=1000, step=1, optional=True),
                io.Int.Input("pass2_guiding_end_step", default=1000, min=0, max=1000, step=1, optional=True),
                io.Int.Input("pass1_horizontal_tiles", default=1, min=1, max=6, step=1, optional=True),
                io.Int.Input("pass1_vertical_tiles", default=1, min=1, max=6, step=1, optional=True),
                io.Int.Input("pass1_spatial_overlap", default=8, min=0, max=128, step=1, optional=True),
                io.Int.Input("pass2_horizontal_tiles", default=1, min=1, max=6, step=1, optional=True),
                io.Int.Input("pass2_vertical_tiles", default=1, min=1, max=6, step=1, optional=True),
                io.Int.Input("pass2_spatial_overlap", default=8, min=0, max=128, step=1, optional=True),
                io.Latent.Input("pass1_guiding_latent", optional=True),
                io.Latent.Input("pass1_negative_index_latent", optional=True),
                io.Float.Input("pass1_negative_index_strength", default=1.0, min=0.0, max=1.0, step=0.01, optional=True),
                io.Latent.Input("pass1_normalizing_latent", optional=True),
                io.String.Input("pass1_seed_offsets", default="0", optional=True),
                io.Latent.Input("pass2_guiding_latent", optional=True),
                io.Latent.Input("pass2_negative_index_latent", optional=True),
                io.Float.Input("pass2_negative_index_strength", default=1.0, min=0.0, max=1.0, step=0.01, optional=True),
                io.Latent.Input("pass2_normalizing_latent", optional=True),
                io.String.Input("pass2_seed_offsets", default="0", optional=True),
                io.Latent.Input("pass1_latent", optional=True),
                io.Latent.Input("pass2_latent", optional=True),
                io.Latent.Input("external_pass1_video_latent", optional=True),
                io.Latent.Input("external_pass1_audio_latent", optional=True),
                io.Combo.Input("checkpoint_policy", options=["off", "pass_boundaries", "every_tile"], default="off", optional=True),
                io.Combo.Input("resume", options=["off", "latest"], default="off", optional=True),
                io.String.Input("checkpoint_prefix", default="ltx_looping_director", optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="av_latent"),
                io.Latent.Output(display_name="video_latent"),
                io.Latent.Output(display_name="audio_latent"),
            ],
        )

    @classmethod
    def _validate_plan(cls, plan):
        if not isinstance(plan, dict) or plan.get("version") != 1:
            raise ValueError("LTXLoopingDirectorSampler: invalid director_plan")
        frame_count = int(plan.get("frame_count", 0))
        chunks = [(int(item["start"]), int(item["end"])) for item in plan.get("chunks", [])]
        if frame_count <= 0 or not chunks:
            raise ValueError("LTXLoopingDirectorSampler: director_plan has no tile schedule")
        media = plan.get("media", {})
        validate_media_timeline(media, frame_count, chunks)
        return frame_count, chunks, media

    @staticmethod
    def _empty_video(plan):
        EmptyLTXVLatentVideo, _ = _ltx_latent_nodes()
        return _unwrap_node_output(EmptyLTXVLatentVideo.execute(
            int(plan["stage_1_width"]), int(plan["stage_1_height"]), int(plan["frame_count"]), 1
        ))

    @staticmethod
    def _empty_audio(plan, audio_vae):
        if audio_vae is None:
            return None
        _, LTXVEmptyLatentAudio = _ltx_latent_nodes()
        return _unwrap_node_output(LTXVEmptyLatentAudio.execute(
            int(plan["frame_count"]), float(plan["frame_rate"]), 1, audio_vae
        ))

    @staticmethod
    def _upsample(video_latent, latent_upscale_model, video_vae):
        if latent_upscale_model is None:
            raise ValueError("LTXLoopingDirectorSampler: connect latent_upscale_model for pass 2")
        from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler

        result = _unwrap_node_output(LTXVLatentUpsampler.execute(video_latent, latent_upscale_model, video_vae))
        if video_latent.get("noise_mask") is not None:
            result["noise_mask"] = _fit_video_mask(video_latent["noise_mask"], result["samples"])
        return result

    @classmethod
    def _run_pass(cls, plan, per_tile_conditionings, cond_images, cond_image_indices,
                  model, video_vae, audio_vae, latent_upscale_model, latents,
                  noise, sampler, sigmas, guider, pass_index, horizontal_tiles, vertical_tiles,
                  spatial_overlap, guiding_strength, overlap_cond_strength, cond_image_strength,
                  adain_factor, guiding_start_step, guiding_end_step, ic_lora_name,
                  ic_lora_strength, legacy_guiding_latent, negative_index_latent,
                  negative_index_strength, normalizing_latent, seed_offsets,
                  checkpoint_policy, checkpoint_prefix, resume_chunks=None,
                  checkpoint_fingerprint=None, checkpoint_generation=None):
        if noise is None or sampler is None or sigmas is None or guider is None:
            raise ValueError(f"LTXLoopingDirectorSampler: pass {pass_index} sampler inputs are required")
        if checkpoint_policy != "off" and not checkpoint_generation:
            raise ValueError("LTXLoopingDirectorSampler: checkpointing requires a checkpoint generation")
        if per_tile_conditionings is None:
            raise ValueError("LTXLoopingDirectorSampler: per_tile_conditionings are required")
        ltx = _ltx_looping_module()
        video_latent, _ = _split_av(latents)
        if video_latent is None:
            raise ValueError(f"LTXLoopingDirectorSampler: pass {pass_index} has no video latent")
        media = plan["media"]
        guide_media = media
        retake_mode = bool(media.get("retake_mode"))
        if retake_mode:
            guide_media = {**media, "video_segments": [], "ic_segments": []}
        ic_segments = guide_media.get("ic_segments", []) or []
        run_model = model
        latent_downscale_factor = 1.0
        if ic_segments:
            if not ic_lora_name or ic_lora_name == "None":
                raise ValueError("LTXLoopingDirectorSampler: select ic_lora_name for IC Video")
            run_model, latent_downscale_factor = _load_lora_model_only(
                model, ic_lora_name, float(ic_lora_strength), get_current_node_id()
            )
        chunks = [(item["start"], item["end"]) for item in plan["chunks"]]
        guides = build_tile_guides(
            guide_media,
            chunks,
            plan["frame_rate"],
            video_vae,
            int(plan["stage_1_width"] if pass_index == 1 else plan["target_width"]),
            int(plan["stage_1_height"] if pass_index == 1 else plan["target_height"]),
            latent_downscale_factor,
        )
        guides = _dilate_guides(guides, latent_downscale_factor)
        has_guides = any(tile_guides for tile_guides in guides)
        # The base sampler only spatially extracts ``optional_guiding_latents`` when it is
        # not None, and that extracted dict is what carries the Director tile guides into
        # the temporal loop. Prefer this pass's real legacy guiding latent so it survives
        # spatial tiling; only fabricate a carrier when no legacy latent is connected.
        carrier = legacy_guiding_latent
        if carrier is None and has_guides:
            first_guide = next(guide for tile_guides in guides for guide in tile_guides)
            carrier = {"samples": first_guide["samples"]}
        if retake_mode:
            cond_images = None
            cond_image_indices = ""
        ltx_sampler = ltx.LTXVLoopingSampler()
        ltx_sampler._director_original_extract = ltx_sampler._extract_spatial_tile
        ltx_sampler._director_tile_guides = guides if has_guides else None
        ltx_sampler._director_legacy_guiding = legacy_guiding_latent is not None
        ltx_sampler._extract_spatial_tile = MethodType(_extract_spatial_tile, ltx_sampler)
        ltx_sampler._process_temporal_chunks = MethodType(_process_temporal_chunks, ltx_sampler)
        ltx_sampler._save_chunk_checkpoint = MethodType(_save_checkpoint, ltx_sampler)
        ltx_sampler._director_checkpoint_context = {
            "prefix": _safe_checkpoint_prefix(checkpoint_prefix),
            "generation": checkpoint_generation,
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "pass_index": pass_index,
            "frame_count": int(plan["frame_count"]),
            "fingerprint": checkpoint_fingerprint,
            "resume_chunks": resume_chunks or {},
        }
        _, input_audio = _split_av(latents)
        ltx_sampler._director_audio_mask = input_audio.get("noise_mask") if input_audio is not None else None
        result = ltx_sampler.sample(
            model=run_model,
            vae=video_vae,
            noise=noise,
            sampler=sampler,
            sigmas=sigmas,
            guider=guider,
            latents=latents,
            guiding_strength=float(guiding_strength),
            adain_factor=float(adain_factor),
            temporal_tile_size=int(plan["temporal_tile_size"]),
            temporal_overlap=int(plan["temporal_overlap"]),
            temporal_overlap_cond_strength=float(overlap_cond_strength),
            horizontal_tiles=int(horizontal_tiles),
            vertical_tiles=int(vertical_tiles),
            spatial_overlap=int(spatial_overlap),
            optional_cond_images=cond_images,
            cond_image_strength=float(cond_image_strength),
            optional_guiding_latents=carrier,
            optional_positive_conditionings=per_tile_conditionings,
            guiding_start_step=int(guiding_start_step),
            guiding_end_step=int(guiding_end_step),
            optional_cond_image_indices=cond_image_indices or "0",
            optional_negative_index_latents=negative_index_latent,
            optional_negative_index_strength=float(negative_index_strength),
            optional_normalizing_latents=normalizing_latent,
            per_tile_seed_offsets=seed_offsets or "0",
            save_checkpoints=checkpoint_policy == "every_tile",
        )
        result = _unwrap_node_output(result)
        return _attach_pass_masks(result, latents)

    @classmethod
    def execute(cls, director_plan, per_tile_conditionings, cond_images=None, cond_image_indices="",
                model=None, video_vae=None, audio_vae=None, latent_upscale_model=None,
                pass1_noise=None, pass1_sampler=None, pass1_sigmas=None, pass1_guider=None,
                pass2_noise=None, pass2_sampler=None, pass2_sigmas=None, pass2_guider=None,
                pass_count=2, execution_mode="full", ic_lora_name="None", ic_lora_strength=1.0,
                pass1_guiding_strength=1.0, pass1_overlap_cond_strength=0.5, pass1_cond_image_strength=1.0,
                pass1_adain_factor=0.0, pass1_guiding_start_step=0, pass1_guiding_end_step=1000,
                pass2_guiding_strength=1.0, pass2_overlap_cond_strength=0.5, pass2_cond_image_strength=1.0,
                pass2_adain_factor=0.0, pass2_guiding_start_step=0, pass2_guiding_end_step=1000,
                pass1_horizontal_tiles=1, pass1_vertical_tiles=1, pass1_spatial_overlap=8,
                pass2_horizontal_tiles=1, pass2_vertical_tiles=1, pass2_spatial_overlap=8,
                pass1_guiding_latent=None, pass1_negative_index_latent=None,
                pass1_negative_index_strength=1.0, pass1_normalizing_latent=None,
                pass1_seed_offsets="0", pass2_guiding_latent=None,
                pass2_negative_index_latent=None, pass2_negative_index_strength=1.0,
                pass2_normalizing_latent=None, pass2_seed_offsets="0",
                pass1_latent=None, pass2_latent=None, checkpoint_policy="off",
                resume="off", checkpoint_prefix="ltx_looping_director",
                external_pass1_video_latent=None, external_pass1_audio_latent=None):
        frame_count, chunks, media = cls._validate_plan(director_plan)
        if audio_vae is None:
            raise ValueError("LTXLoopingDirectorSampler: connect the shared audio VAE")
        pass_count = int(pass_count)
        if pass_count not in (1, 2):
            raise ValueError("LTXLoopingDirectorSampler: pass_count must be 1 or 2")
        if execution_mode not in {"full", "pass_1_only", "pass_2_from_inputs", "resume_latest"}:
            raise ValueError("LTXLoopingDirectorSampler: invalid execution_mode")
        if checkpoint_policy not in {"off", "pass_boundaries", "every_tile"}:
            raise ValueError("LTXLoopingDirectorSampler: invalid checkpoint_policy")
        if execution_mode == "pass_1_only":
            pass_count = 1
        elif execution_mode == "pass_2_from_inputs":
            if pass_count != 2:
                raise ValueError("LTXLoopingDirectorSampler: pass_2_from_inputs requires pass_count 2")
            if external_pass1_video_latent is None:
                raise ValueError(
                    "LTXLoopingDirectorSampler: pass_2_from_inputs requires an external Pass 1 video latent"
                )
        elif execution_mode == "resume_latest":
            resume = "latest"

        if execution_mode == "pass_2_from_inputs":
            required = (
                ("pass2_noise", pass2_noise),
                ("pass2_sampler", pass2_sampler),
                ("pass2_sigmas", pass2_sigmas),
                ("pass2_guider", pass2_guider),
            )
        else:
            required = (
                ("pass1_noise", pass1_noise),
                ("pass1_sampler", pass1_sampler),
                ("pass1_sigmas", pass1_sigmas),
                ("pass1_guider", pass1_guider),
            ) + (
                (
                    ("pass2_noise", pass2_noise),
                    ("pass2_sampler", pass2_sampler),
                    ("pass2_sigmas", pass2_sigmas),
                    ("pass2_guider", pass2_guider),
                ) if pass_count == 2 else ()
            )
        missing = [name for name, value in required if value is None]
        if missing:
            raise ValueError(
                "LTXLoopingDirectorSampler: missing sampler inputs: " + ", ".join(missing)
            )

        checkpoint_active = checkpoint_policy != "off"
        fingerprint_required = checkpoint_active or resume == "latest"
        checkpoint_fingerprint = None
        if fingerprint_required:
            checkpoint_fingerprint = _checkpoint_fingerprint(
                director_plan,
                per_tile_conditionings,
                cond_image_indices,
                pass_count,
                pass1_noise,
                pass1_sampler,
                pass1_sigmas,
                pass1_guider,
                pass2_noise,
                pass2_sampler,
                pass2_sigmas,
                pass2_guider,
                {
                    "pass1_spatial": [pass1_horizontal_tiles, pass1_vertical_tiles, pass1_spatial_overlap],
                    "pass2_spatial": [pass2_horizontal_tiles, pass2_vertical_tiles, pass2_spatial_overlap],
                    "pass1": [pass1_guiding_strength, pass1_overlap_cond_strength, pass1_cond_image_strength, pass1_adain_factor, pass1_guiding_start_step, pass1_guiding_end_step, pass1_negative_index_strength, pass1_seed_offsets],
                    "pass2": [pass2_guiding_strength, pass2_overlap_cond_strength, pass2_cond_image_strength, pass2_adain_factor, pass2_guiding_start_step, pass2_guiding_end_step, pass2_negative_index_strength, pass2_seed_offsets],
                    "ic_lora_name": ic_lora_name,
                    "ic_lora_strength": ic_lora_strength,
                },
                {
                    "model": _component_identity(model),
                    "video_vae": _component_identity(video_vae),
                    "audio_vae": _component_identity(audio_vae),
                    "latent_upscale_model": _component_identity(latent_upscale_model),
                    "cond_images": cond_images,
                    "pass1_latent": pass1_latent,
                    "pass2_latent": pass2_latent,
                    "pass1_guiding_latent": pass1_guiding_latent,
                    "pass1_negative_index_latent": pass1_negative_index_latent,
                    "pass1_normalizing_latent": pass1_normalizing_latent,
                    "pass2_guiding_latent": pass2_guiding_latent,
                    "pass2_negative_index_latent": pass2_negative_index_latent,
                    "pass2_normalizing_latent": pass2_normalizing_latent,
                    "external_pass1_video_latent": external_pass1_video_latent,
                    "external_pass1_audio_latent": external_pass1_audio_latent,
                },
            )
        prefix = _safe_checkpoint_prefix(checkpoint_prefix)
        manifest = None
        generation = None
        if resume == "latest":
            manifest_path = _manifest_path(prefix)
            if os.path.isfile(manifest_path):
                try:
                    with open(manifest_path, "r", encoding="utf-8") as stream:
                        manifest = json.load(stream)
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        "LTXLoopingDirectorSampler: checkpoint manifest is unreadable"
                    ) from exc
            if manifest is None:
                raise ValueError("LTXLoopingDirectorSampler: no saved checkpoint is available to resume")
            if manifest.get("format_version") != CHECKPOINT_FORMAT_VERSION:
                raise ValueError("LTXLoopingDirectorSampler: checkpoint format is unsupported")
            if manifest.get("fingerprint") != checkpoint_fingerprint:
                raise ValueError("LTXLoopingDirectorSampler: checkpoint configuration does not match the current workflow")
            generation = manifest.get("generation")
            if not generation:
                raise ValueError(
                    "LTXLoopingDirectorSampler: checkpoint manifest has no generation identifier"
                )
        elif checkpoint_active:
            # A run that is not resuming owns a fresh generation, so it can never adopt
            # tensors or completed-pass keys written by an earlier run.
            generation = _new_generation()

        def resume_chunks_for(pass_index):
            if resume != "latest" or not manifest:
                return {}
            saved = manifest.get(f"pass{pass_index}_chunks")
            if saved is not None:
                return saved
            if manifest.get("pass_index") == pass_index:
                return manifest.get("completed_chunks", {})
            return {}

        if manifest and manifest.get("completed_pass") == 2 and pass_count == 2:
            av = _load_pass_snapshot(prefix, 2, generation)
            if av is None:
                raise ValueError("LTXLoopingDirectorSampler: saved pass 2 latent is missing")
            video, audio = _split_av(av)
            return io.NodeOutput(av, video, audio)

        if manifest and manifest.get("completed_pass") == 1 and pass_count == 1:
            av = _load_pass_snapshot(prefix, 1, generation)
            if av is None:
                raise ValueError("LTXLoopingDirectorSampler: saved pass 1 latent is missing")
            video, audio = _split_av(av)
            return io.NodeOutput(av, video, audio)

        resumed_pass1 = None
        if execution_mode == "pass_2_from_inputs":
            resumed_pass1 = _concat_av(external_pass1_video_latent, external_pass1_audio_latent)
        if (
            execution_mode != "pass_2_from_inputs"
            and manifest
            and pass_count == 2
            # Only a pass that actually finished may be skipped. ``completed_pass`` is
            # cleared the moment new pass-1 progress is checkpointed, so a partial pass 1
            # can never adopt an earlier run's completed snapshot.
            and manifest.get("completed_pass") == 1
        ):
            resumed_pass1 = _load_pass_snapshot(prefix, 1, generation)
            if resumed_pass1 is None:
                raise ValueError("LTXLoopingDirectorSampler: saved pass 1 latent is missing")
        if execution_mode == "resume_latest" and resumed_pass1 is None and not manifest.get("completed_chunks"):
            raise ValueError("LTXLoopingDirectorSampler: checkpoint has no resumable tile state")

        if resumed_pass1 is not None and audio_vae is not None:
            resumed_pass1 = _ensure_audio_mask(
                resumed_pass1,
                director_plan["media"],
                director_plan,
                audio_vae,
                chunks,
            )
            resumed_video, resumed_audio = _split_av(resumed_pass1)
            if resumed_audio is None:
                _, built_audio = build_audio(director_plan["media"], frame_count, director_plan["frame_rate"], audio_vae, chunks)
                resumed_pass1 = _concat_av(
                    resumed_video,
                    built_audio or cls._empty_audio(director_plan, audio_vae),
                )

        if resumed_pass1 is not None:
            pass1 = resumed_pass1
        else:
            if pass1_latent is None:
                video = build_retake_latent(
                    director_plan["media"],
                    chunks,
                    frame_count,
                    director_plan["frame_rate"],
                    video_vae,
                    int(director_plan["stage_1_width"]),
                    int(director_plan["stage_1_height"]),
                ) or cls._empty_video(director_plan)
                audio = None
                if audio_vae is not None:
                    _, built_audio = build_audio(director_plan["media"], frame_count, director_plan["frame_rate"], audio_vae, chunks)
                    audio = built_audio or cls._empty_audio(director_plan, audio_vae)
                pass1_latent = _concat_av(video, audio)
            elif audio_vae is not None:
                video, audio = _split_av(pass1_latent)
                if audio is None:
                    _, built_audio = build_audio(director_plan["media"], frame_count, director_plan["frame_rate"], audio_vae, chunks)
                    pass1_latent = _concat_av(video, built_audio or cls._empty_audio(director_plan, audio_vae))
                else:
                    pass1_latent = _ensure_audio_mask(
                        pass1_latent,
                        director_plan["media"],
                        director_plan,
                        audio_vae,
                        chunks,
                    )

            pass1 = cls._run_pass(
                director_plan, per_tile_conditionings, cond_images, cond_image_indices,
                model, video_vae, audio_vae, latent_upscale_model, pass1_latent,
                pass1_noise, pass1_sampler, pass1_sigmas, pass1_guider, 1,
                pass1_horizontal_tiles, pass1_vertical_tiles, pass1_spatial_overlap,
                pass1_guiding_strength, pass1_overlap_cond_strength, pass1_cond_image_strength,
                pass1_adain_factor, pass1_guiding_start_step, pass1_guiding_end_step,
                ic_lora_name, ic_lora_strength, pass1_guiding_latent,
                pass1_negative_index_latent, pass1_negative_index_strength,
                pass1_normalizing_latent, pass1_seed_offsets, checkpoint_policy, prefix,
                resume_chunks_for(1), checkpoint_fingerprint, generation,
            )
            if checkpoint_active:
                _save_pass_snapshot(prefix, generation, 1, pass1, checkpoint_fingerprint)
        if pass_count == 1:
            video, audio = _split_av(pass1)
            return io.NodeOutput(pass1, video, audio)

        if pass2_latent is None:
            pass1_video, pass1_audio = _split_av(pass1)
            pass2_video = cls._upsample(pass1_video, latent_upscale_model, video_vae)
            if media.get("retake_mode"):
                base_video = build_retake_latent(
                    media,
                    chunks,
                    frame_count,
                    director_plan["frame_rate"],
                    video_vae,
                    int(director_plan["target_width"]),
                    int(director_plan["target_height"]),
                )
                if base_video is not None:
                    mask = base_video["noise_mask"].to(
                        device=pass2_video["samples"].device,
                        dtype=pass2_video["samples"].dtype,
                    )
                    pass2_video["samples"] = (
                        base_video["samples"].to(pass2_video["samples"].device, pass2_video["samples"].dtype)
                        * (1.0 - mask)
                        + pass2_video["samples"] * mask
                    )
                    pass2_video["noise_mask"] = mask
            pass2_latent = _concat_av(pass2_video, pass1_audio)
        elif _split_av(pass2_latent)[1] is None:
            pass1_audio = _split_av(pass1)[1]
            pass2_latent = _concat_av(_split_av(pass2_latent)[0], pass1_audio)
        if audio_vae is not None:
            pass2_latent = _ensure_audio_mask(
                pass2_latent,
                director_plan["media"],
                director_plan,
                audio_vae,
                chunks,
            )
        pass2 = cls._run_pass(
            director_plan, per_tile_conditionings, cond_images, cond_image_indices,
            model, video_vae, audio_vae, latent_upscale_model, pass2_latent,
            pass2_noise, pass2_sampler, pass2_sigmas, pass2_guider, 2,
            pass2_horizontal_tiles, pass2_vertical_tiles, pass2_spatial_overlap,
            pass2_guiding_strength, pass2_overlap_cond_strength, pass2_cond_image_strength,
            pass2_adain_factor, pass2_guiding_start_step, pass2_guiding_end_step,
            ic_lora_name, ic_lora_strength, pass2_guiding_latent,
            pass2_negative_index_latent, pass2_negative_index_strength,
            pass2_normalizing_latent, pass2_seed_offsets, checkpoint_policy, prefix,
            resume_chunks_for(2), checkpoint_fingerprint, generation,
        )
        if checkpoint_active:
            _save_pass_snapshot(prefix, generation, 2, pass2, checkpoint_fingerprint)
        video, audio = _split_av(pass2)
        return io.NodeOutput(pass2, video, audio)


def _save_pass_snapshot(prefix, generation, pass_index, latent, fingerprint):
    video, audio = _split_av(latent)
    stem = f"{prefix}_g{generation}_pass{pass_index}"

    def save_latent(value, suffix):
        if value is None:
            return None
        return _write_latent_file(
            f"{stem}_{suffix}.latent", value["samples"], value.get("noise_mask")
        )

    video_name = save_latent(video, "video")
    audio_name = save_latent(audio, "audio")
    manifest_path = _manifest_path(prefix)
    manifest = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as stream:
                manifest = json.load(stream)
        except (OSError, json.JSONDecodeError):
            manifest = {}
    if manifest.get("generation") != generation:
        manifest = {}
    manifest["generation"] = generation
    manifest["completed_pass"] = pass_index
    manifest["format_version"] = CHECKPOINT_FORMAT_VERSION
    manifest["fingerprint"] = fingerprint
    manifest.pop(f"pass{pass_index}_av", None)
    manifest[f"pass{pass_index}_video"] = video_name
    manifest[f"pass{pass_index}_audio"] = audio_name
    manifest[f"pass{pass_index}_chunks"] = manifest.get(
        "completed_chunks", manifest.get(f"pass{pass_index}_chunks", {})
    )
    tmp_manifest = manifest_path + ".tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    os.replace(tmp_manifest, manifest_path)
    _prune_generations(prefix, generation)


NODE_CLASS_MAPPINGS = {
    "LTXLoopingDirectorSampler": LTXLoopingDirectorSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXLoopingDirectorSampler": "LTX Looping Director Sampler",
}
