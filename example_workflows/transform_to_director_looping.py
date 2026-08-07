#!/usr/bin/env python3
"""Build the Looping Director workflow from the companion two-pass workflow.

The source graph supplies the checkpoint, text encoder, negative prompt, two guider
chains, and output decoders. The generated graph replaces both looping samplers and
their AV/upsample plumbing with ``LTXLoopingDirectorSampler``. The source workflow is
read from ComfyUI-LTXVideo, so this script does not duplicate that workflow's model
configuration.
"""

import argparse
import json
import math
import os


HERE = os.path.dirname(os.path.abspath(__file__))
BASE_NAME = "LTX-2.3_Two_Pass_I2V_Looping.json"
DEFAULT_OUTPUT = os.path.join(HERE, "LTX-2.3_Director_Looping.json")

LOOPING_DIRECTOR_ID = 300
LOOPING_SAMPLER_ID = 301
PREPROCESS_ID = 2
TIME_SCALE = 8
DEFAULT_FRAME_RATE = 24.0
DEFAULT_TOTAL_DURATION = 48.0
DEFAULT_TILE_DURATION = 10.0
DEFAULT_OVERLAP_DURATION = 2.0
DEFAULT_TARGET_HEIGHT = 1088
DEFAULT_TILE_PROMPT = (
    "The couple continues the choreography at a regular pace while the camera makes "
    "a slow orbit toward the next tile's end reference image."
)
def director_note(frame_rate, total_duration, tile_seconds, overlap_seconds, tile_count):
    """Build the sample note from the timing this run actually generated."""
    return (
        "# LTX-2.3 Looping Director\n\n"
        "**Director:** schedules one prompt per tile, places keyframes in the overlap, and derives the target width from the start-image aspect ratio.\n\n"
        "**Combined sampler:** runs one or two looping passes, optionally resumes saved tile/pass latents, and outputs concatenated AV, video-only, and audio-only latents. VAE decoding stays outside the sampler.\n\n"
        f"**This example:** {frame_rate:g} fps, {total_duration:g} seconds total, "
        f"{tile_seconds:.2f}-second tiles, and {overlap_seconds:.2f}-second overlap produce "
        f"{tile_count} tile{'s' if tile_count != 1 else ''}, so the Director carries "
        f"{tile_count} tile prompt{'s' if tile_count != 1 else ''}. Keyframes after frame 0 sit inside the preceding tile's overlap.\n\n"
        "**Start image:** the Director's conditioning images pass through `LTXV Preprocess` before the sampler, matching the source workflow's compression preprocessing. A text-to-video variant with no keyframes should bypass that node.\n\n"
        "**Prompting:** keep the global prompt stable and describe shared identity, setting, lighting, style, and audio there. Each tile prompt should be one present-tense paragraph describing only the next physical action and camera transition. State concrete camera endpoints and continuity cues; avoid contradictions, cuts, resets, or abstract emotion labels."
    )


def aligned_frames(seconds, frame_rate, minimum):
    return max(minimum, round(seconds * frame_rate / TIME_SCALE) * TIME_SCALE)


def calculate_schedule(frame_rate, total_duration, tile_duration, overlap_duration):
    frame_count = max(
        TIME_SCALE + 1,
        math.floor((total_duration * frame_rate - 1) / TIME_SCALE) * TIME_SCALE + 1,
    )
    tile_size = min(aligned_frames(tile_duration, frame_rate, 24), 1000)
    overlap = min(aligned_frames(overlap_duration, frame_rate, 16), 80, tile_size - TIME_SCALE)
    latent_frames = (frame_count - 1) // TIME_SCALE + 1
    latent_tile_size = tile_size // TIME_SCALE
    latent_overlap = overlap // TIME_SCALE
    tile_count = max(1, math.ceil((latent_frames - latent_overlap) / (latent_tile_size - latent_overlap)))
    return frame_count, tile_size, overlap, tile_count


def resolve_base(path):
    if path:
        candidate = os.path.abspath(path)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"Source workflow not found: {candidate}")
    candidates = (
        os.path.join(HERE, BASE_NAME),
        os.path.join(HERE, "..", "..", "ComfyUI-LTXVideo", "example_workflows", BASE_NAME),
    )
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find {BASE_NAME}; pass it with --base.")


def validate_source(wf):
    nodes = {node["id"]: node for node in wf.get("nodes", [])}
    expected = {
        10: "CheckpointLoaderSimple",
        11: "LTXAVTextEncoderLoader",
        12: "LTXVAudioVAELoader",
        13: "LoraLoaderModelOnly",
        14: "LatentUpscaleModelLoader",
        21: "CLIPTextEncode",
        22: "LTXVConditioning",
        40: "RandomNoise",
        41: "KSamplerSelect",
        42: "ManualSigmas",
        43: "CFGGuider",
        60: "RandomNoise",
        61: "KSamplerSelect",
        62: "ManualSigmas",
        63: "CFGGuider",
        2: "LTXVPreprocess",
        71: "LTXVSpatioTemporalTiledVAEDecode",
        72: "LTXVAudioVAEDecode",
        73: "CreateVideo",
        74: "SaveVideo",
        80: "PrimitiveStringMultiline",
    }
    missing = [f"{node_id} ({node_type})" for node_id, node_type in expected.items()
               if nodes.get(node_id, {}).get("type") != node_type]
    if missing:
        raise ValueError("Unsupported source workflow: missing " + ", ".join(missing))


def _node(node_id, node_type, title, inputs, outputs, widgets=None, pos=None, size=None):
    return {
        "id": node_id,
        "type": node_type,
        "pos": pos or [0, 0],
        "size": size or [300, 300],
        "flags": {},
        "order": node_id,
        "mode": 0,
        "inputs": [{"name": name, "type": typ, "link": None, **extra} for name, typ, extra in inputs],
        "outputs": [{"name": name, "type": typ, "links": [], "slot_index": index} for index, (name, typ) in enumerate(outputs)],
        "properties": {"Node name for S&R": node_type},
        "widgets_values": widgets or [],
        "title": title,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help=f"path to the source {BASE_NAME} workflow")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="generated workflow path")
    args = parser.parse_args(argv)

    with open(resolve_base(args.base), encoding="utf-8") as stream:
        wf = json.load(stream)
    validate_source(wf)
    nodes = {node["id"]: node for node in wf["nodes"]}
    links = {link[0]: link for link in wf["links"]}
    next_link = [max([wf.get("last_link_id", 0)] + list(links))]

    def find_input(node, name):
        for index, item in enumerate(node.get("inputs", [])):
            if item.get("name") == name:
                return index
        raise KeyError(f"{node['type']} has no input {name}")

    def add_link(from_id, from_slot, to_id, to_slot, typ):
        next_link[0] += 1
        link = [next_link[0], from_id, from_slot, to_id, to_slot, typ]
        wf["links"].append(link)
        links[link[0]] = link
        nodes[to_id]["inputs"][to_slot]["link"] = link[0]
        nodes[from_id]["outputs"][from_slot].setdefault("links", []).append(link[0])

    def remove_input_link(to_id, to_slot):
        old_id = nodes[to_id]["inputs"][to_slot].get("link")
        if old_id is None:
            return
        old = links.pop(old_id, None)
        if old:
            wf["links"] = [link for link in wf["links"] if link[0] != old_id]
            for output in nodes[old[1]].get("outputs", []):
                output["links"] = [link_id for link_id in output.get("links", []) if link_id != old_id]
        nodes[to_id]["inputs"][to_slot]["link"] = None

    def rewire(to_id, input_name, from_id, from_slot, typ):
        slot = find_input(nodes[to_id], input_name)
        remove_input_link(to_id, slot)
        add_link(from_id, from_slot, to_id, slot, typ)

    global_prompt = nodes[80].get("widgets_values", [""])[0]
    frame_rate, total_duration, tile_duration, overlap_duration = DEFAULT_FRAME_RATE, DEFAULT_TOTAL_DURATION, DEFAULT_TILE_DURATION, DEFAULT_OVERLAP_DURATION
    # The source schedule node is intentionally read by title/type, because older
    # generated examples used a different numeric node id for it.
    schedule_node = next((node for node in wf["nodes"] if node.get("type") == "LTXVLoopingReferenceSchedule"), None)
    if schedule_node:
        values = schedule_node.get("widgets_values", [])
        frame_rate = float(values[0]) if len(values) > 0 else frame_rate
        total_duration = float(values[1]) if len(values) > 1 else total_duration
        tile_duration = float(values[2]) if len(values) > 2 else tile_duration
        overlap_duration = float(values[3]) if len(values) > 3 else overlap_duration
    frame_count, tile_size, overlap, tile_count = calculate_schedule(frame_rate, total_duration, tile_duration, overlap_duration)
    target_node = next(
        (node for node in wf["nodes"] if node.get("type") == "PrimitiveInt" and "height" in node.get("title", "").lower()),
        None,
    )
    target_height = int(target_node.get("widgets_values", [DEFAULT_TARGET_HEIGHT])[0]) if target_node else DEFAULT_TARGET_HEIGHT
    start_image = next((node for node in wf["nodes"] if node.get("type") == "LoadImage"), None)
    image_file = start_image.get("widgets_values", [""])[0] if start_image else ""
    timeline_data = json.dumps({
        "version": 2,
        "tile_prompts": [DEFAULT_TILE_PROMPT] * tile_count,
        "keyframes": [{"frame": 0, "imageFile": image_file}] if image_file else [],
        "video_segments": [],
        "ic_segments": [],
        "audio_segments": [],
        "use_custom_audio": False,
        "inpaint_audio": True,
        "use_ic_video_audio": False,
        "retake_mode": False,
        "retake": None,
        "ic_settings": {},
    }, separators=(",", ":"))

    director = _node(
        LOOPING_DIRECTOR_ID,
        "LTXLoopingDirector",
        "LTX Looping Director",
        [
            ("clip", "CLIP", {}),
            ("global_prompt", "STRING", {"widget": {"name": "global_prompt"}}),
            ("frame_rate", "FLOAT", {"widget": {"name": "frame_rate"}}),
            ("total_duration", "FLOAT", {"widget": {"name": "total_duration"}}),
            ("tile_duration", "FLOAT", {"widget": {"name": "tile_duration"}}),
            ("overlap_duration", "FLOAT", {"widget": {"name": "overlap_duration"}}),
            ("timeline_data", "STRING", {"widget": {"name": "timeline_data"}}),
            ("target_height", "INT", {"widget": {"name": "target_height"}}),
            ("reference_keyframe_index", "INT", {"widget": {"name": "reference_keyframe_index"}}),
        ],
        [
            ("positive", "CONDITIONING"), ("per_tile_conditionings", "CONDITIONING"),
            ("cond_images", "IMAGE"), ("cond_image_indices", "STRING"),
            ("temporal_tile_size", "INT"), ("temporal_overlap", "INT"),
            ("frame_count", "INT"), ("frame_rate", "FLOAT"),
            ("global_prompt", "STRING"), ("start_image", "IMAGE"),
            ("stage_1_width", "INT"), ("stage_1_height", "INT"),
            ("reference_image", "IMAGE"), ("director_plan", "LTX_LOOPING_DIRECTOR_PLAN"),
        ],
        widgets=[global_prompt, frame_rate, total_duration, tile_duration, overlap_duration, timeline_data, target_height, 0],
        pos=[1050, 700],
        size=[760, 560],
    )
    director["properties"].update({
        "timeline_data": timeline_data,
        "looping_director_timeline": timeline_data,
        "looping_director_settings": {
            "global_prompt": global_prompt,
            "frame_rate": frame_rate,
            "total_duration": total_duration,
            "tile_duration": tile_duration,
            "overlap_duration": overlap_duration,
            "target_height": target_height,
            "reference_keyframe_index": 0,
        },
        "has_serialized_properties": True,
    })

    sampler_inputs = [
        ("director_plan", "LTX_LOOPING_DIRECTOR_PLAN", {}),
        ("per_tile_conditionings", "CONDITIONING", {}), ("cond_images", "IMAGE", {}),
        ("cond_image_indices", "STRING", {}), ("model", "MODEL", {}), ("video_vae", "VAE", {}),
        ("audio_vae", "VAE", {}), ("latent_upscale_model", "LATENT_UPSCALE_MODEL", {}),
        ("pass1_noise", "NOISE", {}), ("pass1_sampler", "SAMPLER", {}), ("pass1_sigmas", "SIGMAS", {}), ("pass1_guider", "GUIDER", {}),
        ("pass2_noise", "NOISE", {}), ("pass2_sampler", "SAMPLER", {}), ("pass2_sigmas", "SIGMAS", {}), ("pass2_guider", "GUIDER", {}),
        ("pass_count", "COMBO", {"widget": {"name": "pass_count"}}),
        ("execution_mode", "COMBO", {"widget": {"name": "execution_mode"}}),
        ("ic_lora_name", "COMBO", {"widget": {"name": "ic_lora_name"}}),
        ("ic_lora_strength", "FLOAT", {"widget": {"name": "ic_lora_strength"}}),
    ]
    for name, default in (
        ("pass1_guiding_strength", 1.0), ("pass1_overlap_cond_strength", 0.5), ("pass1_cond_image_strength", 1.0),
        ("pass1_adain_factor", 0.0), ("pass1_guiding_start_step", 0), ("pass1_guiding_end_step", 1000),
        ("pass2_guiding_strength", 1.0), ("pass2_overlap_cond_strength", 0.5), ("pass2_cond_image_strength", 1.0),
        ("pass2_adain_factor", 0.0), ("pass2_guiding_start_step", 0), ("pass2_guiding_end_step", 1000),
        ("pass1_horizontal_tiles", 1), ("pass1_vertical_tiles", 1), ("pass1_spatial_overlap", 8),
        ("pass2_horizontal_tiles", 1), ("pass2_vertical_tiles", 1), ("pass2_spatial_overlap", 8),
    ):
        typ = "INT" if isinstance(default, int) else "FLOAT"
        sampler_inputs.append((name, typ, {"widget": {"name": name}}))
    sampler_inputs.extend([
        ("pass1_guiding_latent", "LATENT", {}), ("pass1_negative_index_latent", "LATENT", {}),
        ("pass1_negative_index_strength", "FLOAT", {"widget": {"name": "pass1_negative_index_strength"}}),
        ("pass1_normalizing_latent", "LATENT", {}), ("pass1_seed_offsets", "STRING", {"widget": {"name": "pass1_seed_offsets"}}),
        ("pass2_guiding_latent", "LATENT", {}), ("pass2_negative_index_latent", "LATENT", {}),
        ("pass2_negative_index_strength", "FLOAT", {"widget": {"name": "pass2_negative_index_strength"}}),
        ("pass2_normalizing_latent", "LATENT", {}), ("pass2_seed_offsets", "STRING", {"widget": {"name": "pass2_seed_offsets"}}),
        ("pass1_latent", "LATENT", {}), ("pass2_latent", "LATENT", {}),
        ("external_pass1_video_latent", "LATENT", {}),
        ("external_pass1_audio_latent", "LATENT", {}),
        ("checkpoint_policy", "COMBO", {"widget": {"name": "checkpoint_policy"}}),
        ("resume", "COMBO", {"widget": {"name": "resume"}}),
        ("checkpoint_prefix", "STRING", {"widget": {"name": "checkpoint_prefix"}}),
    ])
    sampler = _node(
        LOOPING_SAMPLER_ID,
        "LTXLoopingDirectorSampler",
        "LTX Looping Director — 1/2 Pass",
        sampler_inputs,
        [("av_latent", "LATENT"), ("video_latent", "LATENT"), ("audio_latent", "LATENT")],
        widgets=[2, "full", "None", 1.0, 1.0, 0.5, 1.0, 0.0, 0, 1000, 1.0, 0.5, 1.0, 0.0, 0, 1000, 1, 1, 8, 1, 1, 8, 1.0, 1.0, "0", 1.0, 1.0, "0", "off", "off", "ltx_looping_director"],
        pos=[2500, 500],
        size=[560, 1100],
    )
    note = _node(
        302,
        "MarkdownNote",
        "",
        [],
        [],
        widgets=[director_note(
            frame_rate,
            total_duration,
            tile_size / frame_rate,
            overlap / frame_rate,
            tile_count,
        )],
        pos=[2500, 1750],
        size=[720, 620],
    )
    note.update({"color": "#432", "bgcolor": "#653"})
    wf["nodes"].extend([director, sampler, note])
    nodes.update({LOOPING_DIRECTOR_ID: director, LOOPING_SAMPLER_ID: sampler, 302: note})

    add_link(11, 0, LOOPING_DIRECTOR_ID, find_input(director, "clip"), "CLIP")
    add_link(300, 13, LOOPING_SAMPLER_ID, find_input(sampler, "director_plan"), "LTX_LOOPING_DIRECTOR_PLAN")
    add_link(300, 1, LOOPING_SAMPLER_ID, find_input(sampler, "per_tile_conditionings"), "CONDITIONING")
    # Keep the source workflow's LTXVPreprocess on the conditioning-image path: it is
    # where the compressed start image entered the sampler in the base graph.
    preprocess = nodes[PREPROCESS_ID]
    preprocess["pos"] = [2100, 700]
    rewire(PREPROCESS_ID, "image", LOOPING_DIRECTOR_ID, 2, "IMAGE")
    add_link(PREPROCESS_ID, 0, LOOPING_SAMPLER_ID, find_input(sampler, "cond_images"), "IMAGE")
    add_link(300, 3, LOOPING_SAMPLER_ID, find_input(sampler, "cond_image_indices"), "STRING")
    add_link(300, 10, 30, find_input(nodes[30], "width"), "INT")
    add_link(300, 11, 30, find_input(nodes[30], "height"), "INT")
    add_link(300, 6, 30, find_input(nodes[30], "length"), "INT")
    add_link(30, 0, LOOPING_SAMPLER_ID, find_input(sampler, "pass1_latent"), "LATENT")
    add_link(13, 0, LOOPING_SAMPLER_ID, find_input(sampler, "model"), "MODEL")
    add_link(10, 2, LOOPING_SAMPLER_ID, find_input(sampler, "video_vae"), "VAE")
    add_link(12, 0, LOOPING_SAMPLER_ID, find_input(sampler, "audio_vae"), "VAE")
    add_link(14, 0, LOOPING_SAMPLER_ID, find_input(sampler, "latent_upscale_model"), "LATENT_UPSCALE_MODEL")
    for source, name in ((40, "pass1_noise"), (41, "pass1_sampler"), (42, "pass1_sigmas"), (43, "pass1_guider"),
                         (60, "pass2_noise"), (61, "pass2_sampler"), (62, "pass2_sigmas"), (63, "pass2_guider")):
        add_link(source, 0, LOOPING_SAMPLER_ID, find_input(sampler, name), {40: "NOISE", 41: "SAMPLER", 42: "SIGMAS", 43: "GUIDER", 60: "NOISE", 61: "SAMPLER", 62: "SIGMAS", 63: "GUIDER"}[source])

    rewire(22, "positive", LOOPING_DIRECTOR_ID, 0, "CONDITIONING")
    rewire(22, "frame_rate", LOOPING_DIRECTOR_ID, 7, "FLOAT")
    rewire(43, "model", 13, 0, "MODEL")
    rewire(43, "positive", 22, 0, "CONDITIONING")
    rewire(43, "negative", 21, 0, "CONDITIONING")
    rewire(63, "model", 13, 0, "MODEL")
    rewire(63, "positive", 22, 0, "CONDITIONING")
    rewire(63, "negative", 21, 0, "CONDITIONING")
    rewire(71, "samples", LOOPING_SAMPLER_ID, 1, "LATENT")
    rewire(71, "vae", 10, 2, "VAE")
    rewire(72, "samples", LOOPING_SAMPLER_ID, 2, "LATENT")
    rewire(72, "audio_vae", 12, 0, "VAE")
    rewire(73, "fps", LOOPING_DIRECTOR_ID, 7, "FLOAT")

    remove_ids = {
        1, 2, 4, 5, 6, 7, 16, 17, 18, 19, 20, 23, 24, 28, 30, 31, 32, 33, 34, 35,
        44, 50, 51, 52, 53, 64, 70, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
        *range(110, 183), 25,
    }
    remove_ids.discard(300)
    remove_ids.discard(301)
    remove_ids -= {PREPROCESS_ID, 10, 11, 12, 13, 14, 21, 22, 30, 40, 41, 42, 43, 60, 61, 62, 63, 71, 72, 73, 74, 302}
    wf["links"] = [link for link in wf["links"] if link[1] not in remove_ids and link[3] not in remove_ids]
    wf["nodes"] = [node for node in wf["nodes"] if node["id"] not in remove_ids]
    nodes = {node["id"]: node for node in wf["nodes"]}
    live_links = {link[0] for link in wf["links"]}
    for node in wf["nodes"]:
        for item in node.get("inputs", []):
            if item.get("link") not in live_links:
                item["link"] = None
        for output in node.get("outputs", []):
            output["links"] = [link_id for link_id in (output.get("links") or []) if link_id in live_links]

    wf["groups"] = []
    wf["last_node_id"] = max(node["id"] for node in wf["nodes"])
    wf["last_link_id"] = next_link[0]
    wf.setdefault("extra", {})["info"] = {
        "name": "LTX-2.3 Looping Director",
        "description": "Two-pass AV looping driven by a tile-owned LTX Looping Director plan.",
    }
    validate(wf)
    output = os.path.abspath(args.output)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(wf, stream, indent=2)
    print(f"Wrote {output} ({len(wf['nodes'])} nodes, {len(wf['links'])} links)")


def validate(wf):
    nodes = {node["id"]: node for node in wf["nodes"]}
    director = nodes.get(LOOPING_DIRECTOR_ID)
    sampler = nodes.get(LOOPING_SAMPLER_ID)
    assert director and director["type"] == "LTXLoopingDirector"
    assert sampler and sampler["type"] == "LTXLoopingDirectorSampler"
    assert not any(node["type"] == "LTXVLoopingSampler" for node in wf["nodes"])
    assert not any(node["type"] == "Power Lora Loader (rgthree)" for node in wf["nodes"])
    assert any(node["type"] == "MarkdownNote" for node in wf["nodes"])
    assert [item["name"] for item in director["inputs"]] == [
        "clip", "global_prompt", "frame_rate", "total_duration", "tile_duration", "overlap_duration", "timeline_data", "target_height", "reference_keyframe_index"
    ]
    assert [item["name"] for item in director["outputs"]][-1] == "director_plan"
    assert [item["name"] for item in sampler["outputs"]] == ["av_latent", "video_latent", "audio_latent"]
    assert "external_pass1_video_latent" in [item["name"] for item in sampler["inputs"]]
    assert "external_pass1_audio_latent" in [item["name"] for item in sampler["inputs"]]
    assert [item["name"] for item in sampler["inputs"]][:4] == [
        "director_plan", "per_tile_conditionings", "cond_images", "cond_image_indices"
    ]

    def linked(from_id, from_slot, to_id, name):
        slot = next(index for index, item in enumerate(nodes[to_id]["inputs"]) if item["name"] == name)
        assert any(link[1:5] == [from_id, from_slot, to_id, slot] for link in wf["links"]), f"missing {name} link"

    linked(LOOPING_DIRECTOR_ID, 13, LOOPING_SAMPLER_ID, "director_plan")
    linked(LOOPING_DIRECTOR_ID, 1, LOOPING_SAMPLER_ID, "per_tile_conditionings")
    linked(LOOPING_DIRECTOR_ID, 3, LOOPING_SAMPLER_ID, "cond_image_indices")
    # The source workflow's LTXVPreprocess survives on the conditioning-image path.
    preprocess = nodes.get(PREPROCESS_ID)
    assert preprocess and preprocess["type"] == "LTXVPreprocess", "LTXVPreprocess must be preserved"
    linked(LOOPING_DIRECTOR_ID, 2, PREPROCESS_ID, "image")
    linked(PREPROCESS_ID, 0, LOOPING_SAMPLER_ID, "cond_images")
    linked(13, 0, LOOPING_SAMPLER_ID, "model")
    linked(30, 0, LOOPING_SAMPLER_ID, "pass1_latent")
    linked(LOOPING_SAMPLER_ID, 1, 71, "samples")
    linked(LOOPING_SAMPLER_ID, 2, 72, "samples")
    linked(LOOPING_DIRECTOR_ID, 7, 73, "fps")
    link_ids = {link[0] for link in wf["links"]}
    for node in wf["nodes"]:
        for item in node.get("inputs", []):
            assert item.get("link") is None or item["link"] in link_ids
        for output in node.get("outputs", []):
            assert all(link_id in link_ids for link_id in output.get("links", []))


if __name__ == "__main__":
    main()
