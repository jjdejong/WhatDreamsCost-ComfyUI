#!/usr/bin/env python3
"""Derive a Looping Director workflow from the two-pass looping workflow.

Reads the hand-maintained two-pass AV I2V looping graph and produces a variant whose
per-tile prompts and keyframes come from a dedicated WhatDreamsCost
**LTX Looping Director** node, instead of the MultiPromptProvider + snippet/concatenate
machinery and the LTXVLoopingReferenceSchedule image scheduling.

This is a migration for the graph emitted by the companion
``generate_two_pass_i2v_looping.py`` script. It is intentionally not a generic graph
rewriter: workflows with different sampler or conditioning structures need an adapter
for those structures rather than being guessed from node positions.

What it changes (everything leans on the existing KJNodes Set/Get bus fabric, so only a
few SetNode sources are repointed and all consumers follow):

  * Adds one LTXLoopingDirector node with one fixed prompt slot per sampler tile.
  * Repoints the `tile_prompt_conditioning` bus  <- LoopingDirector.per_tile_conditionings
            the `scheduled_reference_images` bus <- LoopingDirector.cond_images
            the `reference_indices` bus          <- LoopingDirector.cond_image_indices
    (both LTXVLoopingSampler stages read those buses, so they pick up the new sources.)
  * Removes LTXVLoopingReferenceSchedule; timing comes directly from LoopingDirector.
  * Removes the now-obsolete per-tile prompt + late-reference subgraph.
  * Moves output sizing and Stage 1 dimension calculation into LoopingDirector.
  * Removes the standalone reference-size math and FPS conversion nodes.

Deliberately KEPT: the existing bus-driven EmptyLTXVLatentVideo /
LTXVEmptyLatentAudio shells (sized from LoopingDirector.frame_count) and the whole
guider/sampler/upscale/decode/output chain.

Run from this directory:

  python3 transform_to_director_looping.py

The default input is the companion ComfyUI-LTXVideo workflow when the two custom
nodes are installed beside each other. Use --base when the repositories are elsewhere.
The output defaults to LTX-2.3_Director_Looping.json beside this script.
"""

import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_NAME = "LTX-2.3_Two_Pass_I2V_Looping.json"
DEFAULT_OUTPUT = os.path.join(HERE, "LTX-2.3_Director_Looping.json")

LOOPING_DIRECTOR_ID = 300
TIME_SCALE = 8
DEFAULT_FRAME_RATE = 24.0
DEFAULT_TOTAL_DURATION = 48.0
DEFAULT_TILE_DURATION = 10.0
DEFAULT_OVERLAP_DURATION = 2.0
DEFAULT_TARGET_HEIGHT = 1088
DEFAULT_TILE_PROMPT = (
    "While the woman lowers her leg, the camera performs a slow orbit from a front "
    "view to a right quarter view. zhuanchang"
)


def aligned_frames(seconds, frame_rate, minimum):
    return max(minimum, round(seconds * frame_rate / TIME_SCALE) * TIME_SCALE)


def calculate_schedule(frame_rate, total_duration, tile_duration, overlap_duration):
    frame_count = max(
        TIME_SCALE + 1,
        math.floor((total_duration * frame_rate - 1) / TIME_SCALE) * TIME_SCALE + 1,
    )
    tile_size = min(aligned_frames(tile_duration, frame_rate, 24), 1000)
    overlap = min(
        aligned_frames(overlap_duration, frame_rate, 16),
        80,
        tile_size - TIME_SCALE,
    )
    latent_frames = (frame_count - 1) // TIME_SCALE + 1
    latent_tile_size = tile_size // TIME_SCALE
    latent_overlap = overlap // TIME_SCALE
    tile_count = max(
        1,
        math.ceil((latent_frames - latent_overlap) / (latent_tile_size - latent_overlap)),
    )
    return frame_count, tile_size, overlap, tile_count

# Obsolete nodes removed by the transform (rewired first, where needed, below).
REMOVE_CONTENT = [
    81,                      # MultiPromptProvider
    82, 85, 88, 91,          # Late Ref LoadImage
    83, 86, 89, 92,          # Tile snippet PrimitiveStringMultiline
    94, 95, 96, 97,          # ImageBatch (Ref Batch)
    84, 87, 90, 93,          # StringConcatenate (Global + Tile Snippet)
    98, 99, 100,             # StringConcatenate (Join Tile Prompts)
    184, 185, 186, 187,      # MergeString
    174, 175, 176, 177,      # Get global prompt for removed snippet joins
    179, 181,                # SetNode reference_image_batch / joined_tile_prompts
    80,                      # standalone "Global Positive Prompt" primitive
    # --- consolidation: timing into LoopingDirector ---
    24,                      # LTXVLoopingReferenceSchedule
    4,                       # fps PrimitiveFloat
    5,                       # I2V Enable primitive (this workflow is I2V-only)
    113,                     # Set_i2v_enable bus (its consumers are removed below)
    # --- consolidation: drop positive fallback encode ---
    20,                      # CLIPTextEncode "Global Prompt Fallback Encode"
    # --- consolidate output sizing and audio frame-rate handling ---
    7,                       # PrimitiveInt Final Height Target
    16,                      # GetImageSize
    17, 18, 19, 23,          # Final/Stage 1 dimension math
    34,                      # FPS -> INT conversion (audio accepts FLOAT)
    # --- consolidation: slim selected-reference image path ---
    32,                      # Stage 1 I2V Cond (LTXVImgToVideoConditionOnly)
    52,                      # Stage 2 I2V Cond
    1,                       # LoadImage (reference now taken from LoopingDirector.reference_image)
    110,                     # SetNode start_image (dead bus; source was LoadImage)
]

LAYOUT_POSITIONS = {
    # Inputs and model resources.
    2: [0, 260],
    6: [0, 680],
    10: [600, -60],
    11: [600, 100],
    12: [600, 260],
    13: [600, 400],
    14: [600, 540],
    21: [1150, 220],
    22: [1150, 400],
    25: [600, 680],
    # Director and its collapsed bus publishers.
    111: [400, 700],
    112: [400, 770],
    114: [400, 840],
    115: [400, 910],
    116: [400, 980],
    117: [400, 1050],
    118: [620, 700],
    119: [620, 770],
    120: [620, 840],
    170: [620, 910],
    173: [620, 980],
    121: [620, 1050],
    122: [840, 700],
    123: [840, 770],
    124: [840, 840],
    126: [840, 910],
    127: [840, 980],
    300: [1050, 700],
    # Stage-local publishers stay with the stage that consumes them.
    138: [2220, 1180],
    150: [3920, 440],
}

GROUP_LAYOUTS = {
    "Inputs + Timing": [-240, -90, 700, 1200],
    "Models + LoRAs": [560, -90, 670, 900],
    "Prompt Conditioning": [1060, -110, 660, 650],
    "Dimensions + Timing Buses": [300, 620, 1550, 700],
    "Stage 1 Base AV": [1850, -120, 1400, 1490],
    "Upscale + Stage 2": [3410, -120, 1540, 1270],
    "Final Output": [5110, 250, 650, 1120],
}

SOURCE_NODE_TYPES = {
    1: "LoadImage",
    7: "PrimitiveInt",
    11: "LTXAVTextEncoderLoader",
    20: "CLIPTextEncode",
    21: "CLIPTextEncode",
    22: "LTXVConditioning",
    24: "LTXVLoopingReferenceSchedule",
    30: "EmptyLTXVLatentVideo",
    31: "LTXVEmptyLatentAudio",
    32: "LTXVImgToVideoConditionOnly",
    33: "LTXVConcatAVLatent",
    35: "VAEEncode",
    44: "LTXVLoopingSampler",
    50: "LTXVSeparateAVLatent",
    51: "LTXVLatentUpsampler",
    52: "LTXVImgToVideoConditionOnly",
    53: "LTXVConcatAVLatent",
    64: "LTXVLoopingSampler",
    80: "PrimitiveStringMultiline",
}

SOURCE_BUS_NAMES = {
    "tile_prompt_conditioning",
    "scheduled_reference_images",
    "reference_indices",
    "global_prompt",
    "temporal_tile_size",
    "temporal_overlap",
    "frame_count",
    "fps",
    "stage_1_width",
    "stage_1_height",
    "preprocessed_start_image",
}


def validate_source(wf):
    nodes = {node["id"]: node for node in wf.get("nodes", [])}
    missing = [
        f"{node_id} ({node_type})"
        for node_id, node_type in SOURCE_NODE_TYPES.items()
        if node_id not in nodes or nodes[node_id].get("type") != node_type
    ]
    if missing:
        raise ValueError(
            "Unsupported source workflow: expected the companion two-pass graph "
            f"nodes, missing or mismatched {', '.join(missing)}"
        )

    buses = {
        node.get("widgets_values", [None])[0]
        for node in nodes.values()
        if node.get("type") == "SetNode"
    }
    missing_buses = sorted(SOURCE_BUS_NAMES - buses)
    if missing_buses:
        raise ValueError(
            "Unsupported source workflow: missing required SetNode buses: "
            + ", ".join(missing_buses)
        )


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
    searched = "\n  ".join(os.path.abspath(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find {BASE_NAME}. Searched:\n  {searched}\n"
        "Pass the LTXVideo workflow path with --base."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        help=f"path to the source {BASE_NAME} workflow",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="path for the generated Director workflow",
    )
    args = parser.parse_args(argv)
    base = resolve_base(args.base)
    output = os.path.abspath(args.output)

    with open(base) as f:
        wf = json.load(f)
    validate_source(wf)

    nodes = {n["id"]: n for n in wf["nodes"]}
    links = {l[0]: l for l in wf["links"]}
    next_link = [wf["last_link_id"]]

    def new_link(from_id, from_slot, to_id, to_slot, typ):
        next_link[0] += 1
        lid = next_link[0]
        wf["links"].append([lid, from_id, from_slot, to_id, to_slot, typ])
        links[lid] = wf["links"][-1]
        # bookkeeping
        fo = nodes[from_id]["outputs"][from_slot]
        fo.setdefault("links", [])
        if fo["links"] is None:
            fo["links"] = []
        fo["links"].append(lid)
        nodes[to_id]["inputs"][to_slot]["link"] = lid
        return lid

    def find_set(name):
        for n in wf["nodes"]:
            if n["type"] == "SetNode" and n["widgets_values"][0] == name:
                return n
        raise KeyError(name)

    def find_input_slot(node, name):
        for slot, input_data in enumerate(node.get("inputs", [])):
            if input_data.get("name") == name:
                return slot
        raise KeyError(f"{node['type']} input {name}")

    def rewire_input(to_id, to_slot, from_id, from_slot, typ):
        """Detach whatever currently feeds (to_id, to_slot) and wire it from a new source."""
        ip = nodes[to_id]["inputs"][to_slot]
        old = links.get(ip["link"])
        if old is not None:
            for op in nodes.get(old[1], {}).get("outputs", []) or []:
                if op.get("links"):
                    op["links"] = [x for x in op["links"] if x != old[0]]
            wf["links"] = [x for x in wf["links"] if x[0] != old[0]]
            links.pop(old[0], None)
        ip["link"] = None
        new_link(from_id, from_slot, to_id, to_slot, typ)

    # ── 1. Add LTXLoopingDirector ──
    global_prompt = nodes[80]["widgets_values"][0]
    schedule_values = nodes[24].get("widgets_values", [])
    frame_rate = float(schedule_values[0] if len(schedule_values) > 0 else DEFAULT_FRAME_RATE)
    total_duration = float(schedule_values[1] if len(schedule_values) > 1 else DEFAULT_TOTAL_DURATION)
    tile_duration = float(schedule_values[2] if len(schedule_values) > 2 else DEFAULT_TILE_DURATION)
    overlap_duration = float(schedule_values[3] if len(schedule_values) > 3 else DEFAULT_OVERLAP_DURATION)
    target_height = int(nodes[7].get("widgets_values", [DEFAULT_TARGET_HEIGHT])[0])
    frame_count, temporal_tile_size, temporal_overlap, tile_count = calculate_schedule(
        frame_rate,
        total_duration,
        tile_duration,
        overlap_duration,
    )
    timeline_data = json.dumps({
        "version": 1,
        "tile_prompts": [DEFAULT_TILE_PROMPT] * tile_count,
        "keyframes": [{"frame": 0, "imageFile": nodes[1]["widgets_values"][0]}],
    }, separators=(",", ":"))
    looping_director = {
        "id": LOOPING_DIRECTOR_ID,
        "type": "LTXLoopingDirector",
        "pos": [1150, 1500],
        "size": [760, 560],
        "flags": {},
        "order": LOOPING_DIRECTOR_ID,
        "mode": 0,
        "inputs": [
            {"name": "clip", "type": "CLIP", "link": None},
            {"name": "global_prompt", "type": "STRING", "link": None, "widget": {"name": "global_prompt"}},
            {"name": "frame_rate", "type": "FLOAT", "link": None, "widget": {"name": "frame_rate"}},
            {"name": "total_duration", "type": "FLOAT", "link": None, "widget": {"name": "total_duration"}},
            {"name": "tile_duration", "type": "FLOAT", "link": None, "widget": {"name": "tile_duration"}},
            {"name": "overlap_duration", "type": "FLOAT", "link": None, "widget": {"name": "overlap_duration"}},
            {"name": "timeline_data", "type": "STRING", "link": None, "widget": {"name": "timeline_data"}},
            {"name": "target_height", "type": "INT", "link": None, "widget": {"name": "target_height"}},
            {"name": "reference_keyframe_index", "type": "INT", "link": None, "widget": {"name": "reference_keyframe_index"}},
        ],
        "outputs": [
            {"name": "positive", "type": "CONDITIONING", "links": [], "slot_index": 0},
            {"name": "per_tile_conditionings", "type": "CONDITIONING", "links": [], "slot_index": 1},
            {"name": "cond_images", "type": "IMAGE", "links": [], "slot_index": 2},
            {"name": "cond_image_indices", "type": "STRING", "links": [], "slot_index": 3},
            {"name": "temporal_tile_size", "type": "INT", "links": [], "slot_index": 4},
            {"name": "temporal_overlap", "type": "INT", "links": [], "slot_index": 5},
            {"name": "frame_count", "type": "INT", "links": [], "slot_index": 6},
            {"name": "frame_rate", "type": "FLOAT", "links": [], "slot_index": 7},
            {"name": "global_prompt", "type": "STRING", "links": [], "slot_index": 8},
            {"name": "start_image", "type": "IMAGE", "links": [], "slot_index": 9},
            {"name": "stage_1_width", "type": "INT", "links": [], "slot_index": 10},
            {"name": "stage_1_height", "type": "INT", "links": [], "slot_index": 11},
            {"name": "reference_image", "type": "IMAGE", "links": [], "slot_index": 12},
        ],
        "properties": {
            "Node name for S&R": "LTXLoopingDirector",
            "timeline_data": timeline_data,
            "looping_director_timeline": timeline_data,
            "has_serialized_properties": True,
        },
        # Required widgets precede optional widgets in serialized V3 nodes.
        # timeline_data is required; the other displayed values are optional.
        "widgets_values": [
            timeline_data, global_prompt, frame_rate, total_duration,
            tile_duration, overlap_duration, target_height, 0,
        ],
        "title": "LTX Looping Director",
    }
    wf["nodes"].append(looping_director)
    nodes[LOOPING_DIRECTOR_ID] = looping_director

    # ── 2. Wire LTXLoopingDirector ──
    new_link(11, 0, LOOPING_DIRECTOR_ID, 0, "CLIP")

    # ── 3. Repoint SetNode sources to LoopingDirector ──
    def repoint(set_name, new_from_id, new_from_slot):
        sn = find_set(set_name)
        lid = sn["inputs"][0]["link"]
        l = links[lid]
        old_from = l[1]
        # detach from old source's outputs bookkeeping
        for op in nodes[old_from].get("outputs", []):
            if op.get("links"):
                op["links"] = [x for x in op["links"] if x != lid]
        l[1] = new_from_id
        l[2] = new_from_slot
        nodes[new_from_id]["outputs"][new_from_slot]["links"].append(lid)

    repoint("tile_prompt_conditioning", LOOPING_DIRECTOR_ID, 1)
    repoint("scheduled_reference_images", LOOPING_DIRECTOR_ID, 2)
    repoint("reference_indices", LOOPING_DIRECTOR_ID, 3)
    repoint("global_prompt", LOOPING_DIRECTOR_ID, 8)
    repoint("temporal_tile_size", LOOPING_DIRECTOR_ID, 4)
    repoint("temporal_overlap", LOOPING_DIRECTOR_ID, 5)
    repoint("frame_count", LOOPING_DIRECTOR_ID, 6)
    repoint("fps", LOOPING_DIRECTOR_ID, 7)
    repoint("stage_1_width", LOOPING_DIRECTOR_ID, 10)
    repoint("stage_1_height", LOOPING_DIRECTOR_ID, 11)

    # ── 5. Rewire consumers of removed nodes ──
    # Guider base positive from LoopingDirector.positive instead of the dropped encode.
    rewire_input(22, 0, LOOPING_DIRECTOR_ID, 0, "CONDITIONING")
    # Drop the I2V cond nodes (32/52): feed the empty/upscaled video latents straight to the
    # AV concats; frame-0 conditioning now comes from a LoopingDirector frame-0 keyframe.
    rewire_input(33, 0, 30, 0, "LATENT")   # Stage 1 AV concat video <- empty video latent
    rewire_input(53, 0, 51, 0, "LATENT")   # Stage 2 AV concat video <- spatial upscaler
    # The Director now owns output sizing and selects the scene/identity reference.
    # Keep the source workflow's LTXVPreprocess when present. Some workflows omit it,
    # in which case the selected reference feeds the existing image bus directly.
    preprocess = next(
        (n for n in wf["nodes"] if n["type"] == "LTXVPreprocess"),
        None,
    )
    if preprocess is not None:
        rewire_input(
            preprocess["id"],
            find_input_slot(preprocess, "image"),
            LOOPING_DIRECTOR_ID,
            12,
            "IMAGE",
        )
    else:
        reference_bus = find_set("preprocessed_start_image")
        rewire_input(reference_bus["id"], 0, LOOPING_DIRECTOR_ID, 12, "IMAGE")
    rewire_input(31, 2, LOOPING_DIRECTOR_ID, 7, "FLOAT")    # Empty audio <- frame_rate

    # ── 6. Remove obsolete subgraph + dangling links ──
    remove = set(REMOVE_CONTENT)
    wf["links"] = [
        l for l in wf["links"] if l[1] not in remove and l[3] not in remove
    ]
    live = {l[0] for l in wf["links"]}
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in remove]
    nodes = {n["id"]: n for n in wf["nodes"]}
    # clean bookkeeping for surviving nodes
    for n in wf["nodes"]:
        for ip in n.get("inputs", []):
            if ip.get("link") not in live:
                ip["link"] = None
        for op in n.get("outputs", []):
            if op.get("links"):
                op["links"] = [x for x in op["links"] if x in live]

    # ── 7. Prune GetNodes that now feed nothing ──
    def get_consumed(nid):
        return any(l[1] == nid for l in wf["links"])

    pruned = [
        n["id"] for n in wf["nodes"]
        if n["type"] == "GetNode" and not get_consumed(n["id"])
    ]
    wf["nodes"] = [n for n in wf["nodes"] if n["id"] not in set(pruned)]
    nodes = {n["id"]: n for n in wf["nodes"]}

    # These values remain visible in the node even when the corresponding sockets
    # are linked. Keep the display in sync with the Director and remove the stale
    # hard-coded reference frame list from the optional string widget.
    for sampler in wf["nodes"]:
        if sampler["type"] != "LTXVLoopingSampler":
            continue
        values = sampler.get("widgets_values")
        if not isinstance(values, list) or len(values) < 3:
            raise ValueError(
                f"LTXVLoopingSampler {sampler['id']} has an unexpected widget layout"
            )
        values[0] = temporal_tile_size
        values[1] = temporal_overlap
        values[-1] = ""

    for node in wf["nodes"]:
        if node["id"] in LAYOUT_POSITIONS:
            node["pos"] = LAYOUT_POSITIONS[node["id"]]

    # The source frame for late references is removed, so its group is no longer
    # meaningful. The remaining groups are resized around the cleaned layout.
    wf["groups"] = [
        group for group in wf.get("groups", [])
        if group.get("title") != "Late References + Tile Snippets"
    ]
    for group in wf.get("groups", []):
        if group.get("title") in GROUP_LAYOUTS:
            group["bounding"] = GROUP_LAYOUTS[group["title"]]

    wf["last_node_id"] = max(n["id"] for n in wf["nodes"])
    wf["last_link_id"] = next_link[0]
    wf.setdefault("extra", {}).setdefault("info", {})
    wf["extra"]["info"] = {
        "name": "LTX-2.3 Looping Director",
        "description": (
            "Two-pass AV I2V looping driven by LTX Looping Director: one fixed prompt "
            "per temporal tile plus multiple static image keyframes."
        ),
    }

    # VHS_VideoCombine serializes its last preview into widgets_values, which bakes an
    # absolute output path from whoever last ran the base workflow. It is stale UI state,
    # not config -- VHS rebuilds it on the next run -- so drop it rather than ship it.
    stripped = 0
    for n in wf["nodes"]:
        wv = n.get("widgets_values")
        if isinstance(wv, dict) and wv.pop("videopreview", None) is not None:
            stripped += 1

    validate(wf)

    with open(output, "w") as f:
        json.dump(wf, f, indent=2)
    print(f"Wrote {output}")
    if stripped:
        print(f"  stripped stale videopreview from {stripped} node(s)")
    print(f"  {len(wf['nodes'])} nodes, {len(wf['links'])} links; pruned GetNodes: {pruned}")


def validate(wf):
    nodes = {n["id"]: n for n in wf["nodes"]}
    looping_directors = [n for n in wf["nodes"] if n["type"] == "LTXLoopingDirector"]
    assert len(looping_directors) == 1, "expected exactly one LTXLoopingDirector"
    assert not any(n["type"] == "LTXDirector" for n in wf["nodes"]), "standard Director leaked into looping workflow"
    assert not any(n["type"] == "LTXLoopingBridge" for n in wf["nodes"]), "obsolete bridge leaked into looping workflow"
    assert not any(
        n["type"] == "StringConcatenate"
        and n.get("title", "").startswith(("Global + Tile", "Join Tile Prompts"))
        for n in wf["nodes"]
    ), "obsolete tile prompt joins leaked into workflow"
    assert not any(
        n["type"] == "PrimitiveBoolean" and n.get("title") == "I2V Enable"
        for n in wf["nodes"]
    ), "obsolete I2V toggle leaked into workflow"
    assert not any(
        n["type"] == "SetNode"
        and n.get("widgets_values", [None])[0] == "i2v_enable"
        for n in wf["nodes"]
    ), "obsolete I2V bus leaked into workflow"
    assert not any(
        group.get("title") == "Late References + Tile Snippets"
        for group in wf.get("groups", [])
    ), "obsolete late-reference group leaked into workflow"
    looping_director = looping_directors[0]
    assert [i["name"] for i in looping_director["inputs"]] == [
        "clip", "global_prompt", "frame_rate", "total_duration", "tile_duration",
        "overlap_duration", "timeline_data", "target_height", "reference_keyframe_index",
    ], "unexpected LTXLoopingDirector input order"
    assert [o["name"] for o in looping_director["outputs"]] == [
        "positive", "per_tile_conditionings", "cond_images", "cond_image_indices",
        "temporal_tile_size", "temporal_overlap", "frame_count", "frame_rate",
        "global_prompt", "start_image",
        "stage_1_width", "stage_1_height", "reference_image",
    ], "unexpected LTXLoopingDirector output order"
    assert len(looping_director["outputs"]) == 13, "unexpected LTXLoopingDirector output count"
    assert len(looping_director["widgets_values"]) == 8, "unexpected LTXLoopingDirector widget count"
    assert not any(n["type"] in {
        "GetImageSize", "ComfyMathExpression", "PrimitiveInt", "CM_FloatToInt",
    } and n.get("title") in {
        "Reference Image Size", "Align Final Height x64", "Final Width From Ref Aspect",
        "Stage 1 Width", "Stage 1 Height", "Final Height Target", "FPS→Int",
    } for n in wf["nodes"]), "obsolete sizing or FPS conversion nodes leaked into workflow"

    def find_node(node_type, predicate=lambda node: True):
        matches = [
            node for node in wf["nodes"]
            if node["type"] == node_type and predicate(node)
        ]
        assert len(matches) == 1, f"expected one {node_type} matching workflow contract"
        return matches[0]

    def input_slot(node, name):
        for slot, input_data in enumerate(node.get("inputs", [])):
            if input_data.get("name") == name:
                return slot
        raise AssertionError(f"{node['type']} has no {name} input")

    def assert_link(from_slot, to_node, to_slot, description):
        matches = [
            link for link in wf["links"]
            if link[1] == looping_director["id"]
            and link[2] == from_slot
            and link[3] == to_node["id"]
            and link[4] == to_slot
        ]
        assert len(matches) == 1, f"missing or duplicate {description} link"

    stage_width = find_node(
        "SetNode",
        lambda node: node.get("widgets_values", [None])[0] == "stage_1_width",
    )
    stage_height = find_node(
        "SetNode",
        lambda node: node.get("widgets_values", [None])[0] == "stage_1_height",
    )
    fps = find_node(
        "SetNode",
        lambda node: node.get("widgets_values", [None])[0] == "fps",
    )
    audio = find_node("LTXVEmptyLatentAudio")
    assert_link(10, stage_width, 0, "Stage 1 width")
    assert_link(11, stage_height, 0, "Stage 1 height")
    preprocess = [n for n in wf["nodes"] if n["type"] == "LTXVPreprocess"]
    assert len(preprocess) <= 1, "expected at most one LTXVPreprocess"
    if preprocess:
        preprocess = preprocess[0]
        assert_link(12, preprocess, input_slot(preprocess, "image"), "reference image")
    else:
        reference_bus = find_node(
            "SetNode",
            lambda node: node.get("widgets_values", [None])[0]
            == "preprocessed_start_image",
        )
        assert_link(12, reference_bus, 0, "reference image bus")
    assert_link(7, fps, 0, "FPS bus")
    assert_link(7, audio, input_slot(audio, "frame_rate"), "audio frame rate")

    values = looping_director["widgets_values"]
    _, expected_tile_size, expected_overlap, _ = calculate_schedule(
        float(values[2]), float(values[3]), float(values[4]), float(values[5])
    )
    for sampler in (n for n in wf["nodes"] if n["type"] == "LTXVLoopingSampler"):
        values = sampler.get("widgets_values", [])
        assert values[:2] == [expected_tile_size, expected_overlap], "sampler display timing is stale"
        assert values[-1] == "", "sampler display reference indices are stale"

    timeline = json.loads(looping_director["properties"]["timeline_data"])
    assert any(keyframe.get("frame") == 0 for keyframe in timeline["keyframes"]), "example needs a frame-0 keyframe"
    values = looping_director["widgets_values"]
    _, expected_tile_size, expected_overlap, expected_tile_count = calculate_schedule(
        float(values[2]), float(values[3]), float(values[4]), float(values[5])
    )
    assert len(timeline["tile_prompts"]) == expected_tile_count, "example needs one prompt per tile"
    assert all(prompt == DEFAULT_TILE_PROMPT for prompt in timeline["tile_prompts"]), "example tile prompts are stale"
    link_ids = set()
    for l in wf["links"]:
        lid, fid, fs, tid, ts, _ = l
        link_ids.add(lid)
        assert fid in nodes, f"link {lid}: missing from-node {fid}"
        assert tid in nodes, f"link {lid}: missing to-node {tid}"
        assert fs < len(nodes[fid].get("outputs", [])), f"link {lid}: bad from-slot {fs} on {fid}"
        assert ts < len(nodes[tid].get("inputs", [])), f"link {lid}: bad to-slot {ts} on {tid}"
    for n in wf["nodes"]:
        for i, ip in enumerate(n.get("inputs", [])):
            lk = ip.get("link")
            assert lk is None or lk in link_ids, f"node {n['id']} input[{i}] dangling link {lk}"
        for op in n.get("outputs", []):
            for lk in (op.get("links") or []):
                assert lk in link_ids, f"node {n['id']} output dangling link {lk}"
    print("  validate: OK (all links reference live nodes/slots)")


if __name__ == "__main__":
    main()
