# LTX-2.3 Looping Director

This two-pass AV I2V workflow uses the dedicated WhatDreamsCost
`LTXLoopingDirector` node to drive the Lightricks `LTXVLoopingSampler`.

The generator, `transform_to_director_looping.py`, is included beside this sample. It
reads `LTX-2.3_Two_Pass_I2V_Looping.json` from the companion `ComfyUI-LTXVideo`
custom node by default; pass `--base` when the repositories are installed elsewhere.

## Control surface

`LTXLoopingDirector` owns the looping schedule:

- `frame_rate`, `total_duration`, `tile_duration`, and `overlap_duration` are entered
  in seconds;
- the node derives the valid LTX frame count, tile size, and overlap and publishes
  those values to both looping samplers;
- the editor exposes exactly one tile prompt per temporal tile;
- new images can be dropped onto the timeline or added with **Add Image**;
- the timeline is prefilled with one empty slot at frame 0 and one on the last frame
  of every tile. A slot left empty is not a keyframe: nothing is emitted at that
  index;
- a tile-end keyframe makes that tile a "to last image" generation. The tile starts
  from the trailing overlap it inherits from the previous tile and is steered toward
  its own end keyframe, which the next tile then inherits as its start reference;
- dragging a marker makes its frame index manual, so it can be moved anywhere else on
  the 8-frame grid;
- `target_height` is the only output-size control. The frame-0 `start_image` supplies
  the aspect ratio and the final width is calculated and aligned automatically.

The global prompt is prepended to every tile prompt. Tile prompts describe the action
and camera transition for that tile and do not span tile boundaries.

`start_image` is the frame-0 timeline image and controls the output aspect and initial
frame path. `reference_image` is separate: the **External reference** selector chooses
any timeline keyframe for the external scene/identity anchor path. Its pixels are
resized to the final dimensions derived from `start_image`.

## Image preprocessing

The Director batches the conditioning images without applying `img_compression`. The
workflow's `LTXVPreprocess` node sits between the Director's `cond_images` output and
the combined sampler, so it stays the compression point for the start image and every
later keyframe, exactly as it was for the start image in the source workflow. The
sampler then resizes and internally preprocesses those images as before.

A text-to-video variant with no keyframes leaves `cond_images` empty; bypass or delete
the `LTXVPreprocess` node in that case and wire the Director straight to the sampler.

## Guide encoding

The **Guide encoding** row in the Director editor controls how Video, IC Video, and
Retake guides are resized and VAE-encoded: `crop` (center crop or stretch to fit), the
resampling method, and an optional tiled encode with its tile size and overlap. Tiled
encode trades speed for lower peak VRAM on large guides. These values are stored in the
timeline's `ic_settings` and are read by the sampler at encode time.

## Before running

1. Set the global prompt and edit one prompt card per tile. The generated example
   includes a representative camera-transition snippet in every card.
2. Drop the frame-0 start image and any later keyframes onto the timeline. Use the
   **External reference** selector to choose the image used by the anchor path.
3. Set the duration controls in seconds and `target_height`. The sampler timing
   widgets are driven by the Director outputs.
4. Restart ComfyUI after installing or updating `WhatDreamsCost-ComfyUI`.

The transform targets the companion workflow's KJNodes bus and
`LTXVLoopingSampler` socket contract. It is not a generic converter for arbitrary
sampler or conditioning graphs.

## Kept and removed graph parts

Kept: the bus-driven empty video/audio latent shells, both guider/sampler stages,
spatial upscale, tiled VAE decode, audio decode, video output, and `LTXVPreprocess` on
the conditioning-image path.

Removed: the standard Director, Looping Bridge, MultiPromptProvider and prompt
assembly nodes, late-reference loaders/batches, `LTXVLoopingReferenceSchedule`,
standalone timing/sizing primitives, the FPS conversion node, the two I2V conditioning
nodes, and the separate start-image loader.
