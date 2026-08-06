# LTX-2.3 Looping Director

This two-pass AV I2V workflow uses the dedicated WhatDreamsCost
`LTXLoopingDirector` node to drive the Lightricks `LTXVLoopingSampler`.

This checked-in sample was generated from `LTX-2.3_Two_Pass_I2V_Looping.json` by
`transform_to_director_looping.py` in the companion `ComfyUI-LTXVideo` node. Restart
ComfyUI after installing or updating `WhatDreamsCost-ComfyUI`.

## Control surface

`LTXLoopingDirector` owns the looping-specific schedule:

- `global_prompt` is prepended to every tile;
- `first_tile_prompt` is added only to tile 0;
- the editor exposes exactly one tile prompt per temporal sampler tile;
- image keyframes can be placed at multiple absolute pixel-frame positions;
- `frame_count`, `temporal_tile_size`, `temporal_overlap`, and `frame_rate` are emitted
  to the rest of the graph;
- the keyframe at frame 0 supplies `start_image` for dimensions and the identity anchor.

Tile prompts are fixed to their tile. There are no arbitrary prompt spans, overlapping
prompt blocks, audio/motion tracks, retakes, or per-keyframe strengths.

The node's `positive` output feeds the base guider. Its `per_tile_conditionings` output
feeds both looping samplers through the `tile_prompt_conditioning` bus. Its image batch
and index string feed the scheduled-reference buses. The graph continues to use the
plain post-LoRA model; the standard Director's Prompt Relay model patch is not involved.

## Before running

1. Set the global prompt and, if needed, the first-tile continuation prompt.
2. Enter one prompt in each tile card. Empty tile cards use the global prompt alone.
3. Upload the required static images with **Add images**, select a frame on the ruler,
   and drag markers to adjust them. Place the starting image at frame `0`.
   The generated example carries over `reference_image.png` as that initial keyframe.
4. Keep `frame_count`, tile size, and overlap aligned with the sampler settings. The
   defaults are `121`, `240`, and `64`; frame counts and image positions use the LTX
   8-frame temporal grid.
5. Set keyframe resize/compression options only when needed. All keyframes are resized
   to one batch size before being passed to the sampler.

Multiple keyframes in the same tile are supported by the existing looping sampler and
remain separate images in the conditioning batch.

## Kept and removed graph parts

Kept: the bus-driven empty video/audio latent shells, both guider/sampler stages, spatial
upscale, tiled VAE decode, audio decode, video output, and the start-image preprocessing
and identity-anchor path.

Removed: the standard Director, Looping Bridge, MultiPromptProvider and prompt assembly
nodes, late-reference loaders/batches, `LTXVLoopingReferenceSchedule`, the standalone
global prompt/fps primitives, the fallback positive encode, the two I2V conditioning
nodes, and the separate start-image loader.
