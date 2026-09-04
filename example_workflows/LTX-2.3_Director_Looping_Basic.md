# LTX-2.3 Basic Looping Director

LTX-2.3_Director_Looping_Basic.json is a small two-pass AV example using both
the **LTXLoopingDirector** and **LTXLoopingDirectorSampler** nodes.

The Director owns the user-facing timeline. It converts the duration controls to
the LTX frame grid, creates one conditioning per tile, batches the keyframes, and
publishes a director_plan. The combined sampler consumes that plan, runs the
looping sampler for pass 1, performs the latent upscale, runs pass 2, and exposes
concatenated AV, video-only, and audio-only latent outputs. VAE decoding remains
outside the combined sampler, so a model-unload/debug node can be placed before a
tiled VAE decoder.

## Example settings

- Frame rate: 24 fps
- Total duration: 30 seconds
- Tile duration: 12 seconds
- Tile overlap: 2 seconds
- Target height: 1088
- Temporal schedule: three tiles, with 288 frame tiles and 48 frame overlap
- Keyframes: frame 0, frame 264 (11 seconds), and frame 504 (21 seconds)

The 8-frame LTX grid makes the generated clip 713 frames long (29.67 seconds
of frame time plus the initial frame). The two later keyframes sit in the middle
of the overlaps, so they are treated as end references for the earlier tile while
helping the following tile continue the motion.

The sample uses example.png for all three keyframe entries so it can be opened
without additional media. Replace those three entries with distinct images, or
drop images onto the Director timeline. The first image is the start image and
determines the aspect ratio; target_height determines the final height and the
width is calculated automatically.

## Combined sampler settings

The sampler is configured for:

- pass_count = 2
- execution_mode = full
- separate pass-1 and pass-2 noise, sampler, sigmas, and guider inputs
- the latent upscale model between passes
- horizontal_tiles = 1, vertical_tiles = 1
- checkpoint saving and resume disabled by default

Enable save_checkpoints and set a unique checkpoint_prefix when tile
checkpoints are useful. resume = latest resumes only when the saved manifest
matches the current plan, media, conditioning, and sampler configuration.
external_pass1_video_latent and external_pass1_audio_latent are optional
inputs for execution_mode = pass_2_from_inputs.

## Prompt pattern

The global prompt is common scene, identity, style, lighting, and audio context.
Each tile prompt is one flowing paragraph describing only the action and camera
transition for that tile.

**Global prompt — five things, once:**

1. Shot type, subject, setting: *"Continuous cinematic live-action shot of the same
   adult couple dancing in a warmly lit rehearsal studio"*.
2. What must not drift: identity, wardrobe, lighting, environment, stated as holding
   across the entire clip.
3. Pace and continuity: *"a natural regular pace without a cut or reset"*.
4. Composition and look: framing, key surfaces, light quality, grain.
5. Audio: diegetic sources by name, and what must not appear — *"no music, no
   dialogue"*.

**Tile prompt — five beats, in order:**

1. **Continuity opener.** Say the shot is carrying on, never that a scene begins:
   *"The couple continues…"*, *"Without a cut, …"*, *"The choreography continues
   seamlessly…"*.
2. **What moves.** Name the bodies and one or two specific physical actions, not a
   whole routine: *"the woman lowers her raised leg into a controlled step and the man
   guides her through a slow turn"*.
3. **Camera move with the view it settles into.** Give a start and an end framing:
   *"a gentle orbit from the frontal view toward a left three-quarter view"*.
4. **Landing clause.** Name where the tile arrives: *"ending near the second later
   reference image"*. This is the beat that matters most — the tile is generated
   toward its end keyframe, so the prompt has to finish where the image does. The
   final tile has no next reference, so it holds instead: *"then holds on their
   relaxed faces … through the final frames"*.
5. **Continuity tail.** One physical detail that spans the seam: synchronized
   footfalls, steady breathing, natural eye contact.

Present tense throughout, one paragraph per tile, physical acting cues rather than
emotion labels, and enough detail to fill the tile's duration.

The example below follows that pattern:

**Global prompt**

> Continuous cinematic live-action shot of the same adult couple dancing in a
> warmly lit rehearsal studio, with consistent identity, wardrobe, lighting, and
> environment across the entire clip. The action proceeds at a natural regular
> pace without a cut or reset. A medium-wide composition shows the wooden floor,
> soft window light, subtle film grain, and realistic body movement. Diegetic
> audio only: measured footsteps, fabric movement, gentle body contact, and quiet
> exerted breathing; no music, no dialogue.

**Tile 0**

> The couple continues the dance as the woman lowers her raised leg into a
> controlled step and the man guides her through a slow turn. The camera makes a
> gentle orbit from the frontal view toward a left three-quarter view, ending near
> the first later reference image. Their footfalls remain synchronized and
> unhurried.

**Tile 1**

> Without a cut, the couple flows into the next phrase: the woman extends one arm
> and steps backward while the man follows with a measured pivot. The camera
> tracks sideways and eases into a right three-quarter view, ending near the
> second later reference image. Small pauses and natural eye contact preserve the
> continuous performance.

**Tile 2**

> The choreography continues seamlessly as the couple completes the turn and
> settles into another linked step. The camera slowly pushes in while drifting to
> a balanced front-right view, then holds on their relaxed faces, subtle
> micro-expressions, and steady breathing through the final frames.

Read the three tiles together and the pattern is visible: each opens on continuity,
moves one or two bodies, drives the camera to a named view, lands on its end
reference, and closes on a physical through-line. Tile 2 is the exception that
proves the rule — it has no next reference, so its landing clause becomes a hold.

For image-to-video, let the images establish the static appearance and use the
prompt primarily for what moves next. Keep the global prompt stable and avoid
contradicting it in a tile prompt.

See the [LTX-2.3 prompt guide](https://ltx.io/blog/ltx-2-3-prompt-guide) for the
full prompting guidance.
