const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const TIME_SCALE = 8;
const DEFAULT_FRAME_RATE = 24;
const DEFAULT_TOTAL_DURATION = 48;
const DEFAULT_TILE_DURATION = 10;
const DEFAULT_OVERLAP_DURATION = 2;
const DEFAULT_TARGET_HEIGHT = 1088;
const DIRECTOR_SCHEMA_VERSION = 2;
const DEFAULT_TILE_PROMPT = "The couple continues the choreography at a regular pace while the camera makes a slow orbit toward the next tile's end reference image.";
const DEFAULT_SETTINGS = {
  frame_rate: DEFAULT_FRAME_RATE,
  total_duration: DEFAULT_TOTAL_DURATION,
  tile_duration: DEFAULT_TILE_DURATION,
  overlap_duration: DEFAULT_OVERLAP_DURATION,
  target_height: DEFAULT_TARGET_HEIGHT,
  reference_keyframe_index: 0,
};

// The Director's own stylesheet (id "prompt-relay-styles") is injected globally by
// ltx_director.js, which loads from the same web directory. Reusing its `pr-` classes
// keeps both editors visually identical without copying its rules or editing it.
// These icons are duplicated rather than imported because that file exports nothing.
const ICONS = {
  upload: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>`,
  audio: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>`,
  motion: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>`,
  video: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>`,
  trash: `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>`,
  retake: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/></svg>`,
  plus: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>`,
};

function hideWidget(widget) {
  if (!widget) return;
  widget.hidden = true;
  widget.options = widget.options || {};
  widget.options.hidden = true;
  if (!window.LiteGraph || !window.LiteGraph.vueNodesMode) {
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
  }
}

function parseTimeline(value) {
  let data = {};
  try {
    data = value ? JSON.parse(value) : {};
  } catch (error) {
    console.warn("[LTXLoopingDirector] Invalid timeline_data; starting empty", error);
  }
  if (!data || typeof data !== "object" || Array.isArray(data)) data = {};
  const legacyVersion = Number(data.version || 1) < 2;
  return {
    version: 2,
    tile_prompts: Array.isArray(data.tile_prompts) ? data.tile_prompts.map(value => String(value ?? "")) : [],
    keyframes: Array.isArray(data.keyframes) ? data.keyframes.filter(item => item && typeof item === "object") : [],
    overflow_tile_prompts: Array.isArray(data.overflow_tile_prompts) ? data.overflow_tile_prompts.map(value => String(value ?? "")) : [],
    video_segments: Array.isArray(data.video_segments) ? data.video_segments.filter(item => item && typeof item === "object") : [],
    ic_segments: Array.isArray(data.ic_segments) ? data.ic_segments.filter(item => item && typeof item === "object") : [],
    audio_segments: Array.isArray(data.audio_segments) ? data.audio_segments.filter(item => item && typeof item === "object") : [],
    use_custom_audio: Boolean(data.use_custom_audio),
    inpaint_audio: data.inpaint_audio !== false,
    use_ic_video_audio: Boolean(data.use_ic_video_audio),
    retake_mode: Boolean(data.retake_mode || data.retakeMode),
    retake: data.retake || data.retakeVideo || null,
    ic_settings: normalizeIcSettings(data.ic_settings),
    legacyVersion,
  };
}

const IC_SETTING_DEFAULTS = {
  crop: "center",
  upscale_method: "bilinear",
  use_tiled_encode: false,
  tile_size: 256,
  tile_overlap: 64,
};

const IC_UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"];

function normalizeIcSettings(raw) {
  const source = raw && typeof raw === "object" ? raw : {};
  const clampInt = (value, fallback, min, max) => {
    const number = Math.round(Number(value));
    if (!Number.isFinite(number)) return fallback;
    return Math.max(min, Math.min(max, number));
  };
  return {
    crop: source.crop === "disabled" ? "disabled" : IC_SETTING_DEFAULTS.crop,
    upscale_method: IC_UPSCALE_METHODS.includes(source.upscale_method)
      ? source.upscale_method
      : IC_SETTING_DEFAULTS.upscale_method,
    use_tiled_encode: Boolean(source.use_tiled_encode),
    tile_size: clampInt(source.tile_size, IC_SETTING_DEFAULTS.tile_size, 64, 512),
    tile_overlap: clampInt(source.tile_overlap, IC_SETTING_DEFAULTS.tile_overlap, 16, 256),
  };
}

// Retake owns whole tiles. The list is always a contiguous, in-range run so the
// backend mask and the editor agree on what is regenerated.
function retakeTileList(retake, chunkCount) {
  const last = Math.max(0, chunkCount - 1);
  const clamp = value => Math.max(0, Math.min(last, Math.round(Number(value) || 0)));
  let tiles = Array.isArray(retake?.tiles) ? retake.tiles.map(clamp) : [];
  if (!tiles.length && retake && retake.tile != null) tiles = [clamp(retake.tile)];
  if (!tiles.length) tiles = [0];
  const start = Math.min(...tiles);
  const end = Math.max(...tiles);
  return Array.from({ length: end - start + 1 }, (_, index) => start + index);
}

function setRetakeTiles(retake, start, end, lastTile) {
  const first = Math.max(0, Math.min(lastTile, Math.min(start, end)));
  const final = Math.max(0, Math.min(lastTile, Math.max(start, end)));
  retake.tiles = Array.from({ length: final - first + 1 }, (_, index) => first + index);
  // Older readers still look at the single-tile key.
  retake.tile = first;
}

function numericWidget(node, name, fallback) {
  const widget = node.widgets?.find(item => item.name === name);
  const value = Number(widget?.value);
  return Number.isFinite(value) ? value : fallback;
}

function serializedSetting(node, name) {
  const value = node.widgets?.find(widget => widget.name === name)?.value;
  if (name === "global_prompt") return typeof value === "string" ? value : "";
  const number = Number(value);
  return Number.isFinite(number) ? number : DEFAULT_SETTINGS[name];
}

function alignedFrames(seconds, frameRate, minimum) {
  return Math.max(minimum, Math.round(seconds * frameRate / TIME_SCALE) * TIME_SCALE);
}

function loopingSchedule(node) {
  const frameRate = Math.max(0.01, numericWidget(node, "frame_rate", DEFAULT_FRAME_RATE));
  const totalDuration = Math.max(0.1, numericWidget(node, "total_duration", DEFAULT_TOTAL_DURATION));
  const tileDuration = Math.max(0.1, numericWidget(node, "tile_duration", DEFAULT_TILE_DURATION));
  const overlapDuration = Math.max(0, numericWidget(node, "overlap_duration", DEFAULT_OVERLAP_DURATION));
  const frameCount = Math.max(
    TIME_SCALE + 1,
    Math.floor((totalDuration * frameRate - 1) / TIME_SCALE) * TIME_SCALE + 1,
  );
  const tileSize = Math.min(alignedFrames(tileDuration, frameRate, 24), 1000);
  const overlap = Math.min(alignedFrames(overlapDuration, frameRate, 16), 80, tileSize - TIME_SCALE);
  const latentFrames = Math.floor((frameCount - 1) / TIME_SCALE) + 1;
  const latentTileSize = tileSize / TIME_SCALE;
  const latentOverlap = overlap / TIME_SCALE;
  const stride = latentTileSize - latentOverlap;
  const count = Math.max(1, Math.ceil((latentFrames - latentOverlap) / stride));
  const chunks = Array.from({ length: count }, (_, index) => {
    const start = index * stride * TIME_SCALE;
    return {
      startFrame: start,
      endFrame: Math.min(start + tileSize, frameCount),
    };
  });
  const finalFrame = Math.floor((frameCount - 1) / TIME_SCALE) * TIME_SCALE;
  const margin = Math.floor(overlap / 2);
  const referenceFrames = [0];
  for (let index = 0; index < count; index += 1) {
    let frame = Math.min(index * (tileSize - overlap) + tileSize - margin, finalFrame);
    frame -= frame % TIME_SCALE;
    if (!referenceFrames.includes(frame)) referenceFrames.push(frame);
  }
  return { frameRate, frameCount, tileSize, overlap, chunks, referenceFrames };
}

function tileCount(node) {
  return loopingSchedule(node).chunks.length;
}

function snapFrame(frame, frameCount) {
  const last = Math.max(0, frameCount - 1);
  return Math.max(0, Math.min(last, Math.round(frame / TIME_SCALE) * TIME_SCALE));
}

function frameCount(node) {
  return loopingSchedule(node).frameCount;
}

function frameLabel(frame, node) {
  const rate = numericWidget(node, "frame_rate", 24);
  const mode = node._loopingDirectorDisplayMode || "frames";
  if (mode === "seconds") return `${(frame / rate).toFixed(2)}s`;
  return `${frame}f`;
}

function makeButton(label, title, handler, icon) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "pr-btn";
  if (icon) {
    button.innerHTML = icon;
    button.appendChild(document.createTextNode(` ${label}`));
  } else {
    button.textContent = label;
  }
  button.title = title || label;
  button.addEventListener("click", handler);
  return button;
}

function makeIconButton(label, title, handler, icon, danger) {
  const button = makeButton(label, title, handler, icon);
  if (danger) button.classList.add("pr-btn-danger");
  return button;
}

class LoopingDirectorEditor {
  constructor(node, container) {
    this.node = node;
    this.container = container;
    this.selectedFrame = 0;
    this.selectedKeyframe = null;
    this.displayMode = node._loopingDirectorDisplayMode || "seconds";
    this.timeline = parseTimeline(node.widgets?.find(w => w.name === "timeline_data")?.value);
    this._build();
    this._bindTimingWidgets();
    this.refresh(true);
  }

  _build() {
    this.container.className = "pr-wrapper ld-editor";
    this.container.innerHTML = `
      <style>
        /* Layout and looping-specific chrome. Buttons, prompt panels and readouts
           reuse the Director's own pr- classes so both editors look the same. */
        .ld-editor { font-size: 12px; color: #e0e0e0; padding-bottom: 4px; }
        .ld-section-label { color: #666; margin: 6px 0 4px; font-size: 9px; font-weight: bold; text-transform: uppercase; letter-spacing: .5px; user-select: none; }
        .ld-status { color: #aaa; font-size: 11px; }
        .ld-warning { color: #e6b35a; }
        .ld-keyframe-strip { height: 94px; position: relative; overflow: hidden; border: 1px solid #111; border-radius: 6px; background: #2a2a2a; cursor: crosshair; transition: border-color .2s ease; }
        .ld-keyframe-strip.dragover { border-color: #888; background: rgba(255,255,255,.05); }
        .ld-tile-band { position: absolute; top: 0; bottom: 0; border-right: 1px solid rgba(255,255,255,.10); background: rgba(255,255,255,.02); pointer-events: none; }
        .ld-tile-band:nth-child(even) { background: rgba(255,255,255,.05); }
        .ld-frame-marker { position: absolute; top: 4px; width: 92px; height: 76px; transform: translateX(-50%); background: #222; border: 1px solid #111; border-radius: 6px; padding: 3px; box-sizing: border-box; cursor: grab; z-index: 3; transition: border-color .2s ease, background .2s ease; }
        .ld-frame-marker:hover { background: #333; border-color: #555; }
        .ld-frame-marker.reference { border-color: #d7ad63; }
        .ld-frame-marker.auto { border-color: #3a4a3e; }
        .ld-frame-marker.selected { border-color: #888; box-shadow: 0 0 0 1px #555; }
        .ld-frame-marker.invalid { border-color: #cc4444; }
        .ld-frame-marker img { display: block; width: 84px; height: 52px; object-fit: contain; background: #181818; border-radius: 3px; pointer-events: none; }
        .ld-frame-marker span { display: block; text-align: center; color: #aaa; font-size: 10px; line-height: 15px; white-space: nowrap; overflow: hidden; }
        .ld-frame-marker .ld-delete { position: absolute; right: -6px; top: -7px; width: 17px; height: 17px; padding: 0; line-height: 1; border-radius: 9px; border: 1px solid #111; background: #222; color: #e0e0e0; cursor: pointer; transition: all .2s ease; }
        .ld-frame-marker .ld-delete:hover { background: #4a1515; border-color: #cc4444; color: #ffaaaa; }
        /* An empty slot is a placeholder only: it holds a position for a tile but is
           not a keyframe until an image is dropped on it. */
        .ld-frame-marker.empty { border-style: dashed; border-color: #444; background: #1e1e1e; cursor: pointer; }
        .ld-frame-marker.empty:hover { border-color: #888; background: #262626; }
        .ld-frame-marker.empty .ld-slot-drop { display: flex; align-items: center; justify-content: center; width: 84px; height: 52px; border-radius: 3px; background: #181818; color: #555; pointer-events: none; }
        .ld-frame-marker.empty.dragover { border-color: #888; border-style: solid; background: rgba(255,255,255,.06); }
        .retake-locked .ld-frame-marker { cursor: default; opacity: .55; }
        .ld-prompts { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 6px; padding-bottom: 2px; }
        .ld-prompt-cell { height: 118px; }
        .ld-media { display: grid; gap: 6px; margin-bottom: 2px; }
        .ld-media-lane { border: 1px solid #333; border-radius: 6px; background: #1e1e1e; padding: 6px 10px; transition: border-color .2s ease, background .2s ease; }
        .ld-media-lane.dragover { border-color: #888; background: rgba(255,255,255,.05); }
        .ld-media-head { display: flex; align-items: center; gap: 8px; color: #fff; font-size: 11px; font-weight: 600; }
        .ld-media-head .pr-btn { margin-left: auto; padding: 3px 9px; }
        .ld-media-items { display: grid; gap: 4px; margin-top: 6px; }
        .ld-media-item { display: grid; grid-template-columns: minmax(0, 1fr) 52px 52px 52px 52px 22px; gap: 6px; align-items: center; color: #aaa; font-size: 11px; }
        .ld-media-item > span:first-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .ld-media-item input[type=number], .ld-media-item select { font-size: 11px; color: #fff; background: #222; border: 1px solid #444; border-radius: 4px; text-align: center; padding: 3px; box-sizing: border-box; width: 100%; }
        .ld-media-item .ld-delete { padding: 1px 6px; background: #222; color: #e0e0e0; border: 1px solid #111; border-radius: 4px; cursor: pointer; transition: all .2s ease; }
        .ld-media-item .ld-delete:hover { background: #4a1515; border-color: #cc4444; color: #ffaaaa; }
        .ld-media-item label { display: flex; align-items: center; gap: 4px; white-space: nowrap; }
      </style>
      <div class="pr-toolbar">
        <div class="pr-actions"></div>
        <div class="pr-right-group">
          <span class="pr-segment-bounds"></span>
          <span class="pr-timecode"></span>
        </div>
      </div>
      <div class="ld-section-label">Keyframes — one slot per tile at frame 0 and each tile overlap. Drop an image on a slot, or leave it empty to use no keyframe there.</div>
      <div class="ld-keyframe-strip"></div>
      <div class="ld-media"></div>
      <div class="ld-section-label">Prompts — one per looping tile</div>
      <div class="ld-prompts"></div>
    `;
    this.toolbar = this.container.querySelector(".pr-actions");
    this.boundsDisplay = this.container.querySelector(".pr-segment-bounds");
    this.timecodeDisplay = this.container.querySelector(".pr-timecode");
    this.strip = this.container.querySelector(".ld-keyframe-strip");
    this.media = this.container.querySelector(".ld-media");
    this.prompts = this.container.querySelector(".ld-prompts");

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.multiple = true;
    fileInput.style.display = "none";
    fileInput.addEventListener("change", async event => {
      await this._uploadFiles(Array.from(event.target.files || []));
      fileInput.value = "";
    });
    this.fileInput = fileInput;
    this.container.appendChild(fileInput);

    this.mediaInputs = {};
    for (const kind of ["video", "ic", "audio", "retake"]) {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = kind === "audio" ? "audio/*,video/*" : kind === "ic" ? "image/*,video/*" : "video/*";
      input.multiple = kind === "audio";
      input.style.display = "none";
      input.addEventListener("change", async event => {
        await this._uploadMediaFiles(Array.from(event.target.files || []), kind);
        input.value = "";
      });
      this.mediaInputs[kind] = input;
      this.container.appendChild(input);
    }

    this.strip.addEventListener("click", event => {
      if (event.target.closest(".ld-frame-marker")) return;
      const rect = this.strip.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width)));
      this.selectedFrame = snapFrame(ratio * (frameCount(this.node) - 1), frameCount(this.node));
      this._renderToolbar();
    });
    this.strip.addEventListener("dragover", event => {
      if (Array.from(event.dataTransfer?.items || []).some(item => item.kind === "file")) {
        event.preventDefault();
        this.strip.classList.add("dragover");
      }
    });
    this.strip.addEventListener("dragleave", () => this.strip.classList.remove("dragover"));
    this.strip.addEventListener("drop", async event => {
      event.preventDefault();
      this.strip.classList.remove("dragover");
      await this._uploadFiles(Array.from(event.dataTransfer?.files || []));
    });
  }

  _bindTimingWidgets() {
    for (const name of ["frame_rate", "total_duration", "tile_duration", "overlap_duration", "reference_keyframe_index"]) {
      const widget = this.node.widgets?.find(item => item.name === name);
      if (!widget || widget._loopingDirectorBound) continue;
      const callback = widget.callback;
      widget.callback = value => {
        if (callback) callback.call(widget, value);
        this.refresh();
      };
      widget._loopingDirectorBound = true;
    }
  }

  setTimeline(value) {
    this.timeline = parseTimeline(typeof value === "string" ? value : JSON.stringify(value));
    this.refresh(true);
  }

  _ensurePromptCount(count) {
    const prompts = this.timeline.tile_prompts;
    let changed = false;
    if (prompts.length > count) {
      this.timeline.overflow_tile_prompts = prompts.slice(count).concat(this.timeline.overflow_tile_prompts || []);
      this.timeline.tile_prompts = prompts.slice(0, count);
      changed = true;
      if (this.timeline.overflow_tile_prompts.some(prompt => prompt.trim())) {
        this.node._loopingDirectorWarning = "Some tile prompts are outside the current clip and were retained.";
      }
    }
    while (this.timeline.tile_prompts.length < count) {
      const restored = this.timeline.overflow_tile_prompts?.shift();
      this.timeline.tile_prompts.push(restored ?? DEFAULT_TILE_PROMPT);
      changed = true;
    }
    this.node._loopingDirectorWarning = this.timeline.overflow_tile_prompts?.some(prompt => prompt.trim())
      ? "Some tile prompts are outside the current clip and were retained."
      : null;
    return changed;
  }

  _commit() {
    const value = JSON.stringify({
      version: 2,
      tile_prompts: this.timeline.tile_prompts,
      keyframes: this.timeline.keyframes,
      overflow_tile_prompts: this.timeline.overflow_tile_prompts || [],
      video_segments: this.timeline.video_segments || [],
      ic_segments: this.timeline.ic_segments || [],
      audio_segments: this.timeline.audio_segments || [],
      use_custom_audio: Boolean(this.timeline.use_custom_audio),
      inpaint_audio: this.timeline.inpaint_audio !== false,
      use_ic_video_audio: Boolean(this.timeline.use_ic_video_audio),
      retake_mode: Boolean(this.timeline.retake_mode),
      retake: this.timeline.retake || null,
      ic_settings: normalizeIcSettings(this.timeline.ic_settings),
    });
    const widget = this.node.widgets?.find(item => item.name === "timeline_data");
    if (widget) widget.value = value;
    this.node.properties = this.node.properties || {};
    this.node.properties.timeline_data = value;
    this.node.properties.looping_director_timeline = value;
    this.node.properties.looping_director_schema_version = DIRECTOR_SCHEMA_VERSION;
    this.node.properties.looping_director_settings = Object.fromEntries(
      ["global_prompt", "frame_rate", "total_duration", "tile_duration", "overlap_duration", "target_height", "reference_keyframe_index"]
        .map(name => [name, serializedSetting(this.node, name)]),
    );
    this.node.properties.has_serialized_properties = true;
    this.node.setDirtyCanvas?.(true, true);
  }

  _deleteKeyframe(index) {
    const referenceWidget = this.node.widgets?.find(item => item.name === "reference_keyframe_index");
    const referenceIndex = Math.max(0, Math.round(Number(referenceWidget?.value) || 0));
    this.timeline.keyframes.splice(index, 1);
    if (referenceWidget) {
      let nextReference = referenceIndex;
      if (this.timeline.keyframes.length === 0) {
        nextReference = 0;
      } else if (index < referenceIndex) {
        nextReference -= 1;
      }
      nextReference = Math.min(nextReference, this.timeline.keyframes.length - 1);
      referenceWidget.value = nextReference;
    }
    this.selectedKeyframe = null;
  }

  _referenceIndex() {
    return Math.max(0, Math.round(numericWidget(this.node, "reference_keyframe_index", 0)));
  }

  _setReferenceIndex(index) {
    const widget = this.node.widgets?.find(item => item.name === "reference_keyframe_index");
    if (!widget) return;
    widget.value = Math.max(0, Math.min(index, Math.max(0, this.timeline.keyframes.length - 1)));
  }

  _syncAutoKeyframes() {
    const referenceFrames = loopingSchedule(this.node).referenceFrames;
    let changed = false;
    for (const keyframe of this.timeline.keyframes) {
      if (!keyframe.autoPosition || !Number.isInteger(Number(keyframe.defaultSlot))) continue;
      const slot = Number(keyframe.defaultSlot);
      const frame = referenceFrames[slot];
      if (frame === undefined || Number(keyframe.frame) === frame) continue;
      keyframe.frame = frame;
      changed = true;
    }
    return changed;
  }

  _mediaSegments(kind) {
    const key = kind === "ic" ? "ic_segments" : `${kind}_segments`;
    if (!Array.isArray(this.timeline[key])) this.timeline[key] = [];
    return this.timeline[key];
  }

  _mediaLabel(kind) {
    return kind === "ic" ? "IC Video" : kind === "video" ? "Video" : "Audio";
  }

  // Files up to CHUNK_SIZE go through ComfyUI's /upload/image; larger ones are split
  // across the Director's own chunk endpoint, which bypasses the 413 body limit.
  async _uploadFile(file) {
    const CHUNK_SIZE = 50 * 1024 * 1024;
    const safeFileName = file.name.replace(/[^a-zA-Z0-9.\-_]/g, "_");

    try {
      const check = await api.fetchApi(
        `/ltx_director_check_file?filename=${encodeURIComponent(safeFileName)}&size=${file.size}`,
      );
      if (check.ok) {
        const result = await check.json();
        if (result.exists && result.name) return { name: result.name, audioFile: null };
      }
    } catch (error) {
      console.warn("[LTXLoopingDirector] Could not check for an existing upload", error);
    }

    if (file.size > CHUNK_SIZE) {
      const safeName = `${Date.now()}_${safeFileName}`;
      const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
      let last = null;
      for (let index = 0; index < totalChunks; index += 1) {
        const body = new FormData();
        body.append("file", file.slice(index * CHUNK_SIZE, (index + 1) * CHUNK_SIZE));
        body.append("filename", safeName);
        body.append("chunk_index", String(index));
        body.append("total_chunks", String(totalChunks));
        const response = await api.fetchApi("/ltx_director_upload_chunk", { method: "POST", body });
        if (!response.ok) {
          console.warn(`[LTXLoopingDirector] Chunk ${index + 1}/${totalChunks} of ${file.name} failed`);
          return null;
        }
        last = await response.json();
      }
      if (!last || !last.name) return null;
      // The final chunk already reports any extracted audio track.
      return { name: last.name, audioFile: last.audio_file || null };
    }

    const body = new FormData();
    body.append("image", file);
    body.append("subfolder", "whatdreamscost");
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response.ok) {
      console.warn(`[LTXLoopingDirector] Could not upload ${file.name}`);
      return null;
    }
    const data = await response.json();
    const filename = data.name || data.filename;
    if (!filename) return null;
    return { name: data.subfolder ? `${data.subfolder}/${filename}` : filename, audioFile: null };
  }

  async _probeAudioTrack(filename) {
    try {
      const response = await api.fetchApi(
        `/ltx_director_get_audio?filename=${encodeURIComponent(filename)}`,
      );
      if (!response.ok) return null;
      const data = await response.json();
      return data.audio_file || null;
    } catch (error) {
      console.warn("[LTXLoopingDirector] Could not probe the video audio track", error);
      return null;
    }
  }

  async _uploadMediaFiles(files, kind) {
    const schedule = loopingSchedule(this.node);
    const selectedTile = Math.max(0, schedule.chunks.findIndex(chunk => this.selectedFrame >= chunk.startFrame && this.selectedFrame < chunk.endFrame));
    if (kind === "retake") {
      const file = files.find(item => item.type.startsWith("video/"));
      if (!file) return;
      const uploaded = await this._uploadFile(file);
      if (!uploaded) return;
      this.timeline.retake_mode = true;
      this.timeline.retake = {
        imageFile: uploaded.name,
        tiles: [selectedTile],
        tile: selectedTile,
        trimStart: 0,
        strength: 1.0,
      };
      this._commit();
      this.refresh();
      return;
    }
    for (const file of files) {
      const isAudio = kind === "audio";
      if (isAudio && !(file.type.startsWith("audio/") || file.type.startsWith("video/"))) continue;
      if (kind === "video" && !file.type.startsWith("video/")) continue;
      if (kind === "ic" && !(file.type.startsWith("image/") || file.type.startsWith("video/"))) continue;
      const uploaded = await this._uploadFile(file);
      if (!uploaded) continue;
      const imageFile = uploaded.name;
      const segment = {
        id: `${kind}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        imageFile,
        tile: selectedTile,
        trimStart: 0,
        strength: 1.0,
      };
      if (kind === "ic") {
        segment.mediaType = file.type.startsWith("image/") ? "image" : "video";
        segment.attentionStrength = 0.65;
        segment.resampleMode = "nearest";
      }
      if (kind === "audio") {
        segment.audioFile = imageFile;
        segment.linkedVideoId = null;
      }
      this._mediaSegments(kind).push(segment);
      if (kind === "video") {
        // A standard Video that carries sound gets a linked Audio item, so its own
        // track is mixed instead of silently dropped.
        const track = uploaded.audioFile || await this._probeAudioTrack(imageFile);
        if (track) {
          this._mediaSegments("audio").push({
            id: `audio-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
            linkedVideoId: segment.id,
            tile: selectedTile,
            trimStart: 0,
            strength: 1.0,
          });
          this.timeline.use_custom_audio = true;
        }
      }
    }
    if (kind === "audio") this.timeline.use_custom_audio = true;
    this._commit();
    this.refresh();
  }

  _removeMedia(kind, index) {
    const [removed] = this._mediaSegments(kind).splice(index, 1);
    if (kind === "video" && removed && removed.id != null) {
      // A linked Audio item has no file of its own, so it cannot outlive its video.
      this.timeline.audio_segments = this._mediaSegments("audio").filter(
        segment => String(segment.linkedVideoId) !== String(removed.id),
      );
    }
    this._commit();
    this.refresh();
  }

  _mediaName(kind, segment) {
    if (kind === "audio" && !segment.audioFile && segment.linkedVideoId != null) {
      const linked = this._mediaSegments("video").find(
        item => String(item.id) === String(segment.linkedVideoId),
      );
      const source = String(linked?.imageFile || "video").split("/").pop();
      return `${source} (linked audio)`;
    }
    return String(segment.imageFile || segment.audioFile || "media").split("/").pop();
  }

  _renderMedia() {
    this.media.innerHTML = "";
    for (const kind of ["video", "ic", "audio"]) {
      const lane = document.createElement("div");
      lane.className = "ld-media-lane";
      const head = document.createElement("div");
      head.className = "ld-media-head";
      head.appendChild(document.createTextNode(
        `${this._mediaLabel(kind)} — assign each item to a tile`,
      ));
      const addButton = makeButton("Add", `Add ${this._mediaLabel(kind)}`, () => this.mediaInputs[kind].click());
      addButton.disabled = Boolean(this.timeline.retake_mode);
      head.appendChild(addButton);
      if (kind === "ic") {
        const audioToggle = document.createElement("label");
        audioToggle.className = "ld-status";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = Boolean(this.timeline.use_ic_video_audio);
        checkbox.addEventListener("change", event => {
          this.timeline.use_ic_video_audio = event.target.checked;
          this._commit();
          this.refresh();
        });
        audioToggle.appendChild(checkbox);
        audioToggle.appendChild(document.createTextNode(" use IC Video audio"));
        head.appendChild(audioToggle);
      }
      lane.appendChild(head);
      const items = document.createElement("div");
      items.className = "ld-media-items";
      this._mediaSegments(kind).forEach((segment, index) => {
        const row = document.createElement("div");
        row.className = "ld-media-item";
        const name = document.createElement("span");
        name.textContent = this._mediaName(kind, segment);
        name.title = String(segment.imageFile || segment.audioFile || this._mediaName(kind, segment));
        row.appendChild(name);
        const tileInput = document.createElement("input");
        tileInput.type = "number";
        tileInput.min = "0";
        tileInput.max = String(Math.max(0, loopingSchedule(this.node).chunks.length - 1));
        tileInput.step = "1";
        tileInput.value = String(Number(segment.tile) || 0);
        tileInput.title = `${this._mediaLabel(kind)} tile index`;
        tileInput.addEventListener("change", event => {
          segment.tile = Math.max(0, Math.min(loopingSchedule(this.node).chunks.length - 1, Math.round(Number(event.target.value) || 0)));
          delete segment.start;
          delete segment.length;
          this._commit();
          this.refresh();
        });
        row.appendChild(tileInput);
        const trimInput = document.createElement("input");
        trimInput.type = "number";
        trimInput.min = "0";
        trimInput.step = "1";
        trimInput.value = String(Math.max(0, Math.round(Number(segment.trimStart) || 0)));
        trimInput.title = "Source trim start (frames)";
        trimInput.addEventListener("change", event => {
          segment.trimStart = Math.max(0, Math.round(Number(event.target.value) || 0));
          this._commit();
          this.refresh();
        });
        row.appendChild(trimInput);
        const strengthInput = document.createElement("input");
        strengthInput.type = "number";
        strengthInput.min = "0";
        strengthInput.max = "1";
        strengthInput.step = "0.01";
        strengthInput.value = String(Math.max(0, Math.min(1, Number(segment.strength ?? 1))));
        strengthInput.title = "Guide strength";
        strengthInput.addEventListener("change", event => {
          segment.strength = Math.max(0, Math.min(1, Number(event.target.value) || 0));
          this._commit();
          this.refresh();
        });
        row.appendChild(strengthInput);
        if (kind === "ic") {
          const attentionInput = document.createElement("input");
          attentionInput.type = "number";
          attentionInput.min = "0";
          attentionInput.max = "1";
          attentionInput.step = "0.01";
          attentionInput.value = String(Math.max(0, Math.min(1, Number(segment.attentionStrength ?? 1))));
          attentionInput.title = "IC attention strength";
          attentionInput.addEventListener("change", event => {
            segment.attentionStrength = Math.max(0, Math.min(1, Number(event.target.value) || 0));
            this._commit();
            this.refresh();
          });
          row.appendChild(attentionInput);
        } else {
          row.appendChild(document.createElement("span"));
        }
        row.appendChild(makeButton("×", `Remove ${this._mediaLabel(kind)}`, () => this._removeMedia(kind, index)));
        items.appendChild(row);
      });
      lane.appendChild(items);
      if (this.timeline.retake_mode) {
        lane.style.opacity = "0.45";
        lane.style.pointerEvents = "none";
      }
      lane.addEventListener("dragover", event => {
        if (Array.from(event.dataTransfer?.items || []).some(item => item.kind === "file")) {
          event.preventDefault();
          lane.classList.add("dragover");
        }
      });
      lane.addEventListener("dragleave", () => lane.classList.remove("dragover"));
      lane.addEventListener("drop", async event => {
        event.preventDefault();
        lane.classList.remove("dragover");
        await this._uploadMediaFiles(Array.from(event.dataTransfer?.files || []), kind);
      });
      this.media.appendChild(lane);
    }
    if (this.timeline.retake_mode && this.timeline.retake && typeof this.timeline.retake === "object") {
      const retake = this.timeline.retake;
      const lane = document.createElement("div");
      lane.className = "ld-media-lane";
      const head = document.createElement("div");
      head.className = "ld-media-head";
      head.appendChild(document.createTextNode(
        "Retake mode (BETA) — normal keyframe and guide editing is disabled",
      ));
      lane.appendChild(head);
      const row = document.createElement("div");
      row.className = "ld-media-item";
      const name = document.createElement("span");
      name.textContent = String(retake.imageFile || retake.fileName || "retake").split("/").pop();
      row.appendChild(name);

      // Retake selects a contiguous run of tiles; the earlier tile still owns the
      // overlap, so adjacent selections regenerate one continuous region.
      const lastTile = Math.max(0, loopingSchedule(this.node).chunks.length - 1);
      const selected = retakeTileList(retake, lastTile + 1);
      for (const [label, title, current, apply] of [
        ["first", "First retake tile", selected[0], (value) => {
          const end = Math.max(value, selected[selected.length - 1]);
          setRetakeTiles(retake, value, end, lastTile);
        }],
        ["last", "Last retake tile (inclusive)", selected[selected.length - 1], (value) => {
          const start = Math.min(value, selected[0]);
          setRetakeTiles(retake, start, value, lastTile);
        }],
      ]) {
        const input = document.createElement("input");
        input.type = "number";
        input.min = "0";
        input.max = String(lastTile);
        input.step = "1";
        input.value = String(current);
        input.title = title;
        input.setAttribute("aria-label", `${title} (${label})`);
        input.addEventListener("change", event => {
          apply(Math.max(0, Math.min(lastTile, Math.round(Number(event.target.value) || 0))));
          this._commit();
          this.refresh();
        });
        row.appendChild(input);
      }
      for (const [key, min, max, step, title] of [
        ["trimStart", 0, 1000000, 1, "Source trim start (frames)"],
        ["strength", 0, 1, 0.01, "Retake strength"],
      ]) {
        const input = document.createElement("input");
        input.type = "number";
        input.min = String(min);
        input.max = String(max);
        input.step = String(step);
        input.value = String(Math.max(min, Math.min(max, Number(retake[key] ?? (key === "strength" ? 1 : 0)))));
        input.title = title;
        input.addEventListener("change", event => {
          retake[key] = Math.max(min, Math.min(max, Number(event.target.value) || 0));
          this._commit();
          this.refresh();
        });
        row.appendChild(input);
      }
      row.appendChild(makeButton("×", "Clear Retake", () => {
        this.timeline.retake_mode = false;
        this.timeline.retake = null;
        this._commit();
        this.refresh();
      }));
      lane.appendChild(row);
      this.media.appendChild(lane);
    }
    this._renderGuideEncoding();
  }

  _renderGuideEncoding() {
    const settings = normalizeIcSettings(this.timeline.ic_settings);
    this.timeline.ic_settings = settings;
    const lane = document.createElement("div");
    lane.className = "ld-media-lane";
    const head = document.createElement("div");
    head.className = "ld-media-head";
    head.appendChild(document.createTextNode("Guide encoding — Video, IC, and Retake"));
    lane.appendChild(head);
    const row = document.createElement("div");
    row.className = "ld-media-item";

    const update = (key, value) => {
      settings[key] = value;
      this.timeline.ic_settings = normalizeIcSettings(settings);
      this._commit();
      this.refresh();
    };

    for (const [key, options, title] of [
      ["crop", ["center", "disabled"], "Resize crop: center crops, disabled stretches to fit"],
      ["upscale_method", IC_UPSCALE_METHODS, "Resampling used to fit guides to the pass resolution"],
    ]) {
      const select = document.createElement("select");
      select.title = title;
      for (const option of options) {
        const item = document.createElement("option");
        item.value = option;
        item.textContent = `${key === "crop" ? "crop" : ""}${option}`;
        item.selected = settings[key] === option;
        select.appendChild(item);
      }
      select.addEventListener("change", event => update(key, event.target.value));
      row.appendChild(select);
    }

    const tiled = document.createElement("label");
    tiled.className = "ld-status";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = settings.use_tiled_encode;
    checkbox.title = "VAE-encode guides in tiles to cut peak VRAM";
    checkbox.addEventListener("change", event => update("use_tiled_encode", event.target.checked));
    tiled.appendChild(checkbox);
    tiled.appendChild(document.createTextNode(" tiled encode"));
    row.appendChild(tiled);

    for (const [key, min, max, step, title] of [
      ["tile_size", 64, 512, 32, "Tiled encode tile size"],
      ["tile_overlap", 16, 256, 16, "Tiled encode overlap"],
    ]) {
      const input = document.createElement("input");
      input.type = "number";
      input.min = String(min);
      input.max = String(max);
      input.step = String(step);
      input.value = String(settings[key]);
      input.title = title;
      input.disabled = !settings.use_tiled_encode;
      input.addEventListener("change", event => update(key, Number(event.target.value)));
      row.appendChild(input);
    }
    lane.appendChild(row);
    this.media.appendChild(lane);
  }

  _renderToolbar() {
    this.toolbar.innerHTML = "";
    const retakeActive = Boolean(this.timeline.retake_mode);
    const addImages = makeButton("Add Image", "Upload one or more image keyframes", () => this.fileInput.click(), ICONS.upload);
    this.toolbar.appendChild(addImages);
    const addVideo = makeButton("Add Video", "Add a standard video guide to the selected tile", () => this.mediaInputs.video.click(), ICONS.video);
    const addICVideo = makeButton("Add IC Video", "Add an IC-LoRA video or image guide to the selected tile", () => this.mediaInputs.ic.click(), ICONS.motion);
    this.toolbar.appendChild(addVideo);
    this.toolbar.appendChild(addICVideo);
    const addAudio = makeButton("Add Audio", "Add a custom audio segment", () => this.mediaInputs.audio.click(), ICONS.audio);
    this.toolbar.appendChild(addAudio);
    // Retake skips the normal guide path, so its editing entry points are inert.
    for (const button of [addImages, addVideo, addICVideo, addAudio]) {
      button.disabled = retakeActive;
      if (retakeActive) button.title = "Disabled while Retake mode is active";
    }
    const deleteButton = makeIconButton("Delete", "Delete the selected keyframe", () => {
      if (this.selectedKeyframe === null) return;
      this._deleteKeyframe(this.selectedKeyframe);
      this._commit();
      this.refresh();
    }, ICONS.trash, true);
    deleteButton.disabled = retakeActive || this.selectedKeyframe === null;
    this.toolbar.appendChild(deleteButton);

    const retakeButton = makeButton(
      "Retake Mode (BETA)",
      this.timeline.retake_mode ? "Remove the retake guide" : "Add a retake video to the selected tile",
      () => {
        if (this.timeline.retake_mode) {
          this.timeline.retake_mode = false;
          this.timeline.retake = null;
          this._commit();
          this.refresh();
        } else {
          this.mediaInputs.retake.click();
        }
      },
      ICONS.retake,
    );
    if (retakeActive) retakeButton.classList.add("toggle-on");
    this.toolbar.appendChild(retakeButton);

    const units = document.createElement("div");
    units.className = "pr-segmented-control";
    for (const mode of ["frames", "seconds"]) {
      // A div, like the Director's own segmented controls: .pr-segment carries no
      // button reset, so a <button> would keep its default chrome.
      const segment = document.createElement("div");
      segment.className = `pr-segment${this.displayMode === mode ? " active" : ""}`;
      segment.textContent = mode === "frames" ? "Frames" : "Seconds";
      segment.title = `Display keyframe labels as ${mode === "frames" ? "frame numbers" : "seconds"}`;
      segment.addEventListener("click", () => {
        this.displayMode = mode;
        this.node._loopingDirectorDisplayMode = mode;
        this.node.properties = this.node.properties || {};
        this.node.properties.looping_director_display_mode = mode;
        this.refresh();
      });
      units.appendChild(segment);
    }
    this.toolbar.appendChild(units);

    const referenceLabel = document.createElement("label");
    referenceLabel.className = "ld-status";
    referenceLabel.textContent = "External reference";
    const referenceSelect = document.createElement("select");
    referenceSelect.className = "pr-settings-select";
    this.timeline.keyframes.forEach((keyframe, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      const filename = String(keyframe.imageFile || "").split("/").pop() || "image";
      option.textContent = `K${index} · ${frameLabel(Number(keyframe.frame) || 0, this.node)} · ${filename}`;
      option.selected = index === this._referenceIndex();
      referenceSelect.appendChild(option);
    });
    if (!this.timeline.keyframes.length) {
      const option = document.createElement("option");
      option.textContent = "No keyframes";
      option.selected = true;
      referenceSelect.appendChild(option);
      referenceSelect.disabled = true;
    }
    referenceSelect.addEventListener("change", event => {
      this._setReferenceIndex(Number(event.target.value));
      this._commit();
      this.refresh();
    });
    this.toolbar.appendChild(referenceLabel);
    this.toolbar.appendChild(referenceSelect);
    if (this.node._loopingDirectorWarning) {
      const warning = document.createElement("span");
      warning.className = "ld-status ld-warning";
      warning.textContent = this.node._loopingDirectorWarning;
      this.toolbar.appendChild(warning);
    }
    this._renderReadout();
  }

  _renderReadout() {
    const schedule = loopingSchedule(this.node);
    const tile = Math.max(0, schedule.chunks.findIndex(
      chunk => this.selectedFrame >= chunk.startFrame && this.selectedFrame < chunk.endFrame,
    ));
    const chunk = schedule.chunks[tile];
    if (this.boundsDisplay) {
      this.boundsDisplay.textContent = chunk
        ? `Tile ${tile} · ${frameLabel(chunk.startFrame, this.node)}–${frameLabel(Math.max(chunk.startFrame, chunk.endFrame - 1), this.node)}`
        : "";
    }
    if (this.timecodeDisplay) {
      this.timecodeDisplay.textContent = frameLabel(this.selectedFrame, this.node);
    }
  }

  _renderStrip() {
    const schedule = loopingSchedule(this.node);
    const totalFrames = schedule.frameCount;
    const chunks = schedule.chunks;
    this.strip.innerHTML = "";
    for (let index = 0; index < chunks.length; index += 1) {
      const chunk = chunks[index];
      const band = document.createElement("div");
      band.className = "ld-tile-band";
      const left = chunk.startFrame / Math.max(1, totalFrames - 1) * 100;
      const right = chunk.endFrame / Math.max(1, totalFrames - 1) * 100;
      band.style.left = `${Math.max(0, Math.min(100, left))}%`;
      band.style.width = `${Math.max(0, Math.min(100, right) - Math.max(0, left))}%`;
      band.title = `Tile ${index}: ${frameLabel(chunk.startFrame, this.node)}–${frameLabel(Math.max(chunk.startFrame, chunk.endFrame - 1), this.node)}`;
      this.strip.appendChild(band);
    }

    const referenceIndex = this._referenceIndex();
    // Retake replaces the normal guide path entirely, so keyframes stay visible but
    // become read-only rather than silently having no effect.
    const retakeActive = Boolean(this.timeline.retake_mode);
    this.strip.classList.toggle("retake-locked", retakeActive);
    this.timeline.keyframes.forEach((keyframe, index) => {
      const frame = Number(keyframe.frame);
      const marker = document.createElement("div");
      const valid = Number.isInteger(frame) && frame >= 0 && frame < totalFrames && frame % TIME_SCALE === 0;
      marker.className = `ld-frame-marker${index === referenceIndex ? " reference" : ""}${keyframe.autoPosition ? " auto" : ""}${index === this.selectedKeyframe ? " selected" : ""}${valid ? "" : " invalid"}`;
      const markerPosition = Math.max(0, Math.min(100, frame / Math.max(1, totalFrames - 1) * 100));
      marker.style.left = `${markerPosition}%`;
      marker.style.transform = markerPosition <= 4 ? "translateX(2px)" : markerPosition >= 96 ? "translateX(-100%)" : "translateX(-50%)";
      marker.title = valid
        ? `Keyframe K${index} at ${frameLabel(frame, this.node)}${index === referenceIndex ? " (external reference)" : ""}`
        : `Keyframe K${index}: invalid position`;
      const image = document.createElement("img");
      if (keyframe.imageFile) {
        const parts = String(keyframe.imageFile).split("/");
        const filename = parts.pop() || "";
        const subfolder = parts.join("/");
        image.src = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=${encodeURIComponent(subfolder)}`);
      }
      marker.appendChild(image);
      const label = document.createElement("span");
      label.textContent = Number.isFinite(frame) ? `K${index} · ${frameLabel(frame, this.node)}` : `K${index} · invalid`;
      marker.appendChild(label);
      if (!retakeActive) {
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "ld-delete";
        remove.textContent = "×";
        remove.addEventListener("click", event => {
          event.stopPropagation();
          this._deleteKeyframe(index);
          this._commit();
          this.refresh();
        });
        marker.appendChild(remove);
      }

      marker.addEventListener("pointerdown", event => {
        event.stopPropagation();
        this.selectedKeyframe = index;
        if (valid) this.selectedFrame = frame;
        this.strip.querySelectorAll(".ld-frame-marker").forEach(item => {
          item.classList.toggle("selected", item === marker);
        });
        if (retakeActive) return;
        marker.setPointerCapture?.(event.pointerId);
        const move = moveEvent => {
          const rect = this.strip.getBoundingClientRect();
          const ratio = Math.max(0, Math.min(1, (moveEvent.clientX - rect.left) / Math.max(1, rect.width)));
          keyframe.frame = snapFrame(ratio * (totalFrames - 1), totalFrames);
          keyframe.autoPosition = false;
          marker.style.left = `${Math.max(0, Math.min(100, keyframe.frame / Math.max(1, totalFrames - 1) * 100))}%`;
          const position = keyframe.frame / Math.max(1, totalFrames - 1) * 100;
          marker.style.transform = position <= 4 ? "translateX(2px)" : position >= 96 ? "translateX(-100%)" : "translateX(-50%)";
          label.textContent = `K${index} · ${frameLabel(keyframe.frame, this.node)}`;
          this._commit();
          this._renderToolbar();
        };
        const up = () => {
          marker.removeEventListener("pointermove", move);
          marker.removeEventListener("pointerup", up);
          marker.removeEventListener("pointercancel", up);
        };
        marker.addEventListener("pointermove", move);
        marker.addEventListener("pointerup", up);
        marker.addEventListener("pointercancel", up);
        this._renderToolbar();
      });
      this.strip.appendChild(marker);
    });

    this._renderEmptySlots(schedule, totalFrames, retakeActive);
  }

  // One slot per tile: frame 0, then the middle of each tile overlap. Slots are derived
  // from the current schedule, so changing the duration re-lays them out. They are
  // presentation only — an empty slot is never written to the timeline, so no keyframe
  // is used at that index until an image is dropped on it.
  _renderEmptySlots(schedule, totalFrames, retakeActive) {
    if (retakeActive) return;
    const occupied = new Set();
    for (const keyframe of this.timeline.keyframes) {
      occupied.add(Number(keyframe.frame));
      const slot = Number(keyframe.defaultSlot);
      if (Number.isInteger(slot)) occupied.add(schedule.referenceFrames[slot]);
    }
    schedule.referenceFrames.forEach((frame, slot) => {
      if (occupied.has(frame)) return;
      const marker = document.createElement("div");
      marker.className = "ld-frame-marker empty";
      const position = Math.max(0, Math.min(100, frame / Math.max(1, totalFrames - 1) * 100));
      marker.style.left = `${position}%`;
      marker.style.transform = position <= 4 ? "translateX(2px)" : position >= 96 ? "translateX(-100%)" : "translateX(-50%)";
      marker.title = `Empty slot at ${frameLabel(frame, this.node)} — click or drop an image to use a keyframe here`;

      const drop = document.createElement("div");
      drop.className = "ld-slot-drop";
      drop.innerHTML = ICONS.plus;
      marker.appendChild(drop);
      const label = document.createElement("span");
      label.textContent = `${slot === 0 ? "start" : `tile ${slot}`} · ${frameLabel(frame, this.node)}`;
      marker.appendChild(label);

      marker.addEventListener("click", event => {
        event.stopPropagation();
        this.selectedFrame = frame;
        this._pendingSlot = slot;
        this.fileInput.click();
      });
      marker.addEventListener("dragover", event => {
        if (Array.from(event.dataTransfer?.items || []).some(item => item.kind === "file")) {
          event.preventDefault();
          event.stopPropagation();
          marker.classList.add("dragover");
        }
      });
      marker.addEventListener("dragleave", () => marker.classList.remove("dragover"));
      marker.addEventListener("drop", async event => {
        event.preventDefault();
        event.stopPropagation();
        marker.classList.remove("dragover");
        this._pendingSlot = slot;
        await this._uploadFiles(Array.from(event.dataTransfer?.files || []));
      });
      this.strip.appendChild(marker);
    });
  }

  _renderPrompts() {
    const chunks = loopingSchedule(this.node).chunks;
    const count = chunks.length;
    this._ensurePromptCount(count);
    this.prompts.innerHTML = "";
    for (let index = 0; index < count; index += 1) {
      const chunk = chunks[index];
      const cell = document.createElement("div");
      cell.className = "ld-prompt-cell";
      const wrapper = document.createElement("div");
      wrapper.className = "pr-prompt-wrapper";
      const label = document.createElement("div");
      label.className = "pr-prompt-label";
      label.textContent = `Tile ${index} · ${frameLabel(chunk.startFrame, this.node)}–${frameLabel(Math.max(chunk.startFrame, chunk.endFrame - 1), this.node)}`;
      wrapper.appendChild(label);
      const textarea = document.createElement("textarea");
      textarea.className = "pr-prompt-area";
      textarea.placeholder = DEFAULT_TILE_PROMPT;
      textarea.value = this.timeline.tile_prompts[index] || "";
      textarea.addEventListener("input", event => {
        this.timeline.tile_prompts[index] = event.target.value;
        this._commit();
      });
      // Matches the Director: the focused prompt stays lit and its siblings dim.
      textarea.addEventListener("focus", () => {
        wrapper.classList.add("focus-active");
        this.container.classList.add("has-focus");
        this.selectedFrame = snapFrame(chunk.startFrame, frameCount(this.node));
        this._renderReadout();
      });
      textarea.addEventListener("blur", () => {
        wrapper.classList.remove("focus-active");
        this.container.classList.remove("has-focus");
      });
      wrapper.appendChild(textarea);
      cell.appendChild(wrapper);
      this.prompts.appendChild(cell);
    }
  }

  _normalizeGuideSegments() {
    const schedule = loopingSchedule(this.node);
    let changed = false;
    for (const segment of [...this.timeline.video_segments, ...this.timeline.ic_segments, ...this.timeline.audio_segments]) {
      if (!Number.isInteger(Number(segment.tile))) {
        const start = Number(segment.start);
        const tile = schedule.chunks.findIndex(chunk => Number.isFinite(start) && start >= chunk.startFrame && start < chunk.endFrame);
        segment.tile = tile < 0 ? 0 : tile;
        changed = true;
      }
      const tile = Math.max(0, Math.min(schedule.chunks.length - 1, Math.round(Number(segment.tile) || 0)));
      if (Number(segment.tile) !== tile) {
        segment.tile = tile;
        changed = true;
      }
      if (Object.prototype.hasOwnProperty.call(segment, "start") || Object.prototype.hasOwnProperty.call(segment, "length")) {
        delete segment.start;
        delete segment.length;
        changed = true;
      }
    }
    if (this.timeline.retake_mode && this.timeline.retake && typeof this.timeline.retake === "object") {
      const retake = this.timeline.retake;
      if (!Number.isInteger(Number(retake.tile))) {
        const start = Number(retake.start ?? retake.retakeStart ?? 0);
        const tile = schedule.chunks.findIndex(chunk => Number.isFinite(start) && start >= chunk.startFrame && start < chunk.endFrame);
        retake.tile = tile < 0 ? 0 : tile;
        changed = true;
      }
      const tile = Math.max(0, Math.min(schedule.chunks.length - 1, Math.round(Number(retake.tile) || 0)));
      if (Number(retake.tile) !== tile) {
        retake.tile = tile;
        changed = true;
      }
    }
    return changed;
  }

  async _uploadFiles(files) {
    const schedule = loopingSchedule(this.node);
    const totalFrames = schedule.frameCount;
    // A slot the user clicked or dropped onto takes the first image; the rest fall
    // through to the normal next-free-slot search.
    let pendingSlot = Number.isInteger(this._pendingSlot) ? this._pendingSlot : null;
    this._pendingSlot = null;
    const usedSlots = new Set(
      this.timeline.keyframes
        .map(keyframe => Number(keyframe.defaultSlot))
        .filter(Number.isInteger),
    );
    schedule.referenceFrames.forEach((frame, slot) => {
      if (this.timeline.keyframes.some(keyframe => Number(keyframe.frame) === frame)) usedSlots.add(slot);
    });
    for (let index = 0; index < files.length; index += 1) {
      const file = files[index];
      if (!file.type.startsWith("image/")) continue;
      const body = new FormData();
      body.append("image", file);
      body.append("subfolder", "whatdreamscost");
      const response = await api.fetchApi("/upload/image", { method: "POST", body });
      if (!response.ok) {
        console.warn(`[LTXLoopingDirector] Could not upload ${file.name}`);
        continue;
      }
      const data = await response.json();
      const filename = data.name || data.filename;
      const subfolder = data.subfolder || "";
      if (!filename) continue;
      const imageFile = subfolder ? `${subfolder}/${filename}` : filename;
      let slot = pendingSlot !== null && !usedSlots.has(pendingSlot)
        ? pendingSlot
        : schedule.referenceFrames.findIndex((_, candidate) => !usedSlots.has(candidate));
      pendingSlot = null;
      if (slot < 0) slot = schedule.referenceFrames.length;
      const frame = schedule.referenceFrames[slot] ?? snapFrame(this.selectedFrame + index * TIME_SCALE, totalFrames);
      this.timeline.keyframes.push({
        frame,
        imageFile,
        autoPosition: slot < schedule.referenceFrames.length,
        defaultSlot: slot,
      });
      usedSlots.add(slot);
    }
    this._commit();
    this.refresh();
  }

  refresh(commit = false) {
    const moved = this._syncAutoKeyframes();
    const promptsChanged = this._ensurePromptCount(tileCount(this.node));
    const guideChanged = this._normalizeGuideSegments();
    this._renderToolbar();
    this._renderStrip();
    this._renderMedia();
    this._renderPrompts();
    if (commit || moved || promptsChanged || guideChanged) this._commit();
  }

  syncLayout() {
    if (this.container.parentElement) this.container.style.width = "100%";
  }
}

app.registerExtension({
  name: "LTXLoopingDirector",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "LTXLoopingDirector") return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      originalOnNodeCreated?.apply(this, arguments);
      const timelineWidget = this.widgets?.find(widget => widget.name === "timeline_data");
      hideWidget(timelineWidget);
      const referenceWidget = this.widgets?.find(widget => widget.name === "reference_keyframe_index");
      hideWidget(referenceWidget);
      this.properties = this.properties || {};
      this._loopingDirectorDisplayMode = this.properties.looping_director_display_mode || "seconds";

      const container = document.createElement("div");
      const domWidget = this.addDOMWidget("looping_director_ui", "looping_director_ui", container, {
        getValue: () => "",
        setValue: () => {},
      });
      domWidget.computeSize = width => [Math.max(10, width - 20), 470];
      this._loopingDirectorEditor = new LoopingDirectorEditor(this, container);
      this.size[0] = Math.max(this.size?.[0] || 0, 760);
      this.size[1] = Math.max(this.size?.[1] || 0, 620);
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = originalOnConfigure?.apply(this, arguments);
      const values = info?.widgets_values;
      const widgets = this.widgets || [];
      const byName = name => widgets.find(widget => widget.name === name);
      const finite = (value, fallback, minimum = -Infinity) => {
        const number = Number(value);
        return Number.isFinite(number) && number >= minimum ? number : fallback;
      };
      const applySettings = settings => {
        if (!settings || typeof settings !== "object") return;
        const normalized = {
          global_prompt: typeof settings.global_prompt === "string" ? settings.global_prompt : "",
          frame_rate: finite(settings.frame_rate, DEFAULT_FRAME_RATE, 1),
          total_duration: finite(settings.total_duration, DEFAULT_TOTAL_DURATION, 0.1),
          tile_duration: finite(settings.tile_duration, DEFAULT_TILE_DURATION, 0.1),
          overlap_duration: finite(settings.overlap_duration, DEFAULT_OVERLAP_DURATION, 0),
          target_height: finite(settings.target_height, DEFAULT_TARGET_HEIGHT, 32),
          reference_keyframe_index: Math.round(finite(settings.reference_keyframe_index, 0, 0)),
        };
        for (const [name, value] of Object.entries(normalized)) {
          const widget = byName(name);
          if (widget) widget.value = value;
        }
      };
      const storedSettings = info?.properties?.looping_director_settings;
      if (storedSettings) {
        applySettings(storedSettings);
      } else if (Array.isArray(values) && typeof values[0] === "string" && values[0].trim().startsWith("{")) {
        // Older generated workflows serialized the hidden required timeline before
        // the visible optional widgets. Restore by name before the editor reads it.
        applySettings({
          timeline_data: values[0],
          global_prompt: values[1],
          frame_rate: values[2],
          total_duration: values[3],
          tile_duration: values[4],
          overlap_duration: values[5],
          target_height: values[6],
          reference_keyframe_index: values[7],
        });
      } else if (
        Array.isArray(values)
        && typeof values[5] === "string"
        && (values[5].trim() === "" || values[5].trim().startsWith("{"))
      ) {
        applySettings({
          global_prompt: values[0],
          frame_rate: values[1],
          total_duration: values[2],
          tile_duration: values[3],
          overlap_duration: values[4],
          target_height: values[6],
          reference_keyframe_index: values[7],
        });
      } else if (Array.isArray(values) && !Number.isFinite(Number(values[0]))) {
        applySettings({ global_prompt: values[0] });
      } else if (Array.isArray(values)) {
        applySettings({});
        this._loopingDirectorWarning = "Older Looping Director settings were invalid; safe defaults were restored.";
      }
      const raw = info?.properties?.looping_director_timeline
        || info?.properties?.timeline_data
        || this.widgets?.find(widget => widget.name === "timeline_data")?.value;
      if (raw && this._loopingDirectorEditor) {
        this._loopingDirectorEditor.setTimeline(raw);
      }
      return result;
    };

    const originalOnSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function () {
      this._loopingDirectorEditor?._commit();
      return originalOnSerialize?.apply(this, arguments);
    };

    const originalOnResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function () {
      const result = originalOnResize?.apply(this, arguments);
      this._loopingDirectorEditor?.syncLayout();
      return result;
    };
  },
});
