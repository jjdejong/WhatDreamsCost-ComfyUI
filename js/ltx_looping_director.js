const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const TIME_SCALE = 8;
// Timeline geometry, matching ltx_director.js so both editors read the same.
const RULER_HEIGHT = 24;
const BLOCK_HEIGHT = 160;
const IC_TRACK_HEIGHT = 80;
const AUDIO_TRACK_HEIGHT = 80;
const SIDEBAR_WIDTH = 120;
const CANVAS_HEIGHT = RULER_HEIGHT + BLOCK_HEIGHT + IC_TRACK_HEIGHT + AUDIO_TRACK_HEIGHT;
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
  minus: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line></svg>`,
  fit: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="12" x2="21" y2="12"></line><polyline points="8 7 3 12 8 17"></polyline><polyline points="16 7 21 12 16 17"></polyline></svg>`,
  play: `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`,
  pause: `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>`,
  loop: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12A9 9 0 0 0 6 5.3L3 8"></path><polyline points="3 3 3 8 8 8"></polyline><path d="M3 12a9 9 0 0 0 15 6.7l3-2.7"></path><polyline points="21 21 21 16 16 16"></polyline></svg>`,
  speakerOn: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>`,
  speakerOff: `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="1" y1="1" x2="23" y2="23"></line></svg>`,
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
    this.zoom = 1;
    this.selection = null;
    this.thumbnails = new Map();
    this.container.className = "pr-wrapper ld-editor";
    this.container.innerHTML = `
      <style>
        /* Only looping-specific chrome lives here. The toolbar, controls group,
           prompt panels and readouts reuse the Director's own pr- classes, which
           ltx_director.js injects globally, so both editors are one visual system. */
        .ld-editor { font-size: 12px; color: #e0e0e0; }
        .ld-timeline-layout { display: flex; flex-direction: row; width: 100%; border: 1px solid #111; border-radius: 6px; overflow: hidden; }
        .ld-sidebar { width: ${SIDEBAR_WIDTH}px; flex-shrink: 0; display: flex; flex-direction: column; border-right: 1px solid #111; box-sizing: border-box; background: #1e1e1e; user-select: none; }
        .ld-ruler-spacer { height: ${RULER_HEIGHT}px; width: 100%; border-bottom: 1px solid #111; background: #1e1e1e; box-sizing: border-box; flex-shrink: 0; }
        .ld-track-label { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px; border-bottom: 1px solid #111; background: #1e1e1e; box-sizing: border-box; overflow: hidden; flex-shrink: 0; }
        .ld-track-head { display: flex; align-items: center; justify-content: center; gap: 6px; }
        .ld-track-head span { color: #ccc; font-size: 12px; font-weight: bold; line-height: 1; }
        .ld-track-eye { cursor: pointer; display: inline-flex; align-items: center; justify-content: center; width: 14px; height: 14px; }
        .ld-track-badge { font-size: 9px; font-weight: 600; padding: 2px 6px; border-radius: 4px; background: #222; border: 1px solid #111; color: #777; cursor: pointer; transition: all .2s ease; }
        .ld-track-badge.on { background: #1c222d; border-color: #283142; color: #e0e0e0; }
        .ld-viewport { flex-grow: 1; min-width: 0; overflow-x: auto; overflow-y: hidden; background: #2a2a2a; transition: background .2s ease; }
        .ld-viewport.dragover { background: #333; box-shadow: inset 0 0 0 2px #888; }
        .ld-canvas { display: block; background: #2a2a2a; cursor: pointer; outline: none; }
        .ld-status { color: #aaa; font-size: 11px; }
        .ld-warning { color: #e6b35a; }
        .ld-info-left { display: flex; align-items: center; gap: 12px; margin-right: auto; }
        .ld-settings-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; width: 100%; box-sizing: border-box; }
        .ld-settings-row label { display: flex; align-items: center; gap: 5px; white-space: nowrap; font-size: 11px; font-weight: 600; color: #888; }
        .ld-settings-row select, .ld-settings-row input[type=number] { font-size: 11px; color: #fff; background: #222; border: 1px solid #444; border-radius: 4px; padding: 3px 5px; }
        .ld-settings-row select { min-width: 92px; }
        .ld-settings-row input[type=number] { width: 58px; text-align: center; }
        .ld-tile-prop { min-height: 96px; }
        .ld-global-prop { min-height: 76px; }
      </style>
      <div class="pr-toolbar">
        <div class="pr-actions"></div>
        <div class="pr-right-group"></div>
      </div>
      <div class="ld-timeline-layout">
        <div class="ld-sidebar"></div>
        <div class="ld-viewport"><canvas class="ld-canvas" tabindex="0"></canvas></div>
      </div>
      <div class="pr-controls-group">
        <div class="pr-strength-row"></div>
        <div class="pr-player-controls"></div>
        <div class="ld-settings-row"></div>
      </div>
      <div class="pr-prop-container ld-tile-prop"></div>
      <div class="pr-prop-container ld-global-prop"></div>
    `;
    this.toolbar = this.container.querySelector(".pr-actions");
    this.toolbarRight = this.container.querySelector(".pr-right-group");
    this.sidebar = this.container.querySelector(".ld-sidebar");
    this.viewport = this.container.querySelector(".ld-viewport");
    this.canvas = this.container.querySelector(".ld-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.infoRow = this.container.querySelector(".pr-strength-row");
    this.playerRow = this.container.querySelector(".pr-player-controls");
    this.settingsRow = this.container.querySelector(".ld-settings-row");
    this.tileProp = this.container.querySelector(".ld-tile-prop");
    this.globalProp = this.container.querySelector(".ld-global-prop");

    this._buildSidebar();
    this._buildFileInputs();
    this._buildPromptPanels();
    this._bindCanvas();
  }

  _buildFileInputs() {
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
  }

  // Track gutter: same shape as the Director's sidebar — a ruler spacer, then one
  // label per track. Toggles are wired only where the Looping Director has a real
  // setting behind them.
  _buildSidebar() {
    this.sidebar.innerHTML = "";
    const spacer = document.createElement("div");
    spacer.className = "ld-ruler-spacer";
    this.sidebar.appendChild(spacer);

    const makeLabel = (text, height) => {
      const el = document.createElement("div");
      el.className = "ld-track-label";
      el.style.height = `${height}px`;
      const head = document.createElement("div");
      head.className = "ld-track-head";
      const span = document.createElement("span");
      span.textContent = text;
      head.appendChild(span);
      el.appendChild(head);
      el._head = head;
      this.sidebar.appendChild(el);
      return el;
    };

    const makeBadge = (label, isOn, onToggle) => {
      const badge = document.createElement("div");
      badge.className = `ld-track-badge${isOn ? " on" : ""}`;
      badge.textContent = label;
      badge.addEventListener("click", event => {
        event.stopPropagation();
        onToggle();
        this._commit();
        this.refresh();
      });
      return badge;
    };

    this.mainLabel = makeLabel("MAIN", BLOCK_HEIGHT);
    const tiles = tileCount(this.node);
    this.mainLabel.appendChild(makeBadge(`${tiles} tile${tiles === 1 ? "" : "s"}`, false, () => {}));

    this.icLabel = makeLabel("IC-LoRA Input", IC_TRACK_HEIGHT);
    this.icLabel.appendChild(makeBadge(
      `Audio: ${this.timeline.use_ic_video_audio ? "ON" : "OFF"}`,
      Boolean(this.timeline.use_ic_video_audio),
      () => { this.timeline.use_ic_video_audio = !this.timeline.use_ic_video_audio; },
    ));

    this.audioLabel = makeLabel("AUDIO", AUDIO_TRACK_HEIGHT);
    const speaker = document.createElement("div");
    speaker.className = "ld-track-eye";
    const audioOn = this.timeline.use_custom_audio !== false;
    speaker.style.color = audioOn ? "#aaa" : "#444";
    speaker.innerHTML = audioOn ? ICONS.speakerOn : ICONS.speakerOff;
    speaker.title = "Use the custom audio lane";
    speaker.addEventListener("click", event => {
      event.stopPropagation();
      this.timeline.use_custom_audio = !audioOn;
      this._commit();
      this.refresh();
    });
    this.audioLabel._head.appendChild(speaker);
    this.audioLabel.appendChild(makeBadge(
      `Inpaint: ${this.timeline.inpaint_audio !== false ? "ON" : "OFF"}`,
      this.timeline.inpaint_audio !== false,
      () => { this.timeline.inpaint_audio = this.timeline.inpaint_audio === false; },
    ));
  }

  _canvasWidth() {
    const available = Math.max(240, this.viewport.clientWidth || 640);
    return Math.round(available * this.zoom);
  }

  _frameToX(frame, width) {
    const total = Math.max(1, frameCount(this.node) - 1);
    return (Math.max(0, Math.min(total, frame)) / total) * width;
  }

  _xToFrame(x, width) {
    const total = Math.max(1, frameCount(this.node) - 1);
    return snapFrame((Math.max(0, Math.min(width, x)) / Math.max(1, width)) * total, frameCount(this.node));
  }

  _thumbnail(file) {
    if (!file) return null;
    const key = String(file);
    if (this.thumbnails.has(key)) return this.thumbnails.get(key);
    const parts = key.split("/");
    const filename = parts.pop() || "";
    const subfolder = parts.join("/");
    const image = new Image();
    image.onload = () => this._renderCanvas();
    image.onerror = () => this.thumbnails.set(key, null);
    image.src = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=${encodeURIComponent(subfolder)}`);
    this.thumbnails.set(key, image);
    return image;
  }

  _bindCanvas() {
    this.canvas.addEventListener("pointerdown", event => {
      const rect = this.canvas.getBoundingClientRect();
      this._onCanvasPointer(event.clientX - rect.left, event.clientY - rect.top, event);
    });
    this.canvas.addEventListener("dblclick", event => {
      const rect = this.canvas.getBoundingClientRect();
      const hit = this._hitTest(event.clientX - rect.left, event.clientY - rect.top);
      if (hit && hit.kind === "slot") {
        this._pendingSlot = hit.slot;
        this.fileInput.click();
      }
    });
    this.viewport.addEventListener("dragover", event => {
      if (Array.from(event.dataTransfer?.items || []).some(item => item.kind === "file")) {
        event.preventDefault();
        this.viewport.classList.add("dragover");
      }
    });
    this.viewport.addEventListener("dragleave", () => this.viewport.classList.remove("dragover"));
    this.viewport.addEventListener("drop", async event => {
      event.preventDefault();
      this.viewport.classList.remove("dragover");
      const rect = this.canvas.getBoundingClientRect();
      const hit = this._hitTest(event.clientX - rect.left, event.clientY - rect.top);
      const files = Array.from(event.dataTransfer?.files || []);
      if (hit && hit.kind === "slot") this._pendingSlot = hit.slot;
      const images = files.filter(file => file.type.startsWith("image/"));
      const videos = files.filter(file => file.type.startsWith("video/"));
      const audio = files.filter(file => file.type.startsWith("audio/"));
      // Drop target picks the lane, matching the Director's per-track drops.
      if (hit && hit.track === "ic" && (videos.length || images.length)) {
        await this._uploadMediaFiles([...videos, ...images], "ic");
      } else if (hit && hit.track === "audio" && (audio.length || videos.length)) {
        await this._uploadMediaFiles([...audio, ...videos], "audio");
      } else {
        if (images.length) await this._uploadFiles(images);
        if (videos.length) await this._uploadMediaFiles(videos, "video");
        if (audio.length) await this._uploadMediaFiles(audio, "audio");
      }
    });
  }

  _onCanvasPointer(x, y, event) {
    const width = this.canvas.width / (window.devicePixelRatio || 1);
    if (y <= RULER_HEIGHT) {
      this.selectedFrame = this._xToFrame(x, width);
      this._renderAll();
      return;
    }
    const hit = this._hitTest(x, y);
    this.selectedFrame = this._xToFrame(x, width);
    if (hit && hit.kind === "slot") {
      this.selection = null;
      this.selectedKeyframe = null;
      this.selectedTile = hit.tile;
      if (event?.detail === 1) {
        this._pendingSlot = hit.slot;
        this.fileInput.click();
      }
    } else if (hit && hit.kind === "keyframe") {
      this.selection = { kind: "keyframe", index: hit.index };
      this.selectedKeyframe = hit.index;
      this.selectedTile = hit.tile;
    } else if (hit && hit.kind === "media") {
      this.selection = { kind: hit.track, index: hit.index };
      this.selectedKeyframe = null;
      this.selectedTile = hit.tile;
    } else {
      this.selection = null;
      this.selectedKeyframe = null;
      if (hit) this.selectedTile = hit.tile;
    }
    this._renderAll();
  }

  _tileAtFrame(frame) {
    const chunks = loopingSchedule(this.node).chunks;
    const index = chunks.findIndex(chunk => frame >= chunk.startFrame && frame < chunk.endFrame);
    return index < 0 ? Math.max(0, chunks.length - 1) : index;
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

  // Geometry of everything drawn on the canvas, shared by the renderer and hit
  // testing so the two can never disagree.
  _layout() {
    const width = this._canvasWidth();
    const schedule = loopingSchedule(this.node);
    const retakeActive = Boolean(this.timeline.retake_mode);
    const items = [];

    const tileSpan = tile => {
      const chunk = schedule.chunks[tile];
      if (!chunk) return null;
      const startFrame = tile ? schedule.chunks[tile - 1].endFrame - schedule.overlap : chunk.startFrame;
      const owned = tile ? schedule.chunks[tile].startFrame + schedule.overlap : chunk.startFrame;
      return {
        x: this._frameToX(tile ? owned : chunk.startFrame, width),
        end: this._frameToX(chunk.endFrame, width),
        startFrame,
      };
    };

    schedule.chunks.forEach((chunk, tile) => {
      items.push({
        kind: "band",
        tile,
        x: this._frameToX(chunk.startFrame, width),
        w: this._frameToX(chunk.endFrame, width) - this._frameToX(chunk.startFrame, width),
      });
    });

    if (!retakeActive) {
      const occupied = new Set();
      this.timeline.keyframes.forEach(keyframe => {
        occupied.add(Number(keyframe.frame));
        const slot = Number(keyframe.defaultSlot);
        if (Number.isInteger(slot)) occupied.add(schedule.referenceFrames[slot]);
      });
      schedule.referenceFrames.forEach((frame, slot) => {
        if (occupied.has(frame)) return;
        items.push({
          kind: "slot",
          slot,
          tile: this._tileAtFrame(frame),
          frame,
          x: this._frameToX(frame, width),
          track: "main",
        });
      });
      this.timeline.keyframes.forEach((keyframe, index) => {
        const frame = Number(keyframe.frame);
        items.push({
          kind: "keyframe",
          index,
          frame,
          tile: this._tileAtFrame(frame),
          x: this._frameToX(frame, width),
          file: keyframe.imageFile,
          track: "main",
        });
      });
    }

    for (const [track, kind] of [["main", "video"], ["ic", "ic"], ["audio", "audio"]]) {
      this._mediaSegments(kind).forEach((segment, index) => {
        const span = tileSpan(Number(segment.tile) || 0);
        if (!span) return;
        items.push({
          kind: "media",
          track,
          mediaKind: kind,
          index,
          tile: Number(segment.tile) || 0,
          x: span.x,
          w: Math.max(6, span.end - span.x),
          segment,
        });
      });
    }
    return { width, schedule, items, retakeActive };
  }

  _trackBounds(track) {
    if (track === "main") return [RULER_HEIGHT, RULER_HEIGHT + BLOCK_HEIGHT];
    if (track === "ic") return [RULER_HEIGHT + BLOCK_HEIGHT, RULER_HEIGHT + BLOCK_HEIGHT + IC_TRACK_HEIGHT];
    return [RULER_HEIGHT + BLOCK_HEIGHT + IC_TRACK_HEIGHT, CANVAS_HEIGHT];
  }

  _hitTest(x, y) {
    const { items } = this._layout();
    const track = y < RULER_HEIGHT ? null
      : y < RULER_HEIGHT + BLOCK_HEIGHT ? "main"
      : y < RULER_HEIGHT + BLOCK_HEIGHT + IC_TRACK_HEIGHT ? "ic"
      : "audio";
    if (!track) return null;
    // Point items sit above spans, so test them first.
    for (const item of items) {
      if (item.track !== track) continue;
      if (item.kind === "slot" || item.kind === "keyframe") {
        if (Math.abs(x - item.x) <= 46) return item;
      }
    }
    for (const item of items) {
      if (item.kind !== "media" || item.track !== track) continue;
      if (x >= item.x && x <= item.x + item.w) return item;
    }
    const band = items.find(item => item.kind === "band" && x >= item.x && x <= item.x + item.w);
    return band ? { kind: "band", tile: band.tile, track } : null;
  }

  _renderCanvas() {
    if (!this.ctx) return;
    const { width, schedule, items, retakeActive } = this._layout();
    const ratio = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(CANVAS_HEIGHT * ratio);
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${CANVAS_HEIGHT}px`;
    const ctx = this.ctx;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, CANVAS_HEIGHT);

    ctx.fillStyle = "#2a2a2a";
    ctx.fillRect(0, 0, width, CANVAS_HEIGHT);

    this._drawRuler(ctx, width, schedule);

    // Alternating tile bands across every track, so tile ownership reads at a glance.
    for (const item of items) {
      if (item.kind !== "band") continue;
      ctx.fillStyle = item.tile % 2 ? "rgba(255,255,255,0.045)" : "rgba(255,255,255,0.015)";
      ctx.fillRect(item.x, RULER_HEIGHT, item.w, CANVAS_HEIGHT - RULER_HEIGHT);
      ctx.fillStyle = "#111";
      ctx.fillRect(Math.floor(item.x + item.w) - 1, RULER_HEIGHT, 1, CANVAS_HEIGHT - RULER_HEIGHT);
      if (item.tile === this.selectedTile) {
        ctx.fillStyle = "rgba(136,136,136,0.10)";
        ctx.fillRect(item.x, RULER_HEIGHT, item.w, CANVAS_HEIGHT - RULER_HEIGHT);
      }
      ctx.fillStyle = "#777";
      ctx.font = "bold 10px sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(`TILE ${item.tile}`, item.x + 6, RULER_HEIGHT + 4);
    }

    for (const track of ["main", "ic", "audio"]) {
      const [top, bottom] = this._trackBounds(track);
      ctx.fillStyle = "#111";
      ctx.fillRect(0, bottom - 1, width, 1);
      if (track !== "main") {
        const empty = !items.some(item => item.kind === "media" && item.track === track);
        if (empty) {
          ctx.fillStyle = "#555";
          ctx.font = "11px sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(
            track === "ic" ? "Drop an IC video or image" : "Drop audio",
            width / 2,
            (top + bottom) / 2,
          );
        }
      }
    }

    for (const item of items) {
      if (item.kind === "media") this._drawMediaBlock(ctx, item);
    }
    for (const item of items) {
      if (item.kind === "slot") this._drawSlot(ctx, item);
      if (item.kind === "keyframe") this._drawKeyframe(ctx, item);
    }

    if (retakeActive) {
      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.fillRect(0, RULER_HEIGHT, width, BLOCK_HEIGHT);
      ctx.fillStyle = "#e6b35a";
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("RETAKE MODE — keyframe and guide editing disabled", width / 2, RULER_HEIGHT + BLOCK_HEIGHT / 2);
      const tiles = retakeTileList(this.timeline.retake || {}, schedule.chunks.length);
      ctx.fillStyle = "rgba(230,179,90,0.16)";
      for (const tile of tiles) {
        const band = items.find(item => item.kind === "band" && item.tile === tile);
        if (band) ctx.fillRect(band.x, RULER_HEIGHT, band.w, CANVAS_HEIGHT - RULER_HEIGHT);
      }
    }

    // Playhead
    const playX = Math.floor(this._frameToX(this.selectedFrame, width)) + 0.5;
    ctx.strokeStyle = "#e5484d";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(playX, 0);
    ctx.lineTo(playX, CANVAS_HEIGHT);
    ctx.stroke();
    ctx.fillStyle = "#e5484d";
    ctx.fillRect(playX - 4, 0, 8, 5);
  }

  _drawRuler(ctx, width, schedule) {
    const total = Math.max(1, schedule.frameCount - 1);
    const rate = schedule.frameRate;
    const seconds = this.displayMode === "seconds";
    const targetPx = 90;
    const unitTotal = seconds ? total / rate : total;
    let step = Math.max(seconds ? 0.5 : 8, Math.pow(10, Math.floor(Math.log10(unitTotal / (width / targetPx)))));
    for (const candidate of seconds ? [0.5, 1, 2, 5, 10, 15, 30, 60] : [8, 16, 48, 96, 240, 480, 960]) {
      if (candidate >= unitTotal / (width / targetPx)) { step = candidate; break; }
    }
    ctx.fillStyle = "#1e1e1e";
    ctx.fillRect(0, 0, width, RULER_HEIGHT);
    ctx.font = "10px sans-serif";
    ctx.textBaseline = "middle";

    const minor = step / 5;
    ctx.fillStyle = "#444";
    for (let value = 0; value <= unitTotal + 1e-6; value += minor) {
      const frame = seconds ? value * rate : value;
      const x = Math.floor(this._frameToX(frame, width));
      ctx.fillRect(x, RULER_HEIGHT - 3, 1, 3);
    }
    for (let value = 0; value <= unitTotal + 1e-6; value += step) {
      const frame = seconds ? value * rate : value;
      const x = Math.floor(this._frameToX(frame, width));
      ctx.fillStyle = "#aaa";
      ctx.fillRect(x, RULER_HEIGHT - 6, 1, 6);
      if (value > 0) {
        ctx.textAlign = "center";
        ctx.fillText(seconds ? value.toFixed(2) : String(Math.round(value)), x, RULER_HEIGHT / 2);
      }
    }
    ctx.textAlign = "left";
    ctx.fillStyle = "#aaa";
    ctx.fillText(seconds ? "0" : "0", 4, RULER_HEIGHT / 2);
    ctx.fillStyle = "#111";
    ctx.fillRect(0, RULER_HEIGHT - 1, width, 1);
  }

  // Segment block in the Director's style: a translucent header bar carrying the type
  // tag and filename, and a prompt/label strip along the bottom.
  _drawMediaBlock(ctx, item) {
    const [top, bottom] = this._trackBounds(item.track);
    const height = bottom - top;
    const selected = this.selection && this.selection.kind === item.mediaKind && this.selection.index === item.index;
    ctx.save();
    ctx.beginPath();
    ctx.rect(item.x, top, item.w, height - 1);
    ctx.clip();

    ctx.fillStyle = item.mediaKind === "audio" ? "#243447" : item.mediaKind === "ic" ? "#2e2a40" : "#233a2e";
    ctx.fillRect(item.x, top + 1, item.w, height - 3);
    ctx.strokeStyle = selected ? "#888" : "#111";
    ctx.lineWidth = selected ? 2 : 1;
    ctx.strokeRect(item.x + 0.5, top + 1.5, item.w - 1, height - 4);

    const tag = item.mediaKind === "audio" ? "AUDIO" : item.mediaKind === "ic" ? "IC" : "VIDEO";
    ctx.font = "bold 10px sans-serif";
    const tagWidth = ctx.measureText(tag).width + 12;
    ctx.fillStyle = "rgba(0,0,0,0.60)";
    ctx.fillRect(item.x, top + 1, tagWidth, 16);
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(tag, item.x + tagWidth / 2, top + 9);

    let name = this._mediaName(item.mediaKind, item.segment);
    ctx.font = "9px sans-serif";
    const maxWidth = item.w - tagWidth - 10;
    if (maxWidth > 20) {
      while (name.length && ctx.measureText(`${name}…`).width > maxWidth) name = name.slice(0, -1);
      const textWidth = ctx.measureText(name).width;
      ctx.fillStyle = "rgba(0,0,0,0.50)";
      ctx.fillRect(item.x + tagWidth + 1, top + 1, textWidth + 8, 16);
      ctx.fillStyle = "#fff";
      ctx.textAlign = "left";
      ctx.fillText(name, item.x + tagWidth + 5, top + 9);
    }

    const strength = Number(item.segment.strength ?? 1);
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fillRect(item.x, bottom - 17, item.w, 16);
    ctx.fillStyle = "#bbb";
    ctx.font = "9px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`tile ${item.tile} · strength ${strength.toFixed(2)}`, item.x + 5, bottom - 9);
    ctx.restore();
  }

  _drawSlot(ctx, item) {
    const top = RULER_HEIGHT + 26;
    const w = 92;
    const h = 96;
    const x = Math.max(2, Math.min(item.x - w / 2, this._canvasWidth() - w - 2));
    ctx.save();
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = "#4a4a4a";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#1e1e1e";
    ctx.fillRect(x, top, w, h);
    ctx.strokeRect(x + 0.5, top + 0.5, w - 1, h - 1);
    ctx.setLineDash([]);
    ctx.fillStyle = "#555";
    ctx.font = "bold 22px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("+", x + w / 2, top + h / 2 - 6);
    ctx.fillStyle = "#777";
    ctx.font = "9px sans-serif";
    ctx.fillText(item.slot === 0 ? "start" : `tile ${item.slot}`, x + w / 2, top + h - 20);
    ctx.fillText(frameLabel(item.frame, this.node), x + w / 2, top + h - 8);
    ctx.restore();
  }

  _drawKeyframe(ctx, item) {
    const top = RULER_HEIGHT + 26;
    const w = 92;
    const h = 96;
    const x = Math.max(2, Math.min(item.x - w / 2, this._canvasWidth() - w - 2));
    const isReference = item.index === this._referenceIndex();
    const selected = this.selection && this.selection.kind === "keyframe" && this.selection.index === item.index;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, top, w, h);
    ctx.clip();
    ctx.fillStyle = "#181818";
    ctx.fillRect(x, top, w, h);

    const image = this._thumbnail(item.file);
    if (image && image.complete && image.naturalWidth) {
      const scale = Math.min(w / image.naturalWidth, (h - 18) / image.naturalHeight);
      const dw = image.naturalWidth * scale;
      const dh = image.naturalHeight * scale;
      ctx.drawImage(image, x + (w - dw) / 2, top + (h - 18 - dh) / 2, dw, dh);
    }

    ctx.fillStyle = "rgba(0,0,0,0.60)";
    ctx.fillRect(x, top, 42, 16);
    ctx.fillStyle = "#fff";
    ctx.font = "bold 10px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("IMAGE", x + 21, top + 8);

    if (isReference) {
      ctx.font = "bold 9px sans-serif";
      const text = "REFERENCE";
      const badgeWidth = ctx.measureText(text).width + 10;
      ctx.fillStyle = "rgba(215,173,99,0.85)";
      ctx.fillRect(x + w - badgeWidth, top, badgeWidth, 16);
      ctx.fillStyle = "#1a1a1a";
      ctx.fillText(text, x + w - badgeWidth / 2, top + 8);
    }

    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(x, top + h - 16, w, 16);
    ctx.fillStyle = "#ccc";
    ctx.font = "9px sans-serif";
    ctx.fillText(`K${item.index} · ${frameLabel(item.frame, this.node)}`, x + w / 2, top + h - 8);
    ctx.restore();

    ctx.strokeStyle = selected ? "#888" : isReference ? "#d7ad63" : "#111";
    ctx.lineWidth = selected ? 2 : 1;
    ctx.strokeRect(x + 0.5, top + 0.5, w - 1, h - 1);
  }

  _renderToolbar() {
    this.toolbar.innerHTML = "";
    this.toolbarRight.innerHTML = "";
    const retakeActive = Boolean(this.timeline.retake_mode);

    const addImages = makeButton("Add Image", "Upload one or more image keyframes", () => this.fileInput.click(), ICONS.upload);
    const addVideo = makeButton("Add Video", "Add a standard video guide to the selected tile", () => this.mediaInputs.video.click(), ICONS.video);
    const addICVideo = makeButton("Add IC Video", "Add an IC-LoRA video or image guide to the selected tile", () => this.mediaInputs.ic.click(), ICONS.motion);
    const addAudio = makeButton("Add Audio", "Add a custom audio segment", () => this.mediaInputs.audio.click(), ICONS.audio);
    for (const button of [addImages, addVideo, addICVideo, addAudio]) {
      button.disabled = retakeActive;
      if (retakeActive) button.title = "Disabled while Retake mode is active";
      this.toolbar.appendChild(button);
    }
    const deleteButton = makeIconButton("Delete", "Delete the selected item", () => this._deleteSelection(), ICONS.trash, true);
    deleteButton.disabled = retakeActive || !this.selection;
    this.toolbar.appendChild(deleteButton);

    const retakeButton = makeButton(
      "Retake Mode (BETA)",
      retakeActive ? "Remove the retake guide" : "Add a retake video to the selected tile",
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
    this.toolbarRight.appendChild(retakeButton);

    const units = document.createElement("div");
    units.className = "pr-segmented-control";
    for (const mode of ["frames", "seconds"]) {
      const segment = document.createElement("div");
      segment.className = `pr-segment${this.displayMode === mode ? " active" : ""}`;
      segment.textContent = mode === "frames" ? "Frames" : "Seconds";
      segment.addEventListener("click", () => {
        this.displayMode = mode;
        this.node._loopingDirectorDisplayMode = mode;
        this.node.properties = this.node.properties || {};
        this.node.properties.looping_director_display_mode = mode;
        this.refresh();
      });
      units.appendChild(segment);
    }
    this.toolbarRight.appendChild(units);

    if (this.node._loopingDirectorWarning) {
      const warning = document.createElement("span");
      warning.className = "ld-status ld-warning";
      warning.textContent = this.node._loopingDirectorWarning;
      this.toolbarRight.appendChild(warning);
    }
  }

  _deleteSelection() {
    if (!this.selection) return;
    if (this.selection.kind === "keyframe") {
      this._deleteKeyframe(this.selection.index);
    } else {
      this._removeMedia(this.selection.kind, this.selection.index);
      return;
    }
    this.selection = null;
    this.selectedKeyframe = null;
    this._commit();
    this.refresh();
  }

  _renderControls() {
    const schedule = loopingSchedule(this.node);
    const tile = Math.max(0, Math.min(this.selectedTile ?? 0, schedule.chunks.length - 1));
    const chunk = schedule.chunks[tile];
    this.infoRow.innerHTML = "";

    const left = document.createElement("div");
    left.className = "ld-info-left";
    const timecode = document.createElement("span");
    timecode.className = "pr-timecode";
    timecode.textContent = frameLabel(this.selectedFrame, this.node);
    const bounds = document.createElement("span");
    bounds.className = "pr-segment-bounds";
    bounds.textContent = chunk
      ? `Tile ${tile} | Start: ${frameLabel(chunk.startFrame, this.node)} | End: ${frameLabel(chunk.endFrame, this.node)} | Length: ${frameLabel(chunk.endFrame - chunk.startFrame, this.node)}`
      : "";
    left.appendChild(timecode);
    left.appendChild(bounds);
    this.infoRow.appendChild(left);

    // Contextual strength controls for the selected block, like the Director's row.
    const selected = this.selection;
    const segment = selected && selected.kind !== "keyframe"
      ? this._mediaSegments(selected.kind)[selected.index]
      : null;
    const addNumber = (labelText, value, min, max, step, onChange) => {
      const label = document.createElement("span");
      label.className = "pr-strength-label";
      label.textContent = labelText;
      const input = document.createElement("input");
      input.type = "number";
      input.className = "pr-strength-input";
      input.min = String(min);
      input.max = String(max);
      input.step = String(step);
      input.value = String(value);
      input.addEventListener("change", event => {
        onChange(Math.max(min, Math.min(max, Number(event.target.value) || 0)));
        this._commit();
        this.refresh();
      });
      this.infoRow.appendChild(label);
      this.infoRow.appendChild(input);
    };

    if (segment) {
      addNumber("Tile:", Number(segment.tile) || 0, 0, Math.max(0, schedule.chunks.length - 1), 1, value => {
        segment.tile = value;
        delete segment.start;
        delete segment.length;
      });
      addNumber("Trim:", Math.max(0, Math.round(Number(segment.trimStart) || 0)), 0, 1000000, 1, value => {
        segment.trimStart = value;
      });
      addNumber("Guide Strength:", Number(segment.strength ?? 1), 0, 1, 0.01, value => {
        segment.strength = value;
      });
      if (selected.kind === "ic") {
        addNumber("Attention:", Number(segment.attentionStrength ?? 0.65), 0, 1, 0.01, value => {
          segment.attentionStrength = value;
        });
      }
    } else if (this.timeline.retake_mode && this.timeline.retake) {
      const retake = this.timeline.retake;
      const last = Math.max(0, schedule.chunks.length - 1);
      const tiles = retakeTileList(retake, schedule.chunks.length);
      addNumber("Retake first tile:", tiles[0], 0, last, 1, value => {
        setRetakeTiles(retake, value, Math.max(value, tiles[tiles.length - 1]), last);
      });
      addNumber("last tile:", tiles[tiles.length - 1], 0, last, 1, value => {
        setRetakeTiles(retake, Math.min(value, tiles[0]), value, last);
      });
      addNumber("Retake Strength:", Number(retake.strength ?? 1), 0, 1, 0.01, value => {
        retake.strength = value;
      });
    } else {
      const label = document.createElement("span");
      label.className = "pr-strength-label";
      label.textContent = "External reference:";
      const select = document.createElement("select");
      select.className = "pr-settings-select";
      this.timeline.keyframes.forEach((keyframe, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        const filename = String(keyframe.imageFile || "").split("/").pop() || "image";
        option.textContent = `K${index} · ${frameLabel(Number(keyframe.frame) || 0, this.node)} · ${filename}`;
        option.selected = index === this._referenceIndex();
        select.appendChild(option);
      });
      if (!this.timeline.keyframes.length) {
        const option = document.createElement("option");
        option.textContent = "No keyframes";
        select.appendChild(option);
        select.disabled = true;
      }
      select.addEventListener("change", event => {
        this._setReferenceIndex(Number(event.target.value));
        this._commit();
        this.refresh();
      });
      this.infoRow.appendChild(label);
      this.infoRow.appendChild(select);
    }

    this._renderPlayer(schedule);
    this._renderGuideEncoding();
  }

  _renderPlayer(schedule) {
    this.playerRow.innerHTML = "";
    const play = document.createElement("button");
    play.className = "pr-icon-btn";
    play.style.padding = "4px";
    play.innerHTML = this._playTimer ? ICONS.pause : ICONS.play;
    play.title = "Scrub the playhead across the clip";
    play.addEventListener("click", () => this._togglePlay());
    this.playerRow.appendChild(play);

    const loop = document.createElement("button");
    loop.className = "pr-icon-btn";
    loop.style.padding = "4px";
    loop.innerHTML = ICONS.loop;
    loop.title = "Loop the playhead";
    if (this._loopPlayback) loop.classList.add("toggle-on");
    loop.addEventListener("click", () => {
      this._loopPlayback = !this._loopPlayback;
      this._renderPlayer(schedule);
    });
    this.playerRow.appendChild(loop);

    const seek = document.createElement("input");
    seek.type = "range";
    seek.className = "pr-seek-bar";
    seek.style.flex = "1";
    seek.min = "0";
    seek.max = String(Math.max(0, schedule.frameCount - 1));
    seek.step = String(TIME_SCALE);
    seek.value = String(this.selectedFrame);
    seek.addEventListener("input", event => {
      this.selectedFrame = snapFrame(Number(event.target.value), schedule.frameCount);
      this.selectedTile = this._tileAtFrame(this.selectedFrame);
      this._renderCanvas();
      this._renderControls();
    });
    this.playerRow.appendChild(seek);

    const zoom = document.createElement("div");
    zoom.className = "pr-zoom-controls";
    const setZoom = value => {
      this.zoom = Math.max(1, Math.min(8, value));
      this._renderCanvas();
      this._renderControls();
    };
    const out = document.createElement("button");
    out.className = "pr-icon-btn";
    out.style.padding = "4px";
    out.innerHTML = ICONS.minus;
    out.title = "Zoom out";
    out.addEventListener("click", () => setZoom(this.zoom / 1.5));
    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "pr-zoom-slider";
    slider.min = "1";
    slider.max = "8";
    slider.step = "0.1";
    slider.value = String(this.zoom);
    slider.addEventListener("input", event => setZoom(Number(event.target.value)));
    const inBtn = document.createElement("button");
    inBtn.className = "pr-icon-btn";
    inBtn.style.padding = "4px";
    inBtn.innerHTML = ICONS.plus;
    inBtn.title = "Zoom in";
    inBtn.addEventListener("click", () => setZoom(this.zoom * 1.5));
    const fit = document.createElement("button");
    fit.className = "pr-icon-btn";
    fit.style.padding = "4px";
    fit.innerHTML = ICONS.fit;
    fit.title = "Fit the clip to the viewport";
    fit.addEventListener("click", () => setZoom(1));
    for (const control of [out, slider, inBtn, fit]) zoom.appendChild(control);
    this.playerRow.appendChild(zoom);
  }

  _togglePlay() {
    if (this._playTimer) {
      clearInterval(this._playTimer);
      this._playTimer = null;
      this._renderControls();
      return;
    }
    const schedule = loopingSchedule(this.node);
    const step = TIME_SCALE;
    const interval = (step / Math.max(1, schedule.frameRate)) * 1000;
    this._playTimer = setInterval(() => {
      let next = this.selectedFrame + step;
      if (next > schedule.frameCount - 1) {
        if (!this._loopPlayback) {
          clearInterval(this._playTimer);
          this._playTimer = null;
          this._renderControls();
          return;
        }
        next = 0;
      }
      this.selectedFrame = next;
      this.selectedTile = this._tileAtFrame(next);
      this._renderCanvas();
    }, Math.max(30, interval));
    this._renderControls();
  }

  _renderGuideEncoding() {
    const settings = normalizeIcSettings(this.timeline.ic_settings);
    this.timeline.ic_settings = settings;
    this.settingsRow.innerHTML = "";

    const update = (key, value) => {
      settings[key] = value;
      this.timeline.ic_settings = normalizeIcSettings(settings);
      this._commit();
      this.refresh();
    };

    const heading = document.createElement("span");
    heading.className = "pr-strength-label";
    heading.style.marginLeft = "0";
    heading.textContent = "Guide encoding";
    this.settingsRow.appendChild(heading);

    const addSelect = (key, options, labelText, format) => {
      const label = document.createElement("label");
      label.textContent = labelText;
      const select = document.createElement("select");
      for (const option of options) {
        const item = document.createElement("option");
        item.value = option;
        item.textContent = format ? format(option) : option;
        item.selected = settings[key] === option;
        select.appendChild(item);
      }
      select.addEventListener("change", event => update(key, event.target.value));
      label.appendChild(select);
      this.settingsRow.appendChild(label);
    };

    addSelect("crop", ["center", "disabled"], "crop", option => option === "center" ? "center crop" : "stretch to fit");
    addSelect("upscale_method", IC_UPSCALE_METHODS, "resample");

    const tiled = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = settings.use_tiled_encode;
    checkbox.addEventListener("change", event => update("use_tiled_encode", event.target.checked));
    tiled.appendChild(checkbox);
    tiled.appendChild(document.createTextNode("tiled encode"));
    tiled.title = "VAE-encode guides in tiles to cut peak VRAM";
    this.settingsRow.appendChild(tiled);

    for (const [key, min, max, step, labelText] of [
      ["tile_size", 64, 512, 32, "size"],
      ["tile_overlap", 16, 256, 16, "overlap"],
    ]) {
      const label = document.createElement("label");
      label.textContent = labelText;
      const input = document.createElement("input");
      input.type = "number";
      input.min = String(min);
      input.max = String(max);
      input.step = String(step);
      input.value = String(settings[key]);
      input.disabled = !settings.use_tiled_encode;
      input.addEventListener("change", event => update(key, Number(event.target.value)));
      label.appendChild(input);
      this.settingsRow.appendChild(label);
    }
  }

  _buildPromptPanels() {
    const makePanel = (container, labelText) => {
      const wrapper = document.createElement("div");
      wrapper.className = "pr-prompt-wrapper";
      const label = document.createElement("div");
      label.className = "pr-prompt-label";
      label.textContent = labelText;
      const area = document.createElement("textarea");
      area.className = "pr-prompt-area";
      area.addEventListener("focus", () => {
        wrapper.classList.add("focus-active");
        this.container.classList.add("has-focus");
      });
      area.addEventListener("blur", () => {
        wrapper.classList.remove("focus-active");
        this.container.classList.remove("has-focus");
      });
      wrapper.appendChild(label);
      wrapper.appendChild(area);
      container.appendChild(wrapper);
      return { label, area };
    };

    this.tilePanel = makePanel(this.tileProp, "TILE PROMPT");
    this.tilePanel.area.placeholder = DEFAULT_TILE_PROMPT;
    this.tilePanel.area.addEventListener("input", event => {
      const tile = Math.max(0, this.selectedTile ?? 0);
      this.timeline.tile_prompts[tile] = event.target.value;
      this._commit();
    });

    this.globalPanel = makePanel(this.globalProp, "GLOBAL PROMPT");
    this.globalPanel.area.placeholder = "Shared identity, setting, lighting, style and audio…";
    this.globalPanel.area.addEventListener("input", event => {
      const widget = this.node.widgets?.find(item => item.name === "global_prompt");
      if (widget) widget.value = event.target.value;
      this._commit();
    });
  }

  _renderPrompts() {
    const chunks = loopingSchedule(this.node).chunks;
    this._ensurePromptCount(chunks.length);
    const tile = Math.max(0, Math.min(this.selectedTile ?? 0, chunks.length - 1));
    const chunk = chunks[tile];
    this.tilePanel.label.textContent = chunk
      ? `TILE PROMPT — TILE ${tile} · ${frameLabel(chunk.startFrame, this.node)}–${frameLabel(Math.max(chunk.startFrame, chunk.endFrame - 1), this.node)}`
      : "TILE PROMPT";
    if (document.activeElement !== this.tilePanel.area) {
      this.tilePanel.area.value = this.timeline.tile_prompts[tile] || "";
    }
    const widget = this.node.widgets?.find(item => item.name === "global_prompt");
    if (document.activeElement !== this.globalPanel.area) {
      this.globalPanel.area.value = typeof widget?.value === "string" ? widget.value : "";
    }
  }

  _renderAll() {
    this._buildSidebar();
    this._renderToolbar();
    this._renderCanvas();
    this._renderControls();
    this._renderPrompts();
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
    const tiles = tileCount(this.node);
    if (this.selectedTile === undefined || this.selectedTile === null || this.selectedTile >= tiles) {
      this.selectedTile = this._tileAtFrame(this.selectedFrame);
    }
    this._renderAll();
    if (commit || moved || promptsChanged || guideChanged) this._commit();
  }

  syncLayout() {
    if (this.container.parentElement) this.container.style.width = "100%";
    // The canvas is sized from the viewport, so a node resize has to redraw it.
    this._renderCanvas();
  }

  destroy() {
    if (this._playTimer) {
      clearInterval(this._playTimer);
      this._playTimer = null;
    }
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
      // The global prompt is edited in the GLOBAL PROMPT panel, as in the Director.
      // The widget itself stays in the schema and keeps carrying the value.
      hideWidget(this.widgets?.find(widget => widget.name === "global_prompt"));
      this.properties = this.properties || {};
      this._loopingDirectorDisplayMode = this.properties.looping_director_display_mode || "seconds";

      const container = document.createElement("div");
      const domWidget = this.addDOMWidget("looping_director_ui", "looping_director_ui", container, {
        getValue: () => "",
        setValue: () => {},
      });
      const editorHeight = RULER_HEIGHT + BLOCK_HEIGHT + IC_TRACK_HEIGHT + AUDIO_TRACK_HEIGHT + 280;
      domWidget.computeSize = width => [Math.max(10, width - 20), editorHeight];
      this._loopingDirectorEditor = new LoopingDirectorEditor(this, container);
      this.size[0] = Math.max(this.size?.[0] || 0, 900);
      this.size[1] = Math.max(this.size?.[1] || 0, editorHeight + 180);
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
