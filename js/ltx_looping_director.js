const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const TIME_SCALE = 8;
const DEFAULT_FRAME_RATE = 24;
const DEFAULT_TOTAL_DURATION = 48;
const DEFAULT_TILE_DURATION = 10;
const DEFAULT_OVERLAP_DURATION = 2;

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
  return {
    version: 1,
    tile_prompts: Array.isArray(data.tile_prompts) ? data.tile_prompts.map(value => String(value ?? "")) : [],
    keyframes: Array.isArray(data.keyframes) ? data.keyframes.filter(item => item && typeof item === "object") : [],
    overflow_tile_prompts: Array.isArray(data.overflow_tile_prompts) ? data.overflow_tile_prompts.map(value => String(value ?? "")) : [],
  };
}

function numericWidget(node, name, fallback) {
  const widget = node.widgets?.find(item => item.name === name);
  const value = Number(widget?.value);
  return Number.isFinite(value) ? value : fallback;
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

function makeButton(label, title, handler) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.title = title || label;
  button.className = "ld-button";
  button.addEventListener("click", handler);
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
    this.container.className = "ld-editor";
    this.container.innerHTML = `
      <style>
        .ld-editor { color: #ddd; font: 12px ui-sans-serif, system-ui, sans-serif; width: 100%; box-sizing: border-box; padding: 6px 0; }
        .ld-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 7px; }
        .ld-button { background: #292929; border: 1px solid #555; border-radius: 4px; color: #ddd; padding: 4px 8px; cursor: pointer; }
        .ld-button:hover { background: #383838; }
        .ld-button:disabled { cursor: default; opacity: .45; }
        .ld-status { color: #999; margin-left: 4px; }
        .ld-warning { color: #e6b35a; }
        .ld-strip-label { color: #aaa; margin: 5px 0 3px; font-size: 11px; }
        .ld-keyframe-strip { height: 90px; position: relative; overflow: hidden; border: 1px solid #444; border-radius: 4px; background: #171717; cursor: crosshair; transition: border-color .15s; }
        .ld-keyframe-strip.dragover { border-color: #8eb7ff; background: #202735; }
        .ld-tile-band { position: absolute; top: 0; bottom: 0; border-right: 1px solid rgba(255,255,255,.12); background: rgba(100,130,180,.08); pointer-events: none; }
        .ld-tile-band:nth-child(even) { background: rgba(130,100,180,.08); }
        .ld-frame-marker { position: absolute; top: 2px; width: 58px; height: 72px; transform: translateX(-50%); background: #282828; border: 1px solid #777; border-radius: 4px; padding: 2px; box-sizing: border-box; cursor: grab; z-index: 3; }
        .ld-frame-marker.reference { border-color: #d7ad63; }
        .ld-frame-marker.auto { border-color: #7e9e88; }
        .ld-frame-marker.selected { border-color: #8eb7ff; box-shadow: 0 0 0 1px #5178aa; }
        .ld-frame-marker.invalid { border-color: #c86767; }
        .ld-frame-marker img { display: block; width: 52px; height: 52px; object-fit: contain; background: #111; border-radius: 2px; pointer-events: none; }
        .ld-frame-marker span { display: block; text-align: center; color: #bbb; font-size: 10px; line-height: 14px; white-space: nowrap; overflow: hidden; }
        .ld-frame-marker .ld-delete { position: absolute; right: -5px; top: -7px; width: 17px; height: 17px; padding: 0; border-radius: 9px; border: 1px solid #777; background: #292929; color: #eee; cursor: pointer; }
        .ld-prompts { display: flex; gap: 6px; overflow-x: auto; padding: 3px 0 5px; }
        .ld-tile { flex: 0 0 170px; min-height: 115px; border: 1px solid #444; border-radius: 4px; background: #202020; padding: 5px; box-sizing: border-box; }
        .ld-tile-title { color: #9bb7e6; font-size: 11px; margin-bottom: 4px; }
        .ld-tile-range { color: #777; font-size: 10px; margin-left: 3px; }
        .ld-tile textarea { width: 100%; height: 82px; resize: vertical; box-sizing: border-box; border: 1px solid #444; border-radius: 3px; background: #151515; color: #eee; padding: 5px; font: 11px ui-sans-serif, system-ui, sans-serif; }
      </style>
      <div class="ld-toolbar"></div>
      <div class="ld-strip-label">Keyframes — drop images here or use Add images. New images land at frame 0 and the middle of each tile overlap; drag a marker to adjust its index.</div>
      <div class="ld-keyframe-strip"></div>
      <div class="ld-strip-label">One prompt per looping tile. Tile prompts are fixed to their tile and cannot overlap.</div>
      <div class="ld-prompts"></div>
    `;
    this.toolbar = this.container.querySelector(".ld-toolbar");
    this.strip = this.container.querySelector(".ld-keyframe-strip");
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
      this.timeline.tile_prompts.push(restored ?? "");
      changed = true;
    }
    this.node._loopingDirectorWarning = this.timeline.overflow_tile_prompts?.some(prompt => prompt.trim())
      ? "Some tile prompts are outside the current clip and were retained."
      : null;
    return changed;
  }

  _commit() {
    const value = JSON.stringify({
      version: 1,
      tile_prompts: this.timeline.tile_prompts,
      keyframes: this.timeline.keyframes,
      overflow_tile_prompts: this.timeline.overflow_tile_prompts || [],
    });
    const widget = this.node.widgets?.find(item => item.name === "timeline_data");
    if (widget) widget.value = value;
    this.node.properties = this.node.properties || {};
    this.node.properties.timeline_data = value;
    this.node.properties.looping_director_timeline = value;
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

  _renderToolbar() {
    this.toolbar.innerHTML = "";
    this.toolbar.appendChild(makeButton("Add images", "Upload one or more image keyframes", () => this.fileInput.click()));
    this.toolbar.appendChild(makeButton("Frames", "Display keyframe labels as frame numbers", () => {
      this.displayMode = "frames";
      this.node._loopingDirectorDisplayMode = this.displayMode;
      this.node.properties = this.node.properties || {};
      this.node.properties.looping_director_display_mode = this.displayMode;
      this.refresh();
    }));
    this.toolbar.appendChild(makeButton("Seconds", "Display keyframe labels as seconds", () => {
      this.displayMode = "seconds";
      this.node._loopingDirectorDisplayMode = this.displayMode;
      this.node.properties = this.node.properties || {};
      this.node.properties.looping_director_display_mode = this.displayMode;
      this.refresh();
    }));
    this.toolbar.appendChild(makeButton("Delete selected", "Delete the selected keyframe", () => {
      if (this.selectedKeyframe === null) return;
      this._deleteKeyframe(this.selectedKeyframe);
      this._commit();
      this.refresh();
    }));
    const referenceLabel = document.createElement("label");
    referenceLabel.className = "ld-status";
    referenceLabel.textContent = "External reference";
    const referenceSelect = document.createElement("select");
    referenceSelect.className = "ld-button";
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
    const selected = document.createElement("span");
    selected.className = "ld-status";
    selected.textContent = `Selected ${frameLabel(this.selectedFrame, this.node)}`;
    this.toolbar.appendChild(selected);
    if (this.node._loopingDirectorWarning) {
      const warning = document.createElement("span");
      warning.className = "ld-status ld-warning";
      warning.textContent = this.node._loopingDirectorWarning;
      this.toolbar.appendChild(warning);
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
    this.timeline.keyframes.forEach((keyframe, index) => {
      const frame = Number(keyframe.frame);
      const marker = document.createElement("div");
      const valid = Number.isInteger(frame) && frame >= 0 && frame < totalFrames && frame % TIME_SCALE === 0;
      marker.className = `ld-frame-marker${index === referenceIndex ? " reference" : ""}${keyframe.autoPosition ? " auto" : ""}${index === this.selectedKeyframe ? " selected" : ""}${valid ? "" : " invalid"}`;
      const markerPosition = Math.max(0, Math.min(100, frame / Math.max(1, totalFrames - 1) * 100));
      marker.style.left = `${markerPosition}%`;
      marker.style.transform = markerPosition <= 4 ? "translateX(0)" : markerPosition >= 96 ? "translateX(-100%)" : "translateX(-50%)";
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

      marker.addEventListener("pointerdown", event => {
        event.stopPropagation();
        this.selectedKeyframe = index;
        if (valid) this.selectedFrame = frame;
        this.strip.querySelectorAll(".ld-frame-marker").forEach(item => {
          item.classList.toggle("selected", item === marker);
        });
        marker.setPointerCapture?.(event.pointerId);
        const move = moveEvent => {
          const rect = this.strip.getBoundingClientRect();
          const ratio = Math.max(0, Math.min(1, (moveEvent.clientX - rect.left) / Math.max(1, rect.width)));
          keyframe.frame = snapFrame(ratio * (totalFrames - 1), totalFrames);
          keyframe.autoPosition = false;
          marker.style.left = `${Math.max(0, Math.min(100, keyframe.frame / Math.max(1, totalFrames - 1) * 100))}%`;
          const position = keyframe.frame / Math.max(1, totalFrames - 1) * 100;
          marker.style.transform = position <= 4 ? "translateX(0)" : position >= 96 ? "translateX(-100%)" : "translateX(-50%)";
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
  }

  _renderPrompts() {
    const chunks = loopingSchedule(this.node).chunks;
    const count = chunks.length;
    this._ensurePromptCount(count);
    this.prompts.innerHTML = "";
    for (let index = 0; index < count; index += 1) {
      const tile = document.createElement("div");
      tile.className = "ld-tile";
      const title = document.createElement("div");
      title.className = "ld-tile-title";
      title.textContent = `Tile ${index}`;
      const range = document.createElement("span");
      range.className = "ld-tile-range";
      const chunk = chunks[index];
      range.textContent = `#${index + 1}/${count} · ${frameLabel(chunk.startFrame, this.node)}–${frameLabel(Math.max(chunk.startFrame, chunk.endFrame - 1), this.node)}`;
      title.appendChild(range);
      tile.appendChild(title);
      const textarea = document.createElement("textarea");
      textarea.placeholder = index === 0 ? "Tile-specific action and camera motion" : "Tile-specific action and camera motion";
      textarea.value = this.timeline.tile_prompts[index] || "";
      textarea.addEventListener("input", event => {
        this.timeline.tile_prompts[index] = event.target.value;
        this._commit();
      });
      tile.appendChild(textarea);
      this.prompts.appendChild(tile);
    }
  }

  async _uploadFiles(files) {
    const schedule = loopingSchedule(this.node);
    const totalFrames = schedule.frameCount;
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
      let slot = schedule.referenceFrames.findIndex((_, candidate) => !usedSlots.has(candidate));
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
    this._renderToolbar();
    this._renderStrip();
    this._renderPrompts();
    if (commit || moved || promptsChanged) this._commit();
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
      if ((!window.LiteGraph || !window.LiteGraph.vueNodesMode) && this.inputs) {
        const timelineInput = this.inputs.findIndex(input => input.name === "timeline_data");
        if (timelineInput !== -1 && this.inputs[timelineInput].link == null) {
          this.removeInput(timelineInput);
        }
      }
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
      const raw = info?.properties?.looping_director_timeline
        || info?.properties?.timeline_data
        || this.widgets?.find(widget => widget.name === "timeline_data")?.value;
      if (raw && this._loopingDirectorEditor) {
        this._loopingDirectorEditor.setTimeline(raw);
      }
      return result;
    };

    const originalOnResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function () {
      const result = originalOnResize?.apply(this, arguments);
      this._loopingDirectorEditor?.syncLayout();
      return result;
    };
  },
});
