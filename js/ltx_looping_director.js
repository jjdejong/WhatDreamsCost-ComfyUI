const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const TIME_SCALE = 8;

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

function temporalChunks(node) {
  const frameCount = Math.max(9, Math.round(numericWidget(node, "frame_count", 241)));
  const tileSize = Math.max(TIME_SCALE, Math.round(numericWidget(node, "temporal_tile_size", 240) / TIME_SCALE));
  const overlap = Math.max(0, Math.round(numericWidget(node, "temporal_overlap", 64) / TIME_SCALE));
  const latentFrames = Math.floor((frameCount - 1) / TIME_SCALE) + 1;
  const step = Math.max(1, tileSize - overlap);
  const chunks = [];
  for (let start = 0, end = tileSize;
       start < latentFrames + tileSize - overlap && end < latentFrames + tileSize - overlap;
       start += step, end += step) {
    chunks.push({
      startFrame: start * TIME_SCALE,
      endFrame: Math.min(end * TIME_SCALE, frameCount),
    });
  }
  return chunks;
}

function tileCount(node) {
  return temporalChunks(node).length;
}

function snapFrame(frame, frameCount) {
  const last = Math.max(0, frameCount - 1);
  return Math.max(0, Math.min(last, Math.round(frame / TIME_SCALE) * TIME_SCALE));
}

function frameCount(node) {
  return Math.max(9, Math.round(numericWidget(node, "frame_count", 241)));
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
    this.displayMode = "frames";
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
        .ld-keyframe-strip { height: 90px; position: relative; overflow: hidden; border: 1px solid #444; border-radius: 4px; background: #171717; cursor: crosshair; }
        .ld-tile-band { position: absolute; top: 0; bottom: 0; border-right: 1px solid rgba(255,255,255,.12); background: rgba(100,130,180,.08); pointer-events: none; }
        .ld-tile-band:nth-child(even) { background: rgba(130,100,180,.08); }
        .ld-frame-marker { position: absolute; top: 2px; width: 58px; height: 72px; transform: translateX(-50%); background: #282828; border: 1px solid #777; border-radius: 4px; padding: 2px; box-sizing: border-box; cursor: grab; z-index: 3; }
        .ld-frame-marker.selected { border-color: #8eb7ff; box-shadow: 0 0 0 1px #5178aa; }
        .ld-frame-marker.invalid { border-color: #c86767; }
        .ld-frame-marker img { display: block; width: 52px; height: 52px; object-fit: cover; border-radius: 2px; pointer-events: none; }
        .ld-frame-marker span { display: block; text-align: center; color: #bbb; font-size: 10px; line-height: 14px; white-space: nowrap; overflow: hidden; }
        .ld-frame-marker .ld-delete { position: absolute; right: -5px; top: -7px; width: 17px; height: 17px; padding: 0; border-radius: 9px; border: 1px solid #777; background: #292929; color: #eee; cursor: pointer; }
        .ld-prompts { display: flex; gap: 6px; overflow-x: auto; padding: 3px 0 5px; }
        .ld-tile { flex: 0 0 170px; min-height: 115px; border: 1px solid #444; border-radius: 4px; background: #202020; padding: 5px; box-sizing: border-box; }
        .ld-tile-title { color: #9bb7e6; font-size: 11px; margin-bottom: 4px; }
        .ld-tile-range { color: #777; font-size: 10px; margin-left: 3px; }
        .ld-tile textarea { width: 100%; height: 82px; resize: vertical; box-sizing: border-box; border: 1px solid #444; border-radius: 3px; background: #151515; color: #eee; padding: 5px; font: 11px ui-sans-serif, system-ui, sans-serif; }
      </style>
      <div class="ld-toolbar"></div>
      <div class="ld-strip-label">Keyframes — click to select a frame, add images, or drag markers on the 8-frame grid.</div>
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
  }

  _bindTimingWidgets() {
    for (const name of ["frame_count", "temporal_tile_size", "temporal_overlap", "frame_rate"]) {
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
    if (prompts.length > count) {
      this.timeline.overflow_tile_prompts = prompts.slice(count).concat(this.timeline.overflow_tile_prompts || []);
      this.timeline.tile_prompts = prompts.slice(0, count);
      if (this.timeline.overflow_tile_prompts.some(prompt => prompt.trim())) {
        this.node._loopingDirectorWarning = "Some tile prompts are outside the current clip and were retained.";
      }
    }
    while (this.timeline.tile_prompts.length < count) {
      const restored = this.timeline.overflow_tile_prompts?.shift();
      this.timeline.tile_prompts.push(restored ?? "");
    }
    this.node._loopingDirectorWarning = this.timeline.overflow_tile_prompts?.some(prompt => prompt.trim())
      ? "Some tile prompts are outside the current clip and were retained."
      : null;
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
      this.timeline.keyframes.splice(this.selectedKeyframe, 1);
      this.selectedKeyframe = null;
      this._commit();
      this.refresh();
    }));
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
    const totalFrames = frameCount(this.node);
    const chunks = temporalChunks(this.node);
    this.strip.innerHTML = "";
    for (let index = 0; index < chunks.length; index += 1) {
      const chunk = chunks[index];
      const band = document.createElement("div");
      band.className = "ld-tile-band";
      const left = chunk.startFrame / Math.max(1, totalFrames - 1) * 100;
      const right = chunk.endFrame / Math.max(1, totalFrames - 1) * 100;
      band.style.left = `${Math.max(0, Math.min(100, left))}%`;
      band.style.width = `${Math.max(0, Math.min(100, right) - Math.max(0, left))}%`;
      band.title = `Tile ${index}: ${chunk.startFrame}–${Math.max(chunk.startFrame, chunk.endFrame - 1)}f`;
      this.strip.appendChild(band);
    }

    this.timeline.keyframes.forEach((keyframe, index) => {
      const frame = Number(keyframe.frame);
      const marker = document.createElement("div");
      const valid = Number.isInteger(frame) && frame >= 0 && frame < totalFrames && frame % TIME_SCALE === 0;
      marker.className = `ld-frame-marker${index === this.selectedKeyframe ? " selected" : ""}${valid ? "" : " invalid"}`;
      marker.style.left = `${Math.max(0, Math.min(100, frame / Math.max(1, totalFrames - 1) * 100))}%`;
      marker.title = valid ? `Keyframe at ${frameLabel(frame, this.node)}` : "Invalid keyframe position";
      const image = document.createElement("img");
      if (keyframe.imageFile) {
        const parts = String(keyframe.imageFile).split("/");
        const filename = parts.pop() || "";
        const subfolder = parts.join("/");
        image.src = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=${encodeURIComponent(subfolder)}`);
      }
      marker.appendChild(image);
      const label = document.createElement("span");
      label.textContent = Number.isFinite(frame) ? frameLabel(frame, this.node) : "invalid";
      marker.appendChild(label);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "ld-delete";
      remove.textContent = "×";
      remove.addEventListener("click", event => {
        event.stopPropagation();
        this.timeline.keyframes.splice(index, 1);
        this.selectedKeyframe = null;
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
          marker.style.left = `${Math.max(0, Math.min(100, keyframe.frame / Math.max(1, totalFrames - 1) * 100))}%`;
          label.textContent = frameLabel(keyframe.frame, this.node);
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
    const chunks = temporalChunks(this.node);
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
      range.textContent = `#${index + 1}/${count} · ${chunk.startFrame}–${Math.max(chunk.startFrame, chunk.endFrame - 1)}f`;
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
    const totalFrames = frameCount(this.node);
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
      const frame = snapFrame(this.selectedFrame + index * TIME_SCALE, totalFrames);
      this.timeline.keyframes.push({ frame, imageFile });
    }
    this.timeline.keyframes.sort((left, right) => Number(left.frame) - Number(right.frame));
    this._commit();
    this.refresh();
  }

  refresh(commit = false) {
    this._ensurePromptCount(tileCount(this.node));
    this._renderToolbar();
    this._renderStrip();
    this._renderPrompts();
    if (commit) this._commit();
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
      if ((!window.LiteGraph || !window.LiteGraph.vueNodesMode) && this.inputs) {
        const timelineInput = this.inputs.findIndex(input => input.name === "timeline_data");
        if (timelineInput !== -1 && this.inputs[timelineInput].link == null) {
          this.removeInput(timelineInput);
        }
      }
      this.properties = this.properties || {};
      this._loopingDirectorDisplayMode = this.properties.looping_director_display_mode || "frames";

      const container = document.createElement("div");
      const domWidget = this.addDOMWidget("looping_director_ui", "looping_director_ui", container, {
        getValue: () => "",
        setValue: () => {},
      });
      domWidget.computeSize = width => [Math.max(10, width - 20), 430];
      this._loopingDirectorEditor = new LoopingDirectorEditor(this, container);
      this.size[0] = Math.max(this.size?.[0] || 0, 760);
      this.size[1] = Math.max(this.size?.[1] || 0, 560);
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
