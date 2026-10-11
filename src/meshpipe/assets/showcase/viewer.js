"use strict";
const ui = Object.fromEntries(["left", "right", "avatar", "signal", "reset", "play", "timeline", "speed", "time", "status", "provenance", "comparison", "left-label"].map(id => [id, document.getElementById(id)]));
let data = null, position = 0, playing = true, previous = null;
const camera = { yaw: 0, pitch: 0, zoom: 1 };
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function validate(payload) {
  if (payload.version !== 1 || payload.scale !== 10000 || !Array.isArray(payload.frames) || !payload.frames.length || payload.frames.length > 900) throw new Error("Unsupported motion payload");
  if (!Number.isFinite(payload.duration) || payload.duration < 0 || payload.duration > 86400000) throw new Error("Invalid duration");
  if (!Array.isArray(payload.triangles) || payload.triangles.length % 3 || payload.triangles.length > 12000 || payload.triangles.some(value => !Number.isInteger(value) || value < 0 || value >= 468)) throw new Error("Invalid topology");
  let last = -1;
  for (const frame of payload.frames) {
    if (!Number.isFinite(frame.time) || frame.time <= last || frame.time > payload.duration) throw new Error("Invalid frame time");
    last = frame.time;
    for (const name of ["raw", "smooth"]) {
      const points = frame[name];
      if (points !== null && (!Array.isArray(points) || points.length !== 1404 || points.some(value => !Number.isFinite(value) || Math.abs(value) > 10000000))) throw new Error("Invalid landmark coordinates");
    }
    if (!Array.isArray(frame.scores) || frame.scores.length !== 5 || frame.scores.some(value => !Number.isFinite(value) || value < 0 || value > 1)) throw new Error("Invalid expressions");
    if (frame.rotation !== null && (!Array.isArray(frame.rotation) || frame.rotation.length !== 9 || frame.rotation.some(value => !Number.isFinite(value) || Math.abs(value) > 1.01))) throw new Error("Invalid head rotation");
  }
  return payload;
}

function transform(point) {
  const cy = Math.cos(camera.yaw), sy = Math.sin(camera.yaw), cp = Math.cos(camera.pitch), sp = Math.sin(camera.pitch);
  const x = point[0] * cy + point[2] * sy, depth = -point[0] * sy + point[2] * cy;
  return [x, point[1] * cp - depth * sp, point[1] * sp + depth * cp];
}

function render(canvas, frame, signal) {
  const ratio = Math.min(devicePixelRatio || 1, 2), width = canvas.clientWidth, height = canvas.clientHeight;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) { canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const scale = Math.min(width, height) * .19 * camera.zoom;
  const project = point => { const moved = transform(point); const perspective = 7 / Math.max(2, 7 - moved[2]); return [width / 2 + moved[0] * scale * perspective, height / 2 - moved[1] * scale * perspective, moved[2]]; };
  context.strokeStyle = "#4f828329"; context.lineWidth = 1;
  for (let value = -4; value <= 4; value++) { const first = project([value, -2, -4]), second = project([value, -2, 4]); context.beginPath(); context.moveTo(first[0], first[1]); context.lineTo(second[0], second[1]); context.stroke(); }
  const points = frame[signal] || frame.smooth;
  if (!points) { context.fillStyle = "#c7dfd7"; context.font = "14px monospace"; context.textAlign = "center"; context.fillText("NO TRACKED FACE", width / 2, height / 2); return; }
  if (ui.avatar.value !== "mesh") { drawAvatar(context, frame, project); return; }
  const projected = [];
  for (let index = 0; index < points.length; index += 3) projected.push(project(points.slice(index, index + 3).map(value => value / data.scale)));
  const faces = [];
  for (let index = 0; index < data.triangles.length; index += 3) {
    const triangle = data.triangles.slice(index, index + 3).map(vertex => projected[vertex]);
    faces.push({ triangle, depth: triangle.reduce((sum, vertex) => sum + vertex[2], 0) / 3 });
  }
  faces.sort((left, right) => left.depth - right.depth);
  context.lineWidth = .6;
  for (const face of faces) {
    context.beginPath(); face.triangle.forEach((vertex, index) => index ? context.lineTo(vertex[0], vertex[1]) : context.moveTo(vertex[0], vertex[1])); context.closePath();
    const light = clamp(32 + face.depth * 10, 20, 55);
    context.fillStyle = `hsl(162 30% ${light}%)`; context.strokeStyle = signal === "raw" ? "#f2ba5d88" : "#a0efd080"; context.fill(); context.stroke();
  }
}

function drawAvatar(context, frame, project) {
  const matrix = frame.rotation || [1,0,0,0,1,0,0,0,1];
  const rotated = point => project([0,1,2].map(row => matrix[row * 3] * point[0] + matrix[row * 3 + 1] * point[1] + matrix[row * 3 + 2] * point[2]));
  const robot = ui.avatar.value === "robot", center = rotated([0,0,0]), edge = rotated([1.4,0,0]), radius = Math.max(12, Math.hypot(edge[0] - center[0], edge[1] - center[1]));
  const gradient = context.createRadialGradient(center[0] - radius / 3, center[1] - radius / 3, radius / 10, center[0], center[1], radius);
  gradient.addColorStop(0, robot ? "#c8efdf" : "#ffe394"); gradient.addColorStop(1, robot ? "#3b9385" : "#d08b32");
  context.fillStyle = gradient; context.beginPath();
  if (robot) context.roundRect(center[0] - radius, center[1] - radius, radius * 2, radius * 2, radius * .12); else context.arc(center[0], center[1], radius, 0, Math.PI * 2);
  context.fill();
  const shape = (point, wide, tall, color) => { const pixel = rotated(point); context.fillStyle = color; context.beginPath(); context.ellipse(pixel[0], pixel[1], radius * wide, radius * tall, 0, 0, 2 * Math.PI); context.fill(); };
  shape([-.48,.34,1.2], .10, Math.max(.015,.15 * (1 - frame.scores[4])), "#193d3d");
  shape([.48,.34,1.2], .10, Math.max(.015,.15 * (1 - frame.scores[3])), "#193d3d");
  shape([0,-.45,1.25], .23 + (frame.scores[1] + frame.scores[2]) * .06, .025 + frame.scores[0] * .22, "#193d3d");
}

for (const canvas of [ui.left, ui.right]) {
  let pointer = null;
  canvas.addEventListener("pointerdown", event => { pointer = [event.clientX, event.clientY]; canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", event => { if (pointer) { camera.yaw += (event.clientX - pointer[0]) * .008; camera.pitch = clamp(camera.pitch + (event.clientY - pointer[1]) * .008, -1.3, 1.3); pointer = [event.clientX, event.clientY]; } });
  for (const name of ["pointerup", "pointercancel", "lostpointercapture"]) canvas.addEventListener(name, () => { pointer = null; });
  canvas.addEventListener("wheel", event => { event.preventDefault(); camera.zoom = clamp(camera.zoom * Math.exp(-event.deltaY * .001), .4, 2.5); }, { passive: false });
  canvas.addEventListener("keydown", event => { if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "-"].includes(event.key)) return; event.preventDefault(); if (event.key === "ArrowLeft") camera.yaw -= .1; if (event.key === "ArrowRight") camera.yaw += .1; if (event.key === "ArrowUp") camera.pitch = clamp(camera.pitch - .1, -1.3, 1.3); if (event.key === "ArrowDown") camera.pitch = clamp(camera.pitch + .1, -1.3, 1.3); if (event.key === "+") camera.zoom = clamp(camera.zoom * 1.1, .4, 2.5); if (event.key === "-") camera.zoom = clamp(camera.zoom / 1.1, .4, 2.5); });
}
ui.reset.addEventListener("click", () => Object.assign(camera, { yaw: 0, pitch: 0, zoom: 1 }));
ui.play.addEventListener("click", () => { playing = !playing; ui.play.textContent = playing ? "Pause" : "Play"; ui.play.setAttribute("aria-label", playing ? "Pause playback" : "Play playback"); });
ui.timeline.addEventListener("input", () => { position = Number(ui.timeline.value); });
ui.signal.addEventListener("change", () => { ui.comparison.hidden = ui.signal.value !== "compare"; ui["left-label"].textContent = ui.signal.value === "smooth" ? "KALMAN SIGNAL" : "RAW SIGNAL"; });

function tick(clock) {
  if (data) {
    if (playing && previous !== null && data.duration) position = (position + Math.min(100, clock - previous) * Number(ui.speed.value)) % data.duration;
    let low = 0, high = data.frames.length - 1;
    while (low < high) { const middle = Math.ceil((low + high) / 2); if (data.frames[middle].time <= position) low = middle; else high = middle - 1; }
    const frame = data.frames[low];
    render(ui.left, frame, ui.signal.value === "smooth" ? "smooth" : "raw");
    if (!ui.comparison.hidden) render(ui.right, frame, "smooth");
    ui.timeline.value = position; ui.time.textContent = `${(position / 1000).toFixed(2)} s`;
  }
  previous = clock; requestAnimationFrame(tick);
}

async function start() {
  try {
    const response = await fetch("performance.json");
    if (!response.ok) throw new Error(`Motion file unavailable (${response.status})`);
    const text = await response.text();
    if (text.length > 16 * 1024 * 1024) throw new Error("Motion file too large");
    data = validate(JSON.parse(text));
    ui.timeline.max = Math.max(1, data.duration); ui.timeline.disabled = false; ui.play.disabled = false;
    ui.provenance.textContent = data.source_kind === "tracked" ? "RECORDED FACE TRACKING / SOURCE IMAGES OMITTED" : data.source_kind === "synthetic" ? "SYNTHETIC DEMO / NOT CAMERA INFERENCE" : "RECORDED MOTION / SOURCE PROVENANCE UNKNOWN";
    ui.status.textContent = `${data.frames.length} saved frames / shared orbit / all playback stays in your browser`;
  } catch (error) { ui.status.textContent = `${error.message}. Serve this directory over HTTP rather than opening a file URL.`; }
  requestAnimationFrame(tick);
}
start();
