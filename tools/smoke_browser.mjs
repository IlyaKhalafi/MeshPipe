import { writeFile } from "node:fs/promises";

const endpoint = process.argv[2] || "http://127.0.0.1:9223";
const url = process.argv[3] || "http://127.0.0.1:8765";
const pages = await (await fetch(`${endpoint}/json/list`)).json();
const page = pages.find(item => item.type === "page");
if (!page) throw new Error("Start Chrome with remote debugging enabled");
const socket = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
let sequence = 0;
const pending = new Map(), errors = [];
socket.onmessage = event => {
  const response = JSON.parse(event.data);
  if (response.method === "Runtime.exceptionThrown") errors.push(response.params.exceptionDetails);
  if (pending.has(response.id)) {
    const { resolve, reject, timer } = pending.get(response.id);
    pending.delete(response.id); clearTimeout(timer);
    if (response.error) reject(new Error(response.error.message)); else resolve(response.result);
  }
};
function call(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timeout ${method}`)); }, 15000);
    pending.set(id, { resolve, reject, timer });
    socket.send(JSON.stringify({ id, method, params }));
  });
}
async function evaluate(expression) {
  const result = await call("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
try {
  await call("Page.enable"); await call("Runtime.enable");
  await call("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false });
  await call("Page.navigate", { url });
  await evaluate(`new Promise((resolve, reject) => { let count = 0; const timer = setInterval(() => { const play = document.getElementById("play"); if (play && !play.disabled) { clearInterval(timer); resolve(true); } else if (++count > 100) { clearInterval(timer); reject(new Error("Showcase failed to load")); } }, 50); })`);
  const provenance = await evaluate(`document.getElementById("provenance").textContent`);
  if (!provenance.includes("RECORDED FACE TRACKING")) throw new Error("Genuine tracked motion required");
  await evaluate(`document.getElementById("play").click(); document.getElementById("signal").value = "compare"; document.getElementById("signal").dispatchEvent(new Event("change")); document.getElementById("timeline").value = 1500; document.getElementById("timeline").dispatchEvent(new Event("input")); true`);
  await new Promise(resolve => setTimeout(resolve, 200));
  const comparison = await evaluate(`!document.getElementById("comparison").hidden && document.getElementById("left").width > 0 && document.getElementById("right").width > 0`);
  if (!comparison) throw new Error("Comparison view failed");
  for (const avatar of ["emoji", "robot", "mesh"]) {
    await evaluate(`document.getElementById("avatar").value = ${JSON.stringify(avatar)}; true`);
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  await new Promise(resolve => setTimeout(resolve, 1200));
  const visible = await evaluate(`Number(getComputedStyle(document.querySelector(".studio")).opacity) > .95`);
  if (!visible) throw new Error("Studio reveal did not complete");
  const desktop = await call("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  await writeFile("/tmp/meshpipe-browser-desktop.png", Buffer.from(desktop.data, "base64"));
  await call("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await new Promise(resolve => setTimeout(resolve, 100));
  const overflow = await evaluate(`document.documentElement.scrollWidth > innerWidth`);
  if (overflow) throw new Error("Mobile layout overflows");
  const mobile = await call("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  await writeFile("/tmp/meshpipe-browser-mobile.png", Buffer.from(mobile.data, "base64"));
  if (errors.length) throw new Error(JSON.stringify(errors));
  console.log("Genuine playback, comparison, avatar switches, desktop and mobile passed");
} finally { socket.close(); }
