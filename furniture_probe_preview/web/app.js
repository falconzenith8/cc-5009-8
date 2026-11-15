"use strict";

const canvas = document.getElementById("plan");
const ctx = canvas.getContext("2d");
const CELL_PX = 10;

let scene = null;
let fieldData = null;
let leakData = null;
let selected = null;          // {kind:"probe"|"furniture", id}
let hoverPoint = null;
let drag = null;
let regressionData = null;    // kernel-driven verdict from /api/regression

const api = async (path, body) => {
  const opt = body
    ? { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) }
    : {};
  const r = await fetch(path, opt);
  const data = await r.json();
  if (data.error) throw new Error(data.error);
  return data;
};

const scale = () => ({
  sx: canvas.width / scene.bounds.w,
  sy: canvas.height / scene.bounds.h,
});
const toPx = (x, y) => {
  const { sx, sy } = scale();
  return [x * sx, y * sy];
};
const toWorld = (px, py) => {
  const { sx, sy } = scale();
  return [px / sx, py / sy];
};
const rgbCss = (c, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

// ---------------------------------------------------------------------------
// Data refresh
// ---------------------------------------------------------------------------

async function loadPreset(name) {
  scene = await api("/api/load-preset", { preset: name });
  document.getElementById("preset").value = name;
  regressionData = null;
  document.getElementById("regressionOut").textContent = "尚未运行。";
  document.getElementById("regressionOut").className = "";
  await refreshComputed();
  populateFurniture();
  selected = null;
  draw();
}

async function refreshComputed() {
  [fieldData, leakData] = await Promise.all([
    api("/api/field"), api("/api/leak-report"),
  ]);
  document.getElementById("fallback").value = scene.fallback;
  renderAnomalies();
  if (selected) await showSelectionWeights();
}

function populateFurniture() {
  const sel = document.getElementById("furniturePick");
  sel.innerHTML = "";
  for (const f of scene.furniture) {
    const o = document.createElement("option");
    o.value = f.name;
    o.textContent = `${f.kind === "table" ? "桌" : "椅"} · ${f.name}`;
    sel.appendChild(o);
  }
  refreshMaterial();
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

function draw() {
  if (!scene) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawField();
  drawOverlay();
  drawWalls();
  drawProbes();
  drawFurniture();
}

function drawField() {
  const overlay = document.getElementById("overlay").value;
  const g = fieldData.grid;
  const cw = canvas.width / g.nx, ch = canvas.height / g.ny;
  const key = overlay === "reference" ? "refRgb" : "rgb";
  for (const c of fieldData.cells) {
    ctx.fillStyle = rgbCss(c[key]);
    ctx.fillRect(c.i * cw, c.j * ch, cw + 0.6, ch + 0.6);
  }
}

function drawOverlay() {
  const overlay = document.getElementById("overlay").value;
  if (overlay === "none" || !leakData) return;
  const g = leakData.grid;
  const cw = canvas.width / g.nx, ch = canvas.height / g.ny;
  ctx.fillStyle = "rgba(217,194,74,0.35)";
  for (const [i, j] of leakData.risk)
    ctx.fillRect(i * cw, j * ch, cw + 0.6, ch + 0.6);
  ctx.fillStyle = "rgba(229,72,77,0.55)";
  for (const n of leakData.approximation)
    ctx.fillRect(n.i * cw, n.j * ch, cw + 0.6, ch + 0.6);
  ctx.fillStyle = "rgba(120,120,255,0.45)";
  for (const [i, j] of leakData.uncovered)
    ctx.fillRect(i * cw, j * ch, cw + 0.6, ch + 0.6);
}

function drawWalls() {
  ctx.strokeStyle = "#c9d2e0";
  ctx.lineWidth = 3;
  ctx.beginPath();
  for (const w of scene.walls) {
    const [x0, y0] = toPx(w.ax, w.ay);
    const [x1, y1] = toPx(w.bx, w.by);
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
  }
  ctx.stroke();
}

function drawProbes() {
  const failed = new Set(
    regressionData ? regressionData.failed_probe_ids : []
  );
  for (const p of scene.probes) {
    const [x, y] = toPx(p.x, p.y);
    // coverage radius
    const { sx } = scale();
    ctx.beginPath();
    ctx.arc(x, y, p.coverage * sx, 0, Math.PI * 2);
    ctx.strokeStyle = selected && selected.id === p.id
      ? "rgba(55,224,200,0.95)" : "rgba(55,224,200,0.35)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#37e0c8";
    ctx.fill();
    // Kernel-driven failure feedback routes back to the probe itself.
    if (failed.has(p.id)) {
      ctx.beginPath();
      ctx.arc(x, y, 11, 0, Math.PI * 2);
      ctx.strokeStyle = "#ff5d5d";
      ctx.lineWidth = 3;
      ctx.stroke();
    }
    ctx.fillStyle = "#0d0f13";
    ctx.font = "bold 9px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(p.component), x, y + 0.5);
    ctx.fillStyle = failed.has(p.id) ? "#ff8a8a" : "#b9f5ea";
    ctx.font = "9px sans-serif";
    ctx.fillText(p.id + (failed.has(p.id) ? " ✕" : ""), x, y - 12);
  }
}

function drawFurniture() {
  for (const f of scene.furniture) {
    const [x, y] = toPx(f.x, f.y);
    const { sx, sy } = scale();
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(-f.rotation * Math.PI / 180);
    ctx.fillStyle = selected && selected.id === f.name ? "#ffce7a" : "#e0a13c";
    ctx.strokeStyle = "#fff1d6";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.rect(-f.w * sx / 2, -f.h * sy / 2, f.w * sx, f.h * sy);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
    ctx.fillStyle = "#fff1d6";
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(f.name, x, y + f.h * sy / 2 + 12);
  }
}

// ---------------------------------------------------------------------------
// Anomaly panel (everything comes from engine-computed data)
// ---------------------------------------------------------------------------

function renderAnomalies() {
  const el = document.getElementById("anomalies");
  const lines = [];
  lines.push(`回退策略: ${leakData.fallback}`);
  lines.push(`最大绝对误差(探针 vs 直接光参考): ${leakData.maxAbsoluteError}`);
  lines.push(`隔墙漏光风险单元(跨侧探针已拒绝): ${leakData.risk.length}`);
  lines.push(`探针粒度近似偏差单元(非漏光): ${leakData.approximation.length}`);
  lines.push(`未覆盖/回退单元: ${leakData.uncovered.length}`);
  if (leakData.risk.length)
    lines.push("\n[提示] 黄色单元隔墙另一侧有探针在覆盖半径内，已按视线拒绝；这是漏光高风险位置。");
  if (leakData.approximation.length)
    lines.push("[提示] 红色单元为探针网格对直接光场的近似偏差，可加密/移动探针改善，并非墙体漏光。");
  if (leakData.uncovered.length)
    lines.push("[警告] 蓝色单元无视线可达探针，已按策略回退（绝不会跨墙混光）。");
  if (!leakData.approximation.length && !leakData.uncovered.length)
    lines.push("\n[正常] 所有覆盖区域插值均在容差内。");
  el.textContent = lines.join("\n");
  el.className = leakData.approximation.length || leakData.uncovered.length
    ? "warn" : "ok";
}

// ---------------------------------------------------------------------------
// Probe weight panel
// ---------------------------------------------------------------------------

async function showSelectionWeights() {
  const el = document.getElementById("probeWeights");
  let world;
  if (selected.kind === "furniture") {
    const f = scene.furniture.find(x => x.name === selected.id);
    world = [f.x, f.y];
  } else {
    const p = scene.probes.find(x => x.id === selected.id);
    world = [p.x, p.y];
  }
  const cell = nearestCell(world[0], world[1]);
  if (!cell) { el.innerHTML = "<p>点在房间外。</p>"; return; }
  const rows = Object.entries(cell.weights)
    .sort((a, b) => b[1] - a[1])
    .map(([id, w]) => `<tr><td>${id}</td>
      <td style="width:120px"><div class="bar" style="width:${(w * 100).toFixed(1)}%"></div></td>
      <td>${(w * 100).toFixed(1)}%</td></tr>`).join("");
  const rejected = fieldData.probes
    .filter(p => {
      const d = Math.hypot(p.x - world[0], p.y - world[1]);
      return d <= p.coverage &&
        p.component !== cellComponentAt(world);
    })
    .map(p => `<tr><td>${p.id}</td>
      <td><div class="bar reject" style="width:100%"></div></td>
      <td>拒绝(隔墙)</td></tr>`).join("");
  el.innerHTML = `
    <p>状态: <b class="${cell.status === "covered" ? "ok" : "warn"}">${cell.status}</b>
    @ (${world[0].toFixed(2)}, ${world[1].toFixed(2)})</p>
    <table><tr><th>探针</th><th>权重(线性域IDW)</th><th></th></tr>
    ${rows || '<tr><td colspan="3">无可用探针</td></tr>'}
    ${rejected}
    </table>
    <p style="font-size:10px;color:#9aa3b2">灰色行=在覆盖半径内但位于隔墙另一侧，按连通分量排除，防止漏光。</p>`;
}

function cellComponentAt(world) {
  // Component is carried per probe; find nearest field cell's probes' comp
  const cell = nearestCell(world[0], world[1]);
  const pid = Object.keys(cell.weights)[0];
  if (!pid) return -999;
  return fieldData.probes.find(p => p.id === pid).component;
}

function nearestCell(x, y) {
  const g = fieldData.grid;
  const i = Math.min(g.nx - 1, Math.max(0, Math.floor(x / g.dx)));
  const j = Math.min(g.ny - 1, Math.max(0, Math.floor(y / g.dy)));
  return fieldData.cells.find(c => c.i === i && c.j === j);
}

// ---------------------------------------------------------------------------
// Material preview window
// ---------------------------------------------------------------------------

async function refreshMaterial() {
  const name = document.getElementById("furniturePick").value;
  if (!name) return;
  const d = await api("/api/material/" + encodeURIComponent(name));
  paintSwatch(document.getElementById("matProbe"), d.shaded_srgb,
    d.sample_status !== "covered");
  paintSwatch(document.getElementById("matRef"), d.reference_shaded_srgb, false);
  document.getElementById("matInfo").textContent =
    `材质: ${d.material}\n` +
    `位置: (${d.position[0].toFixed(2)}, ${d.position[1].toFixed(2)})\n` +
    `采样状态: ${d.sample_status}\n${d.sample_reason || ""}\n` +
    `探针辐照度(线性): ${d.probe_irradiance.map(v => v.toFixed(3)).join(", ")}\n` +
    `参考辐照度(线性): ${d.reference_irradiance.map(v => v.toFixed(3)).join(", ")}\n` +
    `权重: ${Object.entries(d.weights).map(([k, v]) => `${k}=${(v * 100).toFixed(1)}%`).join(" ") || "无"}\n` +
    `隔墙拒绝探针: ${d.rejected_probes.join(", ") || "无"}\n` +
    `回归判定(内核): ${d.regression.passed ? "通过 ✓" : "失败 ✕"}\n` +
    `主导探针: ${d.regression.dominant_probe ?? "无"}  ` +
    `照明误差: ${d.regression.lighting_error.toFixed(4)} / 预算 ${d.regression.error_budget}` +
    (d.regression.failures.length
      ? "\n[失败原因] " + d.regression.failures.join("; ")
      : "") +
    (d.leak_risk ? "\n[提示] 该点邻近隔墙，存在跨侧探针被排除的漏光风险。" : "");
  paintSwatch(document.getElementById("matProbe"), d.shaded_srgb,
    !d.regression.passed);
}

function paintSwatch(cv, srgb, warn) {
  const c = cv.getContext("2d");
  c.fillStyle = rgbCss(srgb);
  c.fillRect(0, 0, cv.width, cv.height);
  if (warn) {
    c.strokeStyle = "#e5484d";
    c.lineWidth = 4;
    c.strokeRect(2, 2, cv.width - 4, cv.height - 4);
  }
}

// ---------------------------------------------------------------------------
// Kernel-driven furniture probe regression (opened from this preview window)
// ---------------------------------------------------------------------------

async function runRegressionCheck() {
  const out = document.getElementById("regressionOut");
  try {
    regressionData = await api("/api/regression");
  } catch (err) {
    out.className = "fail";
    out.textContent = "回归检查请求失败: " + err.message;
    return;
  }
  const lines = [];
  lines.push(regressionData.passed
    ? "总体判定: 通过 ✓"
    : "总体判定: 失败 ✕");
  for (const c of regressionData.checks) {
    lines.push("");
    lines.push(`· ${c.furniture} @ (${c.position[0]}, ${c.position[1]}) ` +
      `-> ${c.passed ? "通过" : "失败"}`);
    lines.push(`  主导探针: ${c.dominant_probe ?? "无"}  状态: ${c.status}`);
    lines.push(`  权重: ${Object.entries(c.weights)
      .map(([k, v]) => `${k}=${(v * 100).toFixed(1)}%`).join(" ") || "无"}`);
    lines.push(`  探针辐照度: ${c.probe_irradiance
      .map(v => v.toFixed(3)).join(", ")}`);
    lines.push(`  参考辐照度: ${c.reference_irradiance
      .map(v => v.toFixed(3)).join(", ")}`);
    lines.push(`  照明误差: ${c.lighting_error.toFixed(4)} ` +
      `(预算 ${c.error_budget})`);
    if (c.rejected_probes.length)
      lines.push(`  隔墙拒绝探针: ${c.rejected_probes.join(", ")}`);
    for (const f of c.failures) lines.push(`  [失败原因] ${f}`);
  }
  if (regressionData.failed_probe_ids.length)
    lines.push("", "失败反馈已定位到探针: " +
      regressionData.failed_probe_ids.join(", "));
  out.className = regressionData.passed ? "ok" : "fail";
  out.textContent = lines.join("\n");
  draw();
}

// ---------------------------------------------------------------------------
// Pointer interaction: drag probes and furniture
// ---------------------------------------------------------------------------

function hitTest(px, py) {
  const [wx, wy] = toWorld(px, py);
  for (const p of scene.probes) {
    const [x, y] = toPx(p.x, p.y);
    if (Math.hypot(px - x, py - y) < 12)
      return { kind: "probe", id: p.id };
  }
  for (const f of scene.furniture) {
    const [x, y] = toPx(f.x, f.y);
    const { sx, sy } = scale();
    if (Math.abs(px - x) < f.w * sx / 2 + 6 &&
        Math.abs(py - y) < f.h * sy / 2 + 6)
      return { kind: "furniture", id: f.name };
  }
  return null;
}

canvas.addEventListener("mousedown", (e) => {
  const rect = canvas.getBoundingClientRect();
  const px = e.clientX - rect.left, py = e.clientY - rect.top;
  drag = hitTest(px, py);
  selected = drag;
  if (drag) {
    if (drag.kind === "furniture")
      document.getElementById("furniturePick").value = drag.id;
    draw();
  }
});

canvas.addEventListener("mousemove", async (e) => {
  const rect = canvas.getBoundingClientRect();
  const px = e.clientX - rect.left, py = e.clientY - rect.top;
  if (!drag) {
    canvas.style.cursor = hitTest(px, py) ? "grab" : "crosshair";
    return;
  }
  canvas.style.cursor = "grabbing";
  const [wx, wy] = toWorld(px, py);
  try {
    if (drag.kind === "probe") {
      scene = await api("/api/move-probe", { id: drag.id, x: wx, y: wy });
      await refreshComputed();
    } else {
      scene = await api("/api/move-furniture",
        { id: drag.id, x: wx, y: wy });
    }
    draw();
    await showSelectionWeights();
    if (drag.kind === "furniture") await refreshMaterial();
  } catch (err) { alert(err.message); }
});

window.addEventListener("mouseup", async () => {
  if (drag && drag.kind === "furniture") await refreshComputed();
  drag = null;
});

canvas.addEventListener("click", async (e) => {
  if (selected) { await showSelectionWeights(); }
});

// ---------------------------------------------------------------------------
// Toolbar wiring
// ---------------------------------------------------------------------------

document.getElementById("preset").addEventListener("change", e =>
  loadPreset(e.target.value));
document.getElementById("overlay").addEventListener("change", draw);
document.getElementById("fallback").addEventListener("change", async e => {
  scene = await api("/api/set-fallback", { fallback: e.target.value });
  await refreshComputed();
  draw();
});
document.getElementById("furniturePick").addEventListener("change",
  refreshMaterial);
document.getElementById("saveBtn").addEventListener("click", async () => {
  const path = document.getElementById("scenePath").value;
  try {
    const r = await api("/api/save", { path });
    alert("已保存: " + r.saved);
  } catch (e) { alert(e.message); }
});
document.getElementById("loadBtn").addEventListener("click", async () => {
  const path = document.getElementById("scenePath").value;
  try {
    scene = await api("/api/load", { path });
    await refreshComputed();
    populateFurniture();
    draw();
    alert("场景已重新打开，探针解释与物体位置已恢复。");
  } catch (e) { alert(e.message); }
});
document.getElementById("openValidation").addEventListener("click",
  () => loadPreset("validation").then(runRegressionCheck));
document.getElementById("openCorrupt").addEventListener("click",
  () => loadPreset("validation-corrupt").then(runRegressionCheck));
document.getElementById("runRegression").addEventListener("click",
  runRegressionCheck);

loadPreset("uniform");
