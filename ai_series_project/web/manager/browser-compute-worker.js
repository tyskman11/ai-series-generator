"use strict";

let running = false;
let workerId = "";
let cpuIntensity = 60;
let workerProfile = "balanced";
let completedTasks = 0;
let failedTasks = 0;

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    credentials: "same-origin",
    cache: "no-store",
    headers: { "Content-Type": "application/json", "X-Series-Web": "1" },
    body: JSON.stringify(payload),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`);
  return body;
}

function maxSampleDimension() {
  return Math.round(320 + (Math.max(10, Math.min(100, cpuIntensity)) / 100) * 704);
}

async function analyzeFrame(inputUrl) {
  const response = await fetch(inputUrl, { credentials: "same-origin", cache: "no-store" });
  if (!response.ok) throw new Error(`Frame input failed: ${response.status}`);
  const bitmap = await createImageBitmap(await response.blob());
  const maximum = maxSampleDimension();
  const scale = Math.min(1, maximum / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!context) throw new Error("OffscreenCanvas 2D is unavailable.");
  context.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const pixels = context.getImageData(0, 0, width, height).data;
  let sum = 0;
  let sumSquares = 0;
  let dark = 0;
  let bright = 0;
  let edges = 0;
  let edgeComparisons = 0;
  const previousRow = new Float32Array(width);
  for (let y = 0; y < height; y += 1) {
    let previous = 0;
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4;
      const luma = (0.2126 * pixels[offset] + 0.7152 * pixels[offset + 1] + 0.0722 * pixels[offset + 2]) / 255;
      sum += luma;
      sumSquares += luma * luma;
      if (luma < 0.04) dark += 1;
      if (luma > 0.96) bright += 1;
      if (x > 0) { edges += Math.abs(luma - previous); edgeComparisons += 1; }
      if (y > 0) { edges += Math.abs(luma - previousRow[x]); edgeComparisons += 1; }
      previous = luma;
      previousRow[x] = luma;
    }
  }
  const count = Math.max(1, width * height);
  const mean = sum / count;
  return {
    width: Math.max(1, Math.round(width / scale)),
    height: Math.max(1, Math.round(height / scale)),
    sample_width: width,
    sample_height: height,
    mean_luma: mean,
    luma_stddev: Math.sqrt(Math.max(0, sumSquares / count - mean * mean)),
    edge_score: edges / Math.max(1, edgeComparisons),
    dark_ratio: dark / count,
    bright_ratio: bright / count,
  };
}

async function registerWorker() {
  const response = await postJson("/api/browser-worker/register", {
    profile: workerProfile,
    cpu_intensity: cpuIntensity,
    hardware_concurrency: navigator.hardwareConcurrency || 0,
    webgpu_available: Boolean(navigator.gpu),
  });
  workerId = response.worker_id;
  self.postMessage({ type: "registered", worker_id: workerId, status: response.status });
}

async function workLoop() {
  while (running) {
    try {
      if (!workerId) await registerWorker();
      const claimed = await postJson("/api/browser-worker/claim", { worker_id: workerId });
      const task = claimed.task;
      if (!task) {
        await postJson("/api/browser-worker/heartbeat", { worker_id: workerId, cpu_intensity: cpuIntensity });
        self.postMessage({ type: "idle", status: claimed.status || null, completed: completedTasks, failed: failedTasks });
        await delay(Math.max(1000, Number(claimed.retry_after_seconds || 3) * 1000));
        continue;
      }
      self.postMessage({ type: "task", phase: "analyzing", task_id: task.task_id, completed: completedTasks, failed: failedTasks });
      try {
        const result = await analyzeFrame(task.input_url);
        const saved = await postJson("/api/browser-worker/result", {
          worker_id: workerId,
          task_id: task.task_id,
          status: "success",
          result,
        });
        completedTasks += 1;
        self.postMessage({ type: "result", metric: saved.metric, status: saved.status, completed: completedTasks, failed: failedTasks });
      } catch (error) {
        failedTasks += 1;
        await postJson("/api/browser-worker/result", {
          worker_id: workerId,
          task_id: task.task_id,
          status: "failed",
          error: String(error?.message || error),
        }).catch(() => {});
        self.postMessage({ type: "task-error", error: String(error?.message || error), completed: completedTasks, failed: failedTasks });
      }
      await delay(Math.round((100 - cpuIntensity) * 18));
    } catch (error) {
      self.postMessage({ type: "error", error: String(error?.message || error), completed: completedTasks, failed: failedTasks });
      workerId = "";
      await delay(5000);
    }
  }
}

self.addEventListener("message", async (event) => {
  const message = event.data || {};
  if (message.type === "start" && !running) {
    cpuIntensity = Math.max(10, Math.min(100, Number(message.cpu_intensity) || 60));
    workerProfile = String(message.profile || "balanced");
    running = true;
    workLoop();
    return;
  }
  if (message.type === "configure") {
    cpuIntensity = Math.max(10, Math.min(100, Number(message.cpu_intensity) || cpuIntensity));
    workerProfile = String(message.profile || workerProfile);
    return;
  }
  if (message.type === "stop") {
    running = false;
    if (workerId) {
      await postJson("/api/browser-worker/unregister", { worker_id: workerId }).catch(() => {});
    }
    self.postMessage({ type: "stopped" });
    self.close();
  }
});
