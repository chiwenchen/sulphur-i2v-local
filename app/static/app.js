"use strict";

const el = (id) => document.getElementById(id);

const dropzone = el("dropzone");
const imageInput = el("image-input");
const preview = el("preview");
const promptInput = el("prompt");
const generateBtn = el("generate");
const progressMsg = el("progress-msg");
const progressWrap = el("progress-wrap");
const progressBar = el("progress-bar");
const resultCard = el("result");
const videoEl = el("video");
const downloadLink = el("download");
const resetBtn = el("reset");
const backendStatus = el("backend-status");

let pickedFile = null;

// ---------- backend health ----------
async function pollHealth() {
  try {
    const r = await fetch("/api/health");
    const data = await r.json();
    if (data.comfyui_reachable) {
      backendStatus.textContent = "backend OK (ComfyUI reachable)";
      backendStatus.className = "status ok";
    } else {
      backendStatus.textContent = "ComfyUI not reachable — start it on :8188";
      backendStatus.className = "status error";
    }
  } catch (e) {
    backendStatus.textContent = "backend offline";
    backendStatus.className = "status error";
  }
}
pollHealth();
setInterval(pollHealth, 10000);

// ---------- file picker / drag-drop ----------
function setPickedFile(file) {
  pickedFile = file;
  if (!file) {
    preview.hidden = true;
    preview.removeAttribute("src");
    dropzone.querySelector(".dropzone-empty").style.display = "";
    generateBtn.disabled = true;
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
    preview.hidden = false;
    dropzone.querySelector(".dropzone-empty").style.display = "none";
  };
  reader.readAsDataURL(file);
  generateBtn.disabled = false;
}

dropzone.addEventListener("click", () => imageInput.click());
imageInput.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (f) setPickedFile(f);
});

["dragover", "dragenter"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });
});
["dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  });
});
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f && f.type.startsWith("image/")) setPickedFile(f);
});

// ---------- generation flow ----------
function setBusy(busy) {
  for (const id of [
    "image-input", "prompt", "negative-prompt", "seed", "width", "height",
    "length", "steps", "guidance", "frame-rate", "num-chains", "lora-strength",
  ]) {
    const e = document.getElementById(id);
    if (e) e.disabled = busy;
  }
  generateBtn.disabled = busy || !pickedFile;
  dropzone.style.pointerEvents = busy ? "none" : "";
  dropzone.style.opacity = busy ? "0.6" : "";
}

async function submitJob() {
  if (!pickedFile) return;
  if (!promptInput.value.trim()) {
    progressMsg.textContent = "prompt is required";
    progressMsg.className = "status error";
    return;
  }
  setBusy(true);
  resultCard.hidden = true;
  progressWrap.hidden = false;
  progressBar.value = 0;
  progressMsg.className = "status";
  progressMsg.textContent = "Submitting…";

  const fd = new FormData();
  fd.append("image", pickedFile);
  fd.append("prompt", promptInput.value);
  fd.append("negative_prompt", el("negative-prompt").value || "");
  fd.append("seed", el("seed").value);
  fd.append("width", el("width").value);
  fd.append("height", el("height").value);
  fd.append("length", el("length").value);
  fd.append("steps", el("steps").value);
  fd.append("guidance", el("guidance").value);
  fd.append("frame_rate", el("frame-rate").value);
  fd.append("num_chains", el("num-chains").value);
  fd.append("lora_strength", el("lora-strength").value);

  let jobId;
  try {
    const r = await fetch("/api/generate", { method: "POST", body: fd });
    if (r.status === 409) {
      progressMsg.textContent = "another generation is already in flight";
      progressMsg.className = "status error";
      setBusy(false);
      return;
    }
    if (!r.ok) {
      const body = await r.text();
      throw new Error(`HTTP ${r.status}: ${body}`);
    }
    const data = await r.json();
    jobId = data.job_id;
  } catch (e) {
    progressMsg.textContent = `failed: ${e.message}`;
    progressMsg.className = "status error";
    setBusy(false);
    return;
  }

  progressMsg.textContent = "Generating… 預估 5–15 分鐘，請勿關閉分頁";
  pollJob(jobId);
}

async function pollJob(jobId) {
  let backoff = 3000;
  const start = Date.now();
  while (true) {
    await new Promise((r) => setTimeout(r, backoff));
    let data;
    try {
      const r = await fetch(`/api/jobs/${jobId}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      data = await r.json();
    } catch (e) {
      progressMsg.textContent = `polling failed: ${e.message}`;
      progressMsg.className = "status error";
      setBusy(false);
      return;
    }

    progressBar.value = Math.round((data.progress || 0) * 100);
    const elapsedSec = Math.round((Date.now() - start) / 1000);
    progressMsg.textContent =
      `Generating… ${data.status} · ${progressBar.value}% · elapsed ${elapsedSec}s`;

    if (data.status === "done") {
      progressMsg.textContent = `Done in ${elapsedSec}s`;
      progressMsg.className = "status ok";
      progressBar.value = 100;
      videoEl.src = data.video_url;
      downloadLink.href = data.video_url;
      downloadLink.download = `i2v-${jobId.slice(0, 8)}.mp4`;
      resultCard.hidden = false;
      videoEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
      setBusy(false);
      return;
    }
    if (data.status === "error") {
      progressMsg.textContent = `Error: ${data.error || "unknown"}`;
      progressMsg.className = "status error";
      setBusy(false);
      return;
    }
    if (elapsedSec > 300) backoff = 8000;
  }
}

generateBtn.addEventListener("click", submitJob);

resetBtn.addEventListener("click", () => {
  setPickedFile(null);
  promptInput.value = "";
  progressMsg.textContent = "";
  progressWrap.hidden = true;
  resultCard.hidden = true;
  videoEl.removeAttribute("src");
  videoEl.load();
});
