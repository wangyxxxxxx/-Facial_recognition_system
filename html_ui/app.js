const API_PREFIX = "/api";
const TOPK = "5";
const ADMIN_TOKEN_KEY = "arcface_admin_token";
const ADMIN_INFO_KEY = "arcface_admin_info";

let mainFile = null;
let enrollFile = null;
let adminToken = sessionStorage.getItem(ADMIN_TOKEN_KEY) || "";
let cameraStream = null;
let cameraTarget = "main";

const $ = (id) => document.getElementById(id);
const pageName = document.body.dataset.page || "main";

function bind(id, eventName, handler) {
  const el = $(id);
  if (el) el.addEventListener(eventName, handler);
}

function showToast(msg, ms = 2600) {
  const el = $("toast");
  if (!el) return;
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), ms);
}

function setBusy(isBusy) {
  [
    "predictBtn",
    "watermarkBtn",
    "bothBtn",
    "mainCameraBtn",
    "saveModeBtn",
    "enrollBtn",
    "deleteBtn",
    "enrollCameraBtn",
    "logoutAdminBtn",
    "adminLoginBtn",
    "adminLoginCameraBtn"
  ].forEach((id) => {
    const el = $(id);
    if (el) el.disabled = isBusy;
  });
}

function setRaw(obj) {
  const el = $("rawResult");
  if (!el) return;
  el.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
}

function fileToPreview(file, previewId, nameId, prefix) {
  if (!file) return;

  const preview = $(previewId);
  const name = $(nameId);
  if (!preview || !name) return;

  const url = URL.createObjectURL(file);
  preview.innerHTML = "";

  const img = document.createElement("img");
  img.src = url;
  img.onload = () => URL.revokeObjectURL(url);

  preview.appendChild(img);
  name.textContent = `${prefix}：${file.name}`;
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_PREFIX}/health`);
    const data = await res.json();

    if (data.status === "ok") {
      if ($("apiDot")) $("apiDot").className = "dot ok";
      if ($("apiStatus")) $("apiStatus").textContent = "已连接";
    } else {
      if ($("apiDot")) $("apiDot").className = "dot bad";
      if ($("apiStatus")) $("apiStatus").textContent = "异常";
    }
  } catch (e) {
    if ($("apiDot")) $("apiDot").className = "dot bad";
    if ($("apiStatus")) $("apiStatus").textContent = "未连接";
  }
}

async function postForm(endpoint, formData) {
  const res = await fetch(`${API_PREFIX}${endpoint}`, {
    method: "POST",
    body: formData,
  });

  let data;

  try {
    data = await res.json();
  } catch (e) {
    data = { success: false, error: await res.text() };
  }

  if (!res.ok || data.success === false) {
    throw new Error(data.error || `请求失败：HTTP ${res.status}`);
  }

  return data;
}

async function getJson(endpoint) {
  const res = await fetch(`${API_PREFIX}${endpoint}`);

  let data;

  try {
    data = await res.json();
  } catch (e) {
    data = { success: false, error: await res.text() };
  }

  if (!res.ok || data.success === false) {
    throw new Error(data.error || `请求失败：HTTP ${res.status}`);
  }

  return data;
}

function requireMainFile() {
  if (!mainFile) {
    showToast("请先选择或采集一张图片");
    return false;
  }

  return true;
}

function requireAdmin() {
  if (!adminToken) {
    showToast("请先进入管理员登录页面完成人脸验证");
    return false;
  }

  return true;
}

function setAdminAuthorizedUI(isAuthorized, message = "") {
  const panel = $("adminPanel");
  const tag = $("adminTag");
  const notice = $("adminNotice");
  const sessionStatus = $("adminSessionStatus");
  const permissionMini = $("adminPermissionMini");

  if (panel) panel.classList.toggle("locked", !isAuthorized);
  if (tag) tag.textContent = isAuthorized ? "已授权" : "未授权";

  if (notice) {
    notice.textContent =
      message ||
      (isAuthorized
        ? "管理员授权有效，可以进行操作。"
        : "管理员授权无效，请返回管理员登录页面重新验证。");
  }

  if (sessionStatus) sessionStatus.textContent = isAuthorized ? "AUTHORIZED" : "LOCKED";
  if (permissionMini) permissionMini.textContent = isAuthorized ? "Granted" : "Denied";
}

function saveAdminSession(data) {
  const token = data.admin_token || data.token || "";
  if (!token) return false;

  adminToken = token;

  sessionStorage.setItem(ADMIN_TOKEN_KEY, token);
  sessionStorage.setItem(
    ADMIN_INFO_KEY,
    JSON.stringify({
      admin_label: data.admin_label || "",
      token_expires_at: data.token_expires_at || "",
      score: data.score || 0,
      saved_at: new Date().toISOString(),
    })
  );

  return true;
}

function clearAdminSession() {
  adminToken = "";
  sessionStorage.removeItem(ADMIN_TOKEN_KEY);
  sessionStorage.removeItem(ADMIN_INFO_KEY);
}

function getRuntimeApiThreshold(data) {
  return Number(data?.runtime_config?.api_threshold ?? data?.api_threshold ?? 0.30);
}

function fillRuntimeConfig(data) {
  if (!data) return;

  const mode = data.gallery_mode || "protected";

  if ($("systemGalleryMode")) {
    $("systemGalleryMode").value = mode;
  }

  if ($("apiThreshold")) {
    $("apiThreshold").value = data.api_threshold ?? "0.30";
  }

  if ($("watermarkThreshold")) {
    $("watermarkThreshold").value = data.watermark_threshold ?? "0.085";
  }

  if ($("adminRuntimeMini")) {
    const apiTh = data.api_threshold ?? "0.30";
    const wmTh = data.watermark_threshold ?? "0.085";
    $("adminRuntimeMini").textContent = `${mode} · ${apiTh}/${wmTh}`;
  }
}

async function loadRuntimeConfig() {
  if (!$("systemGalleryMode")) return;

  try {
    const data = await getJson("/admin_runtime_config");
    fillRuntimeConfig(data);
    setRaw(data);
  } catch (e) {
    showToast(e.message);
    setRaw(String(e.stack || e));
  }
}

function initAdminPage() {
  if (pageName !== "admin") return;

  if (!adminToken) {
    setAdminAuthorizedUI(false, "管理员授权不存在，请返回管理员登录页面重新验证。");
    setRaw("管理员授权不存在，请返回 admin_login.html 重新验证。");
    return;
  }

  let infoText = "管理员授权有效，可以进行录入、删除和系统模式设置。";

  try {
    const info = JSON.parse(sessionStorage.getItem(ADMIN_INFO_KEY) || "{}");
    if (info.token_expires_at) {
      infoText = `管理员授权有效，token 过期时间：${info.token_expires_at}`;
    }
  } catch (e) {
    // ignore
  }

  setAdminAuthorizedUI(true, infoText);
  loadRuntimeConfig();
}

async function callPredict() {
  if (!requireMainFile()) return;

  setBusy(true);

  try {
    const fd = new FormData();
    fd.append("image", mainFile);
    fd.append("gallery_mode", "auto");
    fd.append("topk", TOPK);

    const data = await postForm("/predict", fd);

    const score = Number(data.top1_cosine ?? 0);
    const threshold = getRuntimeApiThreshold(data);

    // 不超过识别阈值时显示为未检测到
    const passed = score > threshold;
    const displayLabel = passed ? (data.pred_label ?? "未知") : "未检测到";

    if ($("identityResult")) {
      $("identityResult").textContent = displayLabel;
    }

    data.display_result = displayLabel;
    data.recognition_threshold = threshold;
    data.recognition_passed = passed;

    setRaw(data);

    if (passed) {
      showToast(`身份识别完成：${data.pred_label}`);
    } else {
      showToast(`未检测到有效身份，score=${score.toFixed(4)}，threshold=${threshold.toFixed(4)}`);
    }
  } catch (e) {
    showToast(e.message);
    setRaw(String(e.stack || e));
  } finally {
    setBusy(false);
  }
}

async function callWatermark() {
  if (!requireMainFile()) return;

  setBusy(true);

  try {
    const fd = new FormData();
    fd.append("image", mainFile);
    fd.append("gallery_mode", "auto");
    fd.append("topk", TOPK);

    const data = await postForm("/both", fd);

    const pred = data.predict || {};
    const wm = data.watermark || {};

    const score = Number(pred.top1_cosine ?? 0);
    const threshold = getRuntimeApiThreshold(data);

    // 不超过识别阈值时显示为未检测到
    const passed = score > threshold;
    const wmDetected = Boolean(wm.detected);

    if ($("identityResult")) {
      $("identityResult").textContent = passed ? (pred.pred_label ?? "未知") : "未检测到";
    }

    if ($("watermarkResult")) {
      if (!passed) {
        $("watermarkResult").textContent = "未检测到";
      } else {
        $("watermarkResult").textContent = wmDetected ? "检测到水印" : "未检测到水印";
      }
    }

    if ($("wmBox")) {
      // 检测到水印：红色；未检测到水印：绿色
      if (!passed) {
        $("wmBox").className = "result-box green";
      } else {
        $("wmBox").className = `result-box ${wmDetected ? "red" : "green"}`;
      }
    }

    data.display_identity_result = passed ? (pred.pred_label ?? "未知") : "未检测到";
    data.display_watermark_result = !passed ? "未检测到" : (wmDetected ? "检测到水印" : "未检测到水印");
    data.recognition_threshold = threshold;
    data.recognition_passed = passed;

    setRaw(data);

    if (passed) {
      showToast("身份识别与水印检测完成");
    } else {
      showToast(`未检测到有效身份，score=${score.toFixed(4)}，threshold=${threshold.toFixed(4)}`);
    }
  } catch (e) {
    showToast(e.message);
    setRaw(String(e.stack || e));
  } finally {
    setBusy(false);
  }
}

async function callAdminVerify() {
  if (!requireMainFile()) return;

  setBusy(true);

  try {
    const fd = new FormData();
    fd.append("image", mainFile);

    const data = await postForm("/admin_verify", fd);

    if (!data.verified) {
      clearAdminSession();

      if ($("adminLoginResult")) {
        $("adminLoginResult").textContent = "验证失败";
      }

      setRaw(data);
      showToast(`管理员验证失败，score=${Number(data.score || 0).toFixed(4)}`);
      return;
    }

    if (!saveAdminSession(data)) {
      setRaw(data);
      showToast("管理员验证通过，但后端没有返回 admin_token");
      return;
    }

    if ($("adminLoginResult")) {
      $("adminLoginResult").textContent = "验证通过";
    }

    setRaw(data);
    showToast("管理员验证通过，正在进入管理员页面...");

    setTimeout(() => {
      window.location.href = "admin.html";
    }, 500);
  } catch (e) {
    showToast(e.message);
    setRaw(String(e.stack || e));
  } finally {
    setBusy(false);
  }
}

async function saveRuntimeConfig() {
  if (!requireAdmin()) return;

  const systemGalleryMode = $("systemGalleryMode")?.value || "protected";
  const apiThreshold = $("apiThreshold")?.value || "0.30";
  const watermarkThreshold = $("watermarkThreshold")?.value || "0.085";

  if (Number.isNaN(Number(apiThreshold)) || Number(apiThreshold) < -1 || Number(apiThreshold) > 1) {
    showToast("API 验证阈值必须是 -1 到 1 之间的数字");
    return;
  }

  if (
    Number.isNaN(Number(watermarkThreshold)) ||
    Number(watermarkThreshold) < -1 ||
    Number(watermarkThreshold) > 1
  ) {
    showToast("水印检测阈值必须是 -1 到 1 之间的数字");
    return;
  }

  setBusy(true);

  try {
    const fd = new FormData();
    fd.append("admin_token", adminToken);
    fd.append("gallery_mode", systemGalleryMode);
    fd.append("api_threshold", apiThreshold);
    fd.append("watermark_threshold", watermarkThreshold);

    const data = await postForm("/admin_runtime_config", fd);

    fillRuntimeConfig(data);
    setRaw(data);
    showToast("系统图库模式和阈值设置已保存");
  } catch (e) {
    showToast(e.message);
    setRaw(String(e.stack || e));
  } finally {
    setBusy(false);
  }
}

async function enrollFace() {
  if (!requireAdmin()) return;

  const label = $("enrollLabel").value.trim();

  if (!label) {
    showToast("请输入要录入的身份编号 label");
    return;
  }

  if (!enrollFile) {
    showToast("请选择或采集录入图片");
    return;
  }

  setBusy(true);

  try {
    const fd = new FormData();
    fd.append("admin_token", adminToken);
    fd.append("label", label);
    fd.append("overwrite", $("overwriteEnroll").checked ? "true" : "false");
    fd.append("image", enrollFile);

    const data = await postForm("/enroll_face", fd);

    setRaw(data);
    showToast(`录入完成：${label}`);
  } catch (e) {
    showToast(e.message);
    setRaw(String(e.stack || e));
  } finally {
    setBusy(false);
  }
}

async function deleteFace() {
  if (!requireAdmin()) return;

  const label = $("deleteLabel").value.trim();

  if (!label) {
    showToast("请输入要删除的身份编号 label");
    return;
  }

  if (!confirm(`确认删除身份 ${label} 吗？此操作会修改 clean gallery 并自动重建 protected gallery。`)) {
    return;
  }

  setBusy(true);

  try {
    const fd = new FormData();
    fd.append("admin_token", adminToken);
    fd.append("label", label);

    const data = await postForm("/delete_face", fd);

    setRaw(data);
    showToast(`删除完成：${label}`);
  } catch (e) {
    showToast(e.message);
    setRaw(String(e.stack || e));
  } finally {
    setBusy(false);
  }
}

function logoutAdmin() {
  clearAdminSession();
  setAdminAuthorizedUI(false, "已退出管理员授权，请返回管理员登录页面重新验证。");
  setRaw("已退出管理员授权。");
  showToast("已退出管理员");
}

async function openCamera(target) {
  cameraTarget = target;

  if ($("cameraTitle")) {
    if (target === "main") {
      $("cameraTitle").textContent = "摄像头采集识别图片";
    } else if (target === "enroll") {
      $("cameraTitle").textContent = "摄像头采集录入图片";
    } else if (target === "adminLogin") {
      $("cameraTitle").textContent = "摄像头采集管理员人脸";
    } else {
      $("cameraTitle").textContent = "摄像头采集";
    }
  }

  if ($("cameraModal")) {
    $("cameraModal").classList.remove("hidden");
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    if ($("cameraVideo")) {
      $("cameraVideo").srcObject = cameraStream;
    }
  } catch (e) {
    closeCamera();
    showToast("无法打开摄像头，请检查浏览器权限或设备占用情况");
  }
}

function closeCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((t) => t.stop());
    cameraStream = null;
  }

  if ($("cameraVideo")) {
    $("cameraVideo").srcObject = null;
  }

  if ($("cameraModal")) {
    $("cameraModal").classList.add("hidden");
  }
}

function captureCurrentFrame() {
  const video = $("cameraVideo");

  if (!video || !video.videoWidth || !video.videoHeight) {
    showToast("摄像头画面还未准备好");
    return;
  }

  const canvas = $("cameraCanvas");
  const srcW = video.videoWidth;
  const srcH = video.videoHeight;

  const side = Math.min(srcW, srcH);
  const sx = Math.floor((srcW - side) / 2);
  const sy = Math.floor((srcH - side) / 2);

  canvas.width = 512;
  canvas.height = 512;

  const ctx = canvas.getContext("2d");

  // 摄像头采集图像左右镜像保存，保证提交图片与预览方向一致。
  ctx.save();
  ctx.translate(512, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, sx, sy, side, side, 0, 0, 512, 512);
  ctx.restore();

  canvas.toBlob(
    (blob) => {
      if (!blob) {
        showToast("图片采集失败");
        return;
      }

      const file = new File([blob], `camera_${Date.now()}.jpg`, {
        type: "image/jpeg",
      });

      if (cameraTarget === "main") {
        mainFile = file;
        fileToPreview(mainFile, "mainPreview", "mainFileName", "图片来源");

        if ($("identityResult")) $("identityResult").textContent = "未检测";
        if ($("watermarkResult")) $("watermarkResult").textContent = "未检测";
        if ($("wmBox")) $("wmBox").className = "result-box gray";
      } else if (cameraTarget === "enroll") {
        enrollFile = file;
        fileToPreview(enrollFile, "enrollPreview", "enrollFileName", "录入图片");
      } else if (cameraTarget === "adminLogin") {
        mainFile = file;
        fileToPreview(mainFile, "adminLoginPreview", "adminLoginFileName", "图片来源");

        if ($("adminLoginResult")) {
          $("adminLoginResult").textContent = "未验证";
        }
      }

      closeCamera();
      showToast("摄像头采集成功");
    },
    "image/jpeg",
    0.95
  );
}

function initEvents() {
  bind("mainFileInput", "change", (e) => {
    mainFile = e.target.files[0] || null;

    if (mainFile) {
      fileToPreview(mainFile, "mainPreview", "mainFileName", "图片来源");

      if ($("identityResult")) $("identityResult").textContent = "未检测";
      if ($("watermarkResult")) $("watermarkResult").textContent = "未检测";
      if ($("wmBox")) $("wmBox").className = "result-box gray";
    }
  });

  bind("adminLoginFileInput", "change", (e) => {
    mainFile = e.target.files[0] || null;

    if (mainFile) {
      fileToPreview(mainFile, "adminLoginPreview", "adminLoginFileName", "图片来源");

      if ($("adminLoginResult")) {
        $("adminLoginResult").textContent = "未验证";
      }
    }
  });

  bind("enrollFileInput", "change", (e) => {
    enrollFile = e.target.files[0] || null;

    if (enrollFile) {
      fileToPreview(enrollFile, "enrollPreview", "enrollFileName", "录入图片");
    }
  });

  bind("predictBtn", "click", callPredict);
  bind("watermarkBtn", "click", callWatermark);
  bind("bothBtn", "click", callWatermark);

  bind("adminLoginBtn", "click", callAdminVerify);

  bind("saveModeBtn", "click", saveRuntimeConfig);
  bind("enrollBtn", "click", enrollFace);
  bind("deleteBtn", "click", deleteFace);
  bind("logoutAdminBtn", "click", logoutAdmin);

  bind("mainCameraBtn", "click", () => openCamera("main"));
  bind("adminLoginCameraBtn", "click", () => openCamera("adminLogin"));
  bind("enrollCameraBtn", "click", () => openCamera("enroll"));

  bind("captureBtn", "click", captureCurrentFrame);
  bind("cancelCameraBtn", "click", closeCamera);
  bind("closeCameraBtn", "click", closeCamera);
}

initEvents();
checkHealth();
initAdminPage();
setInterval(checkHealth, 5000);
