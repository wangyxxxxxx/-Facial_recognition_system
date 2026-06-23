const API_PREFIX = "/api";
const TOPK = "5";
const ADMIN_TOKEN_KEY = "arcface_admin_token";
const ADMIN_INFO_KEY = "arcface_admin_info";

let mainFile = null;
let enrollFile = null;
let enrollFiles = [];
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
    "clearEnrollImagesBtn",
    "logoutAdminBtn",
    "adminLoginBtn",
    "adminLoginCameraBtn",
    "loadEnrollLogsBtn",
    "loadDeleteLogsBtn",
    "loadFaceLogsBtn"
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

function renderEnrollFilesPreview() {
  const preview = $("enrollPreview");
  const name = $("enrollFileName");
  if (!preview || !name) return;

  preview.innerHTML = "";

  if (!enrollFiles.length) {
    preview.innerHTML = "<span>未选择录入图片</span>";
    name.textContent = "录入图片：未选择";
    return;
  }

  const wrap = document.createElement("div");
  wrap.style.display = "grid";
  wrap.style.gridTemplateColumns = "repeat(auto-fit, minmax(72px, 1fr))";
  wrap.style.gap = "8px";
  wrap.style.width = "100%";

  enrollFiles.slice(0, 8).forEach((file, idx) => {
    const item = document.createElement("div");
    item.style.position = "relative";
    item.style.minHeight = "72px";

    const img = document.createElement("img");
    const url = URL.createObjectURL(file);
    img.src = url;
    img.onload = () => URL.revokeObjectURL(url);
    img.style.width = "100%";
    img.style.height = "72px";
    img.style.objectFit = "cover";
    img.style.borderRadius = "10px";

    const badge = document.createElement("span");
    badge.textContent = String(idx + 1);
    badge.style.position = "absolute";
    badge.style.left = "6px";
    badge.style.top = "6px";
    badge.style.padding = "2px 6px";
    badge.style.borderRadius = "999px";
    badge.style.background = "rgba(0,0,0,0.55)";
    badge.style.color = "#fff";
    badge.style.fontSize = "12px";

    item.appendChild(img);
    item.appendChild(badge);
    wrap.appendChild(item);
  });

  preview.appendChild(wrap);

  const names = enrollFiles.map((f) => f.name).join("、");
  name.textContent = `录入图片：已选择 ${enrollFiles.length} 张${names ? `（${names}）` : ""}`;
}

function clearEnrollFiles() {
  enrollFiles = [];
  enrollFile = null;

  const input = $("enrollFileInput");
  if (input) input.value = "";

  renderEnrollFilesPreview();
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

function runtimeBool(value, defaultValue = false) {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;

  const text = String(value ?? "").trim().toLowerCase();

  if (["1", "true", "yes", "on", "enable", "enabled", "开启", "打开"].includes(text)) {
    return true;
  }

  if (["0", "false", "no", "off", "disable", "disabled", "关闭", "关"].includes(text)) {
    return false;
  }

  return Boolean(defaultValue);
}

function fillRuntimeConfig(data) {
  if (!data) return;

  const mode = data.gallery_mode || "protected";
  const defenseEnabled = runtimeBool(data.score_defense_enabled, false);

  if ($("systemGalleryMode")) {
    $("systemGalleryMode").value = mode;
  }

  if ($("apiThreshold")) {
    $("apiThreshold").value = data.api_threshold ?? "0.30";
  }

  if ($("watermarkThreshold")) {
    $("watermarkThreshold").value = data.watermark_threshold ?? "0.085";
  }

  if ($("scoreDefenseEnabled")) {
    $("scoreDefenseEnabled").value = defenseEnabled ? "true" : "false";
  }

  if ($("adminRuntimeMini")) {
    const apiTh = data.api_threshold ?? "0.30";
    const wmTh = data.watermark_threshold ?? "0.085";
    $("adminRuntimeMini").textContent = `${mode} · ${apiTh}/${wmTh} · ${defenseEnabled ? "防御开" : "防御关"}`;
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
    const faceFd = new FormData();
    faceFd.append("image", mainFile);
    faceFd.append("gallery_mode", "auto");
    faceFd.append("topk", TOPK);

    // 主页面识别按钮现在同时触发：
    // 1. /face：身份识别；
    // 2. /detect_watermark：使用 /face 预测出的身份 label 做水印检测。
    const faceData = await postForm("/face", faceFd);

    const score = Number(faceData.top1_cosine ?? 0);
    const threshold = getRuntimeApiThreshold(faceData);

    // 不超过识别阈值时显示为未检测到，但仍然会用 pred_label 尝试水印检测。
    const passed = score > threshold;
    const predLabel = faceData.pred_label ?? "";
    const displayLabel = passed ? (predLabel || "未知") : "未检测到";

    if ($("identityResult")) {
      $("identityResult").textContent = displayLabel;
    }

    let watermarkData = null;
    let watermarkError = "";

    if (!passed) {
      // 身份识别未通过阈值时，没有可靠身份 label 用于确权，水印状态显示为识别失败。
      if ($("watermarkResult")) {
        $("watermarkResult").textContent = "识别失败";
      }

      if ($("wmBox")) {
        $("wmBox").className = "result-box gray";
      }
    } else if (predLabel) {
      try {
        const wmFd = new FormData();
        wmFd.append("image", mainFile);
        wmFd.append("label", String(predLabel));
        wmFd.append("threshold", String(faceData?.runtime_config?.watermark_threshold ?? "0.085"));

        watermarkData = await postForm("/detect_watermark", wmFd);

        if ($("watermarkResult")) {
          $("watermarkResult").textContent = watermarkData.detected ? "检测到水印" : "未检测到水印";
        }

        if ($("wmBox")) {
          $("wmBox").className = watermarkData.detected ? "result-box red" : "result-box green";
        }
      } catch (wmErr) {
        watermarkError = wmErr.message || String(wmErr);

        if ($("watermarkResult")) {
          $("watermarkResult").textContent = "水印检测失败";
        }

        if ($("wmBox")) {
          $("wmBox").className = "result-box red";
        }
      }
    } else {
      if ($("watermarkResult")) {
        $("watermarkResult").textContent = "识别失败";
      }

      if ($("wmBox")) {
        $("wmBox").className = "result-box gray";
      }
    }

    faceData.display_result = displayLabel;
    faceData.recognition_threshold = threshold;
    faceData.recognition_passed = passed;

    setRaw({
      face: faceData,
      watermark: watermarkData,
      watermark_error: watermarkError,
    });

    if (passed) {
      showToast(`身份识别完成：${predLabel || "未知"}；水印检测已执行`);
    } else {
      showToast(`未检测到有效身份，score=${score.toFixed(4)}，threshold=${threshold.toFixed(4)}；水印显示识别失败`);
    }
  } catch (e) {
    showToast(e.message);
    setRaw(String(e.stack || e));
  } finally {
    setBusy(false);
  }
}

async function callWatermark() {
  // 主页面识别按钮：同时执行身份识别和水印检测。
  return callPredict();
}

async function callAdminVerify() {
  if (!requireMainFile()) return;

  const adminId = ($("adminLoginId")?.value || "").trim();

  if (!adminId) {
    showToast("请输入管理员 ID");
    return;
  }

  setBusy(true);

  try {
    // 先使用现有 /face 接口做正常身份识别，检查预测身份是否与输入管理员 ID 一致。
    const faceFd = new FormData();
    faceFd.append("image", mainFile);
    faceFd.append("gallery_mode", "auto");
    faceFd.append("topk", TOPK);

    const faceData = await postForm("/face", faceFd);
    const predLabel = String(faceData.pred_label ?? "");
    const idMatchedByFace = predLabel === adminId;

    if (!idMatchedByFace) {
      clearAdminSession();

      if ($("adminLoginResult")) {
        $("adminLoginResult").textContent = "ID 不匹配";
      }

      setRaw({
        face: faceData,
        input_admin_id: adminId,
        id_match: false,
        error: `识别身份 ${predLabel || "未知"} 与输入管理员 ID ${adminId} 不一致`,
      });
      showToast("管理员验证失败：人脸识别结果与输入 ID 不匹配");
      return;
    }

    // 再做水印检测：管理员登录要求检测不到水印。
    const wmFd = new FormData();
    wmFd.append("image", mainFile);
    wmFd.append("label", adminId);
    wmFd.append("threshold", String(faceData?.runtime_config?.watermark_threshold ?? "0.085"));

    const watermarkData = await postForm("/detect_watermark", wmFd);

    if (watermarkData.detected) {
      clearAdminSession();

      if ($("adminLoginResult")) {
        $("adminLoginResult").textContent = "检测到水印";
      }

      setRaw({
        face: faceData,
        watermark: watermarkData,
        input_admin_id: adminId,
        id_match: true,
        watermark_clear: false,
      });
      showToast("管理员验证失败：检测到水印");
      return;
    }

    // 最后调用原有 /admin_verify 领取管理员 token。后端会再次做 ID、人脸和无水印的权威校验。
    const fd = new FormData();
    fd.append("image", mainFile);
    fd.append("admin_id", adminId);

    const data = await postForm("/admin_verify", fd);

    if (!data.verified) {
      clearAdminSession();

      if ($("adminLoginResult")) {
        $("adminLoginResult").textContent = "验证失败";
      }

      setRaw({
        face: faceData,
        watermark: watermarkData,
        admin_verify: data,
      });
      showToast(`管理员验证失败，score=${Number(data.score || 0).toFixed(4)}`);
      return;
    }

    if (!saveAdminSession(data)) {
      setRaw({
        face: faceData,
        watermark: watermarkData,
        admin_verify: data,
      });
      showToast("管理员验证通过，但后端没有返回 admin_token");
      return;
    }

    if ($("adminLoginResult")) {
      $("adminLoginResult").textContent = "验证通过";
    }

    setRaw({
      face: faceData,
      watermark: watermarkData,
      admin_verify: data,
    });
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
  const scoreDefenseEnabled = $("scoreDefenseEnabled")?.value || "false";

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
    fd.append("score_defense_enabled", scoreDefenseEnabled);

    const data = await postForm("/admin_runtime_config", fd);

    fillRuntimeConfig(data);
    setRaw(data);
    showToast("系统图库模式、阈值和分数防御设置已保存");
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

  if (!enrollFiles.length) {
    showToast("请选择或采集至少一张录入图片，建议 3-5 张");
    return;
  }

  setBusy(true);

  try {
    const fd = new FormData();
    fd.append("admin_token", adminToken);
    fd.append("label", label);
    fd.append("overwrite", $("overwriteEnroll").checked ? "true" : "false");

    enrollFiles.forEach((file) => {
      fd.append("images", file);
    });

    const data = await postForm("/enroll_face", fd);

    setRaw(data);
    showToast(`录入完成：${label}，使用 ${data.num_enroll_images || enrollFiles.length} 张图片生成模板`);
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

function getCameraGuideRectOnCanvas(canvasSize = 512) {
  const video = $("cameraVideo");
  const guide = document.querySelector(".face-guide");

  if (!video || !guide) {
    return {
      x: canvasSize * 0.255,
      y: canvasSize * 0.135,
      width: canvasSize * 0.49,
      height: canvasSize * 0.73,
    };
  }

  const videoRect = video.getBoundingClientRect();
  const guideRect = guide.getBoundingClientRect();

  if (!videoRect.width || !videoRect.height) {
    return {
      x: canvasSize * 0.255,
      y: canvasSize * 0.135,
      width: canvasSize * 0.49,
      height: canvasSize * 0.73,
    };
  }

  // captureCurrentFrame() 保存的是摄像头画面中心正方形区域，
  // 因此把预览中的绿色框映射到同一个中心正方形坐标系。
  const squareSize = Math.min(videoRect.width, videoRect.height);
  const squareLeft = videoRect.left + (videoRect.width - squareSize) / 2;
  const squareTop = videoRect.top + (videoRect.height - squareSize) / 2;

  const x = ((guideRect.left - squareLeft) / squareSize) * canvasSize;
  const y = ((guideRect.top - squareTop) / squareSize) * canvasSize;
  const width = (guideRect.width / squareSize) * canvasSize;
  const height = (guideRect.height / squareSize) * canvasSize;

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  return {
    x: clamp(x, 0, canvasSize),
    y: clamp(y, 0, canvasSize),
    width: clamp(width, 0, canvasSize),
    height: clamp(height, 0, canvasSize),
  };
}

function applySoftFaceFocusMask(ctx, canvasSize = 512) {
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = canvasSize;
  sourceCanvas.height = canvasSize;
  sourceCanvas.getContext("2d").drawImage(ctx.canvas, 0, 0);

  const maskCanvas = document.createElement("canvas");
  maskCanvas.width = canvasSize;
  maskCanvas.height = canvasSize;
  const maskCtx = maskCanvas.getContext("2d");

  // 裁剪后再做一个软椭圆注意力区域：中心脸部保留，四角、衣服和背景区域逐渐弱化。
  // 这里不用硬切白边，而是用中性灰色替代外围，避免衣服/背景颜色被模型学习到。
  const cx = canvasSize * 0.50;
  const cy = canvasSize * 0.47;
  const rx = canvasSize * 0.43;
  const ry = canvasSize * 0.49;

  maskCtx.save();
  maskCtx.translate(cx, cy);
  maskCtx.scale(rx, ry);

  const gradient = maskCtx.createRadialGradient(0, 0, 0, 0, 0, 1);
  gradient.addColorStop(0.00, "rgba(255,255,255,1)");
  gradient.addColorStop(0.70, "rgba(255,255,255,1)");
  gradient.addColorStop(0.90, "rgba(255,255,255,0.45)");
  gradient.addColorStop(1.00, "rgba(255,255,255,0)");

  maskCtx.fillStyle = gradient;
  maskCtx.fillRect(-1, -1, 2, 2);
  maskCtx.restore();

  const focusedCanvas = document.createElement("canvas");
  focusedCanvas.width = canvasSize;
  focusedCanvas.height = canvasSize;
  const focusedCtx = focusedCanvas.getContext("2d");
  focusedCtx.drawImage(sourceCanvas, 0, 0);
  focusedCtx.globalCompositeOperation = "destination-in";
  focusedCtx.drawImage(maskCanvas, 0, 0);

  ctx.clearRect(0, 0, canvasSize, canvasSize);
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, canvasSize, canvasSize);
  ctx.drawImage(focusedCanvas, 0, 0);
}

function cropCameraGuideRegion(ctx, canvasSize = 512) {
  const guide = getCameraGuideRectOnCanvas(canvasSize);

  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = canvasSize;
  sourceCanvas.height = canvasSize;
  sourceCanvas.getContext("2d").drawImage(ctx.canvas, 0, 0);

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  // 改进点：不再按绿色框完整高度裁大正方形。
  // 旧逻辑容易把脖子、衣服和背景裁进来；现在主要按绿色框宽度确定头脸区域，
  // 同时把裁剪中心略微上移，让提交图更集中于额头、脸颊、下巴等人脸区域。
  const cx = guide.x + guide.width / 2;
  const cy = guide.y + guide.height * 0.42;

  // 以绿色框宽度为主，保留完整脸部轮廓和少量头发边缘，尽量减少衣服和背景。
  let cropSide = Math.max(guide.width * 1.55, guide.height * 0.74);
  cropSide = clamp(cropSide, canvasSize * 0.42, canvasSize * 0.88);

  let sx = cx - cropSide / 2;
  let sy = cy - cropSide / 2;
  sx = clamp(sx, 0, canvasSize - cropSide);
  sy = clamp(sy, 0, canvasSize - cropSide);

  ctx.clearRect(0, 0, canvasSize, canvasSize);
  ctx.drawImage(
    sourceCanvas,
    sx,
    sy,
    cropSide,
    cropSide,
    0,
    0,
    canvasSize,
    canvasSize
  );

  // 裁剪后再做软边缘人脸聚焦，进一步弱化背景/衣服对 ArcFace embedding 的影响。
  applySoftFaceFocusMask(ctx, canvasSize);
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

  // 采集提交前，只裁剪绿色人脸框附近区域，并缩放为提交图片。
  cropCameraGuideRegion(ctx, 512);

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
        enrollFiles.push(file);
        renderEnrollFilesPreview();
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


function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

const ADMIN_LOG_CONFIG = {
  enroll: {
    title: "录入操作记录",
    endpoint: "/logs/enroll_face",
    columns: [
      ["time", "时间"],
      ["action", "操作"],
      ["label", "身份 label"],
      ["label_index", "索引"],
      ["num_identities", "身份数"],
      ["overwrite", "覆盖"],
      ["gallery_mode", "图库"],
      ["saved_image_path", "图片备份"],
      ["backup_path", "gallery 备份"],
      ["admin_label", "管理员"],
    ],
  },
  delete: {
    title: "删除操作记录",
    endpoint: "/logs/delete_face",
    columns: [
      ["time", "时间"],
      ["action", "操作"],
      ["label", "身份 label"],
      ["deleted_index", "删除索引"],
      ["num_identities_before", "删除前"],
      ["num_identities_after", "删除后"],
      ["gallery_mode", "图库"],
      ["backup_path", "gallery 备份"],
      ["admin_label", "管理员"],
    ],
  },
  face: {
    title: "统一 /face 记录",
    endpoint: "/logs/face",
    columns: [
      ["time", "时间"],
      ["face_mode", "模式"],
      ["target_id", "目标 ID"],
      ["pred_label", "预测 label"],
      ["num_images", "图片数"],
      ["score", "返回分数"],
      ["score_mean", "均值"],
      ["score_min", "最小值"],
      ["score_max", "最大值"],
      ["api_threshold", "阈值"],
      ["verified", "通过"],
      ["score_defense_enabled", "防御"],
      ["gallery_mode", "图库"],
      ["label_index", "索引"],
      ["original_filename", "原文件名"],
    ],
  },
  watermark: {
    title: "水印检测记录",
    endpoint: "/logs/watermark",
    columns: [
      ["time", "时间"],
      ["label", "检测 label"],
      ["label_index", "索引"],
      ["detected", "检测结果"],
      ["s_wm", "水印分数"],
      ["threshold", "阈值"],
      ["cos_clean", "clean 相似度"],
      ["cos_wm", "watermarked 相似度"],
      ["cos_wm_minus_clean", "差值"],
      ["theta", "theta"],
      ["sin_theta", "sin(theta)"],
      ["image_sha256", "图片 SHA256"],
      ["original_filename", "原文件名"],
      ["key_path", "key 路径"],
    ],
  },
};

function formatLogValue(key, value) {
  if (value === undefined || value === null || value === "") return "-";

  const numericKeys = new Set(["score", "score_mean", "score_min", "score_max", "api_threshold"]);
  if (numericKeys.has(key)) {
    const num = Number(value);
    if (!Number.isNaN(num)) return num.toFixed(6);
  }

  if (key === "success" || key === "overwrite" || key === "verified" || key === "score_defense_enabled") {
    return String(value) === "1" || value === true || String(value).toLowerCase() === "true" ? "是" : "否";
  }

  return String(value);
}

function renderAdminLogs(logType, data) {
  const cfg = ADMIN_LOG_CONFIG[logType];
  const wrap = $("adminLogTableWrap");
  const summary = $("adminLogSummary");
  if (!cfg || !wrap) return;

  const logs = Array.isArray(data?.logs) ? data.logs : [];
  const total = data?.num_logs ?? logs.length;

  if (summary) {
    summary.textContent = `${cfg.title}：共 ${total} 条，当前展示 ${logs.length} 条。`;
  }

  if (!logs.length) {
    wrap.innerHTML = `<div class="log-empty">暂无${escapeHtml(cfg.title)}</div>`;
    return;
  }

  const rows = logs.slice().reverse();
  const thead = cfg.columns
    .map(([, title]) => `<th>${escapeHtml(title)}</th>`)
    .join("");

  const tbody = rows
    .map((row) => {
      const cells = cfg.columns
        .map(([key]) => `<td>${escapeHtml(formatLogValue(key, row[key]))}</td>`)
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");

  wrap.innerHTML = `
    <table class="log-table">
      <thead><tr>${thead}</tr></thead>
      <tbody>${tbody}</tbody>
    </table>
  `;
}

async function loadAdminLogs(logType) {
  if (!requireAdmin()) return;

  const cfg = ADMIN_LOG_CONFIG[logType];
  if (!cfg) return;

  setBusy(true);

  try {
    const data = await getJson(cfg.endpoint);
    renderAdminLogs(logType, data);
    setRaw(data);
    showToast(`${cfg.title}已加载`);
  } catch (e) {
    showToast(e.message);
    setRaw(String(e.stack || e));
  } finally {
    setBusy(false);
  }
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
    enrollFiles = Array.from(e.target.files || []);
    enrollFile = enrollFiles[0] || null;
    renderEnrollFilesPreview();
  });

  bind("predictBtn", "click", callPredict);
  bind("watermarkBtn", "click", callWatermark);
  bind("bothBtn", "click", callWatermark);

  bind("adminLoginBtn", "click", callAdminVerify);

  bind("saveModeBtn", "click", saveRuntimeConfig);
  bind("enrollBtn", "click", enrollFace);
  bind("deleteBtn", "click", deleteFace);

  bind("loadEnrollLogsBtn", "click", () => loadAdminLogs("enroll"));
  bind("loadDeleteLogsBtn", "click", () => loadAdminLogs("delete"));
  bind("loadFaceLogsBtn", "click", () => loadAdminLogs("face"));
  bind("loadWatermarkLogsBtn", "click", () => loadAdminLogs("watermark"));

  bind("logoutAdminBtn", "click", logoutAdmin);

  bind("mainCameraBtn", "click", () => openCamera("main"));
  bind("adminLoginCameraBtn", "click", () => openCamera("adminLogin"));
  bind("enrollCameraBtn", "click", () => openCamera("enroll"));
  bind("clearEnrollImagesBtn", "click", clearEnrollFiles);

  bind("captureBtn", "click", captureCurrentFrame);
  bind("cancelCameraBtn", "click", closeCamera);
  bind("closeCameraBtn", "click", closeCamera);
}

initEvents();
checkHealth();
initAdminPage();
setInterval(checkHealth, 5000);
