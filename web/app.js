"use strict";
const $ = (id) => document.getElementById(id);
const materials = ["wood", "metal"];
const labels = {wood: "Дерево", metal: "Металл", other: "Неподходящий звук", knock: "Стук"};
const analysisMode = () => document.querySelector('input[name="analysis-mode"]:checked').value;
const headings = {
  analyze: ["Дерево или металл?", "Определите материал по звуку удара."],
  dataset: ["Данные и обучение", "Собственные записи, независимые сессии и прозрачное обучение."],
  quality: ["Проверка качества", "Точность, ошибки и условия, в которых работает классификатор."]
};
let state = null, selectedFile = null, lastResult = null, isBusy = false, recording = null, recordingPending = false;
let batch = [], batchMode = null, activeBatchIndex = -1, activeHitIndex = 0;
let playbackUrl = null;
const escapeHTML = (value) => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = (value) => (100 * value).toLocaleString("ru-RU", {maximumFractionDigits: 1}) + "%";
const prettyDate = (value) => new Date(value).toLocaleString("ru-RU");
const icon = (name) => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;

function navigate() {
  const candidate = location.hash.slice(1);
  const requestedPage = Object.hasOwn(headings, candidate) ? candidate : "analyze";
  const page = state?.hosted_read_only && requestedPage !== "analyze" ? "analyze" : requestedPage;
  if (page !== requestedPage) history.replaceState(null, "", "#analyze");
  document.querySelectorAll(".page").forEach(el => { el.hidden = el.id !== `page-${page}`; });
  document.querySelectorAll(".nav-item").forEach(el => {
    const active = el.dataset.page === page;
    el.classList.toggle("active", active);
    if (active) el.setAttribute("aria-current", "page"); else el.removeAttribute("aria-current");
    el.setAttribute("aria-label", el.textContent.replace("↗", "").trim());
  });
  if (page === "analyze") {
    $("page-title").replaceChildren(...["Дерево или ", "металл?"].map(text => {const line = document.createElement("span"); line.textContent = text; return line;}));
  } else $("page-title").textContent = headings[page][0];
  $("page-title").setAttribute("aria-label", headings[page][0]);
  document.body.dataset.page = page;
  $("page-description").textContent = headings[page][1];
  if (page === "analyze") requestAnimationFrame(drawPlots);
  window.scrollTo({top: 0, behavior: "instant"});
}

function notify(message, success = false) {
  $("notice").textContent = message;
  $("notice").className = "notice" + (success ? " success" : "");
  $("notice").setAttribute("role", success ? "status" : "alert");
  $("notice").hidden = false;
}

async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Ошибка запроса.");
  return result;
}

async function busy(button, message, operation) {
  if (isBusy || recording || recordingPending) return;
  isBusy = true;
  $("notice").hidden = true;
  const original = button.innerHTML;
  button.disabled = true;
  button.textContent = message;
  updateButtons();
  document.body.classList.add("busy");
  document.querySelectorAll(".page").forEach(el => el.setAttribute("aria-busy", "true"));
  try { await operation(); }
  catch (error) { notify(error.message || "Сервер анализа недоступен. Проверьте подключение и попробуйте ещё раз."); }
  finally {
    isBusy = false;
    button.innerHTML = original;
    button.disabled = false;
    document.body.classList.remove("busy");
    document.querySelectorAll(".page").forEach(el => el.removeAttribute("aria-busy"));
    if (state) updateButtons();
  }
}

function updateButtons() {
  const locked = isBusy || !!recording || recordingPending;
  $("train-btn").disabled = locked || !state?.can_train;
  $("train-gate-btn").disabled = locked || !state?.can_train_gate;
  $("train-material-btn").disabled = locked || !state?.can_train_material;
  $("evaluate-btn").disabled = locked || state?.hosted_read_only || !( $("evaluation-mode").value === "material" ? state?.can_evaluate_material : state?.can_evaluate );
  $("analyze-btn").disabled = locked || !batch.length;
  $("record-btn").disabled = isBusy || recordingPending;
  for (const id of ["save-data-btn", "refresh-data", "audio-file", "dataset-files", "dataset-class", "dataset-split", "dataset-session", "evaluation-mode"]) $(id).disabled = locked;
  document.querySelectorAll('input[name="analysis-mode"]').forEach(input => {input.disabled = locked;});
  $("multiple-impacts").disabled = locked;
  document.querySelectorAll("[data-batch-index]").forEach(button => {
    const result = batch[Number(button.dataset.batchIndex)]?.result;
    button.disabled = locked || !result || result.impacts?.length === 0;
  });
}

async function refresh() {
  state = await api("/api/state");
  if (state.hosted_read_only) {
    document.querySelectorAll('.nav-item[data-page="dataset"], .nav-item[data-page="quality"]').forEach(item => {item.style.display = "none";});
    $("result-action").hidden = true;
    document.querySelector(".next-steps").hidden = true;
    document.querySelector(".local-card").innerHTML = `<span class="status-dot"></span><strong>Анализ на сервере</strong><p>Загруженные аудиозаписи передаются на сервер для обработки.</p>`;
    document.querySelector(".sidebar-footer span").textContent = "ПУБЛИЧНЫЙ РЕЖИМ";
    state.report = null;
    state.validation = null;
    navigate();
  }
  renderState();
}

function renderState() {
  const counts = state.counts;
  const modelSummary = (title, model, featureCount, kind) => {
    const features = model?.n_features ?? featureCount;
    const training = model?.n_train;
    const fitted = model?.n_fit ?? training;
    return `<section class="model-summary-item"><h3>${title}</h3><p>${kind} · ${features ? `${features} признаков` : "признаки: нет данных"}</p><p>Обучение: ${training ?? "—"} исходных записей · ${fitted ?? "—"} после аугментации</p></section>`;
  };
  $("model-summary").innerHTML = [
    modelSummary("Детектор стука", state.gate_model, 42, "LogReg · спектральные и временные"),
    modelSummary("Классификатор материала", state.model, 39, "LogReg · MFCC и спектральные")
  ].join("");
  const fitCount = (model) => model?.n_fit && model.n_fit > model.n_train
    ? ` · ${model.n_fit} примеров с аугментацией` : "";
  const trainingRecordings = counts.train.wood + counts.train.metal + counts.train.other;
  const augmentedTrainingExamples = state.gate_model?.n_fit ?? trainingRecordings * 3;
  const testCounts = ["val", "test"].reduce((total, split) => ({
    wood: total.wood + counts[split].wood,
    metal: total.metal + counts[split].metal,
    other: total.other + counts[split].other
  }), {wood: 0, metal: 0, other: 0});
  const testRecordings = testCounts.wood + testCounts.metal + testCounts.other;
  const augmentationDetails = state.model && state.gate_model
    ? `<br>После аугментации: материал ${state.model.n_fit ?? state.model.n_train} · детектор ${state.gate_model.n_fit ?? state.gate_model.n_train}`
    : "";
  $("data-summary").innerHTML = `
    <div class="data-stat"><h2>Обучение</h2><div class="count">${augmentedTrainingExamples}</div><small>${trainingRecordings} исходных записей с учётом аугментации<br>Дерево: ${counts.train.wood} · Металл: ${counts.train.metal}<br>Неподходящие: ${counts.train.other}${augmentationDetails}</small></div>
    <div class="data-stat"><h2>Тест</h2><div class="count">${testRecordings}</div><small>Валидация + тест<br>Дерево: ${testCounts.wood} · Металл: ${testCounts.metal} · Неподходящие: ${testCounts.other}</small></div>`;
  $("training-status").textContent = `01 · Детектор: ${state.gate_trained ? "готов · " + state.gate_model.n_train + " записей" + fitCount(state.gate_model) : "нужны обучение и корректные данные"}\n02 · Материал: ${state.material_trained ? "готов · " + state.model.n_train + " стуков" + fitCount(state.model) : "нужны обучение и корректные данные"}`;
  const issues = [...state.issues, ...state.errors.map(e => `${e.path}: ${e.error}`)];
  $("data-issues").innerHTML = issues.length ? `<div class="notice"><strong>Исправьте данные перед обучением</strong><ul>${issues.map(e => `<li>${escapeHTML(e)}</li>`).join("")}</ul></div>` : "";
  if (state.warnings?.length) $("data-issues").innerHTML += `<div class="notice neutral">${state.warnings.map(escapeHTML).join("<br>")}</div>`;
  $("dataset-table").innerHTML = state.rows.length ? table(["Файл", "Тип записи", "Выборка", "Сессия"], state.rows.map(r => [r.path, labels[r.label], r.split, r.session])) : `<div class="empty-table">Пока нет записей. Добавьте свои WAV / MP3 через форму выше.<br>Тестовые звуки и результаты не подставляются автоматически.</div>`;
  if (!lastResult) resetResult();
  renderReport();
  updateButtons();
}

function table(headers, rows) {
  return `<div class="table-scroller"><table><thead><tr>${headers.map(x => `<th scope="col">${escapeHTML(x)}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${escapeHTML(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function resetResult() {
  delete document.querySelector(".result-panel").dataset.material;
  if (!lastResult) {
    $("result-file").hidden = true;
    $("signal-reading").hidden = true;
    $("wave-empty").hidden = false;
    $("spectrum-empty").hidden = false;
    $("signal-tag").textContent = selectedFile ? "ОЖИДАЕТ АНАЛИЗА" : "НЕТ СИГНАЛА";
    $("audio-player").hidden = true;
    $("audio-player").pause();
    $("audio-player").removeAttribute("src");
    if (playbackUrl) { URL.revokeObjectURL(playbackUrl); playbackUrl = null; }
  }
  const ready = analysisMode() === "material" ? state?.material_trained : state?.trained;
  $("result-kicker").textContent = ready ? "ВЫБРАННЫЙ РЕЖИМ ГОТОВ" : "НУЖНО ОБУЧЕНИЕ";
  $("result-title").textContent = ready ? "Готов к приёму" : "Нет образцов";
  $("result-description").textContent = ready ? "Выберите новый файл и запустите анализ. Проверяйте результат по известному материалу предмета." : "Добавьте примеры и обучите нужные модели. Для каскада нужны также неподходящие звуки: речь, музыка и шум.";
  $("result-action").textContent = ready ? "Посмотреть проверку качества →" : "Добавить обучающие записи →";
  $("result-action").href = ready ? "#quality" : "#dataset";
  $("result-dot").style.background = ready ? "var(--green)" : "#afb8ac";
  for (const label of materials) {
    $("score-" + label).textContent = "—";
    $("bar-" + label).value = 0;
  }
  renderStages(null);
}

function renderStages(result) {
  const gate = result?.gate;
  const status = gate?.status || (analysisMode() === "material" ? "skipped" : "waiting");
  $("gate-stage").dataset.status = status;
  $("gate-status").textContent = ({waiting: "Ожидает запись", skipped: "Пропущен", not_trained: "Не обучен", accepted: "Стук принят", rejected: "Запись отклонена"})[status];
  $("gate-score").textContent = gate?.scores ? `${status === "accepted" ? "Похоже на стук" : "Не похоже на стук"} · оценка класса «стук»: ${pct(gate.scores.knock)}` : status === "skipped" ? "Выбран прямой прогноз материала" : gate?.reason || "Детектор не запускался";
  $("material-stage").dataset.status = result?.prediction ? "accepted" : "waiting";
  $("material-status").textContent = result?.prediction ? labels[result.prediction] : result ? (gate?.status === "rejected" || gate?.status === "not_trained" ? "Не запускалась" : "Не обучена") : "Ожидает анализ";
}

function setFile(file) { setFiles(file ? [file] : []); }

function setFiles(files) {
  if (isBusy || recording || recordingPending) return;
  const selection = Array.from(files);
  if (!selection.length) return;
  batch = selection.map(file => ({file, status: "queued", result: null, error: null}));
  $("notice").hidden = true;
  $("file-loaded").hidden = false;
  $("file-name").textContent = batch.length === 1 ? batch[0].file.name : `Выбрано файлов: ${batch.length}`;
  $("file-name").title = $("file-name").textContent;
  const totalSize = batch.reduce((sum, row) => sum + row.file.size, 0);
  $("file-meta").textContent = `${Math.ceil(totalSize / 1024)} КБ всего · обработка по очереди`;
  $("analyze-btn").textContent = batch.length === 1 ? "Анализировать" : `Обработать ${batch.length} файлов`;
  $("upload-title").textContent = batch.length === 1 ? "Запись выбрана" : "Записи выбраны";
  $("upload-hint").textContent = "Нажмите, чтобы заменить выбранные файлы";
  resetBatchResults(); updateButtons();
}

function resetBatchResults() {
  for (const row of batch) {row.status = "queued"; row.result = null; row.error = null;}
  batchMode = null; activeBatchIndex = -1;
  selectedFile = batch[0]?.file || null;
  lastResult = null;
  resetResult(); drawPlots(); renderBatch();
}

function batchResultText(row) {
  if (row.error) return row.error;
  const result = row.result;
  if (!result) return "—";
  if (result.impacts) return result.impacts.length ? `Найдено ударов: ${result.impacts.length}` : "Отдельные удары не найдены";
  if (result.prediction) return labels[result.prediction];
  if (result.gate.status === "rejected") return ({silence: "Тишина", clipping: "Перегрузка записи"})[result.gate.code] || "Неподходящий звук";
  return result.gate.status === "not_trained" ? "Детектор не обучен" : "Модель материала не обучена";
}

function renderBatch() {
  $("batch-results").hidden = !batch.length;
  if (!batch.length) return;
  const completed = batch.filter(row => row.status === "done" || row.status === "error").length;
  const current = batch.find(row => row.status === "processing");
  $("batch-mode").textContent = (batchMode || analysisMode()) === "cascade" ? "КАСКАД 1 → 2" : "ТОЛЬКО МОДЕЛЬ 2";
  $("batch-progress").textContent = `Обработано ${completed} из ${batch.length}` + (current ? ` · сейчас: ${current.file.name}` : completed === batch.length ? " · готово" : " · нажмите кнопку обработки");
  $("batch-meter").max = batch.length;
  $("batch-meter").value = completed;
  const statuses = {queued: "В очереди", processing: "Обработка…", done: "Готово", error: "Ошибка"};
  const previousScroller = $("batch-table").querySelector(".table-scroller");
  const scrollTop = previousScroller?.scrollTop || 0, scrollLeft = previousScroller?.scrollLeft || 0;
  const hitRows = (row, index) => (row.result?.impacts || []).map((hit, hitIndex) => `<tr class="impact-row ${index === activeBatchIndex && hitIndex === activeHitIndex ? "selected" : ""}"><td>↳ Удар ${hitIndex + 1} · ${hit.impact.time_seconds.toLocaleString("ru-RU", {minimumFractionDigits: 2})} с</td><td>Готово</td><td><strong>${escapeHTML(batchResultText({result: hit}))}</strong>${hit.prediction ? `<small>${pct(hit.scores[hit.prediction])}</small>` : ""}</td><td><button class="button quiet" type="button" data-batch-index="${index}" data-hit-index="${hitIndex}" aria-pressed="${index === activeBatchIndex && hitIndex === activeHitIndex}" ${isBusy ? "disabled" : ""}>Показать</button></td></tr>`).join("");
  $("batch-table").innerHTML = `<div class="table-scroller"><table class="batch-table"><thead><tr><th scope="col">Файл</th><th scope="col">Статус</th><th scope="col">Результат</th><th scope="col">Подробнее</th></tr></thead><tbody>${batch.map((row, index) => `<tr data-status="${row.status}" class="${index === activeBatchIndex && !row.result?.impacts ? "selected" : ""}"><td class="batch-name">${escapeHTML(row.file.name)}</td><td>${statuses[row.status]}</td><td><strong>${escapeHTML(batchResultText(row))}</strong>${row.result?.gate?.status === "rejected" ? `<small>${escapeHTML(row.result.gate.reason)}</small>` : ""}${row.result?.warnings?.length ? `<small>${escapeHTML(row.result.warnings.join(" "))}</small>` : ""}</td><td><button class="button quiet" type="button" data-batch-index="${index}" aria-pressed="${index === activeBatchIndex && !row.result?.impacts}" aria-label="Показать результат: ${escapeHTML(row.file.name)}" ${isBusy || !row.result || row.result.impacts?.length === 0 ? "disabled" : ""}>Показать</button></td></tr>${hitRows(row, index)}`).join("")}</tbody></table></div>`;
  const scroller = $("batch-table").querySelector(".table-scroller");
  scroller.scrollTop = scrollTop; scroller.scrollLeft = scrollLeft;
}

function highlightBatchResult() {
  $("batch-table").querySelectorAll("[data-batch-index]").forEach(button => {
    const index = Number(button.dataset.batchIndex);
    const isFileSummary = batch[index]?.result?.impacts && !button.hasAttribute("data-hit-index");
    const selected = !isFileSummary && index === activeBatchIndex && Number(button.dataset.hitIndex || 0) === activeHitIndex;
    button.closest("tr").classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
  });
}

async function encodeFile(file) {
  const data = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = () => reject(new Error("Не удалось прочитать файл."));
    reader.readAsDataURL(file);
  });
  return {audio: data, filename: file.name};
}

async function analyze() {
  if (!batch.length) return;
  await busy($("analyze-btn"), "Обработка…", async () => {
    resetBatchResults();
    batchMode = analysisMode();
    $("processing").hidden = false;
    if (innerWidth < 700 && batch.length > 1) $("batch-results").scrollIntoView({block: "start", behavior: "instant"});
    try {
      for (let index = 0; index < batch.length; index++) {
        const row = batch[index];
        row.status = "processing";
        $("processing").textContent = `${index + 1} из ${batch.length} · ${row.file.name}`;
        renderBatch();
        try {
          if (!/\.(wav|mp3)$/i.test(row.file.name)) throw new Error("Поддерживаются только WAV и MP3.");
          if (!row.file.size || row.file.size > 12 * 1024 * 1024) throw new Error("Нужен непустой файл размером до 12 МБ.");
          row.result = await api("/api/analyze", {...await encodeFile(row.file), mode: batchMode, multiple: $("multiple-impacts").checked});
          row.status = "done";
          if (activeBatchIndex === -1 && (!row.result.impacts || row.result.impacts.length)) showBatchResult(index);
        } catch (error) {
          row.status = "error";
          row.error = error.message || "Не удалось обработать запись.";
        }
        renderBatch();
      }
      await refresh();
      const errors = batch.filter(row => row.status === "error").length;
      notify(`Обработано файлов: ${batch.length - errors} из ${batch.length}. Ошибок: ${errors}. Результаты — в таблице.`, errors === 0);
      if (innerWidth < 700 && batch.length === 1 && lastResult) document.querySelector(".result-panel").scrollIntoView({block: "start", behavior: "instant"});
    } finally { $("processing").hidden = true; }
  });
}

function showBatchResult(index, hitIndex = 0) {
  const row = batch[index];
  if (!row?.result) return;
  const result = row.result.impacts ? row.result.impacts[hitIndex] : row.result;
  if (!result) return;
  activeBatchIndex = index; activeHitIndex = hitIndex;
  selectedFile = row.file;
  lastResult = result;
  resetResult();
  $("result-file").textContent = row.file.name + (lastResult.impact ? ` · удар ${lastResult.impact.number} · ${lastResult.impact.time_seconds.toLocaleString("ru-RU")} с` : "");
  $("result-file").hidden = false;
  $("wave-empty").hidden = true;
  $("spectrum-empty").hidden = true;
  $("signal-tag").textContent = `${lastResult.info.features} ПРИЗНАКОВ · 16 КГЦ`;
  $("signal-reading").hidden = false;
  $("centroid-value").textContent = lastResult.info.centroid_hz.toLocaleString("ru-RU") + " Гц";
  const bytes = Uint8Array.from(atob(lastResult.playback), c => c.charCodeAt(0));
  if (playbackUrl) URL.revokeObjectURL(playbackUrl);
  playbackUrl = URL.createObjectURL(new Blob([bytes], {type: "audio/wav"}));
  $("audio-player").src = playbackUrl;
  $("audio-player").hidden = false;
  if (lastResult.prediction) {
    document.querySelector(".result-panel").dataset.material = lastResult.prediction;
    const score = lastResult.scores[lastResult.prediction];
    $("result-kicker").textContent = "ПРЕДПОЛАГАЕМЫЙ МАТЕРИАЛ";
    $("result-dot").style.background = `var(--${lastResult.prediction})`;
    $("result-title").textContent = labels[lastResult.prediction];
    $("result-description").textContent = score < 0.65 ? "Баллы классов близки. Вывод неустойчив — сравните с известным материалом и повторите запись." : "Модель выбрала этот класс среди двух известных. Сверьте результат с известным материалом.";
    for (const label of materials) {
      $("score-" + label).textContent = pct(lastResult.scores[label]);
      $("bar-" + label).value = lastResult.scores[label];
    }
    $("result-action").textContent = "Посмотреть проверку качества →";
    $("result-action").href = "#quality";
  } else {
    resetResult();
    const rejected = lastResult.gate.status === "rejected";
    $("result-kicker").textContent = rejected ? "МАТЕРИАЛ НЕ ОПРЕДЕЛЯЛСЯ" : "НУЖНО ОБУЧЕНИЕ";
    $("result-title").textContent = rejected ? "Не подходит" : "Нужны примеры";
    $("result-description").textContent = rejected || lastResult.gate.status === "not_trained" ? lastResult.gate.reason : "Сигнал обработан. Обучите модель материала на записях дерева и металла.";
    $("result-dot").style.background = "var(--orange)";
  }
  renderStages(lastResult);
  drawPlots();
  highlightBatchResult();
}

function canvasContext(id) {
  const canvas = $(id), rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width)), height = Math.max(1, Math.round(rect.height));
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = width * ratio; canvas.height = height * ratio;
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  return {ctx, width, height};
}

function drawPlots() {
  if ($("page-analyze").hidden) return;
  const {ctx, width, height} = canvasContext("waveform");
  ctx.clearRect(0, 0, width, height);
  if (lastResult) {
    const values = lastResult.waveform;
    ctx.strokeStyle = "#d8f66a"; ctx.lineWidth = 1;
    values.forEach((pair, i) => {
      const x = i * width / values.length;
      ctx.beginPath(); ctx.moveTo(x, height / 2 - pair[1] * height * .45); ctx.lineTo(x, height / 2 - pair[0] * height * .45); ctx.stroke();
    });
    ctx.strokeStyle = "#d8f66a50";
    ctx.beginPath(); ctx.moveTo(0, height / 2); ctx.lineTo(width, height / 2); ctx.stroke();
  }
  const spectral = canvasContext("spectrogram");
  spectral.ctx.fillStyle = "#111613"; spectral.ctx.fillRect(0, 0, spectral.width, spectral.height);
  if (!lastResult) return;
  const matrix = lastResult.spectrogram, rows = matrix.length, cols = matrix[0].length;
  const stops = [[17,22,19],[39,67,53],[102,139,66],[210,233,92],[255,244,207]];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const value = Math.max(0, Math.min(1, (matrix[r][c] + 80) / 80));
    const band = Math.min(3, Math.floor(value * 4)), t = value * 4 - band;
    const rgb = stops[band].map((v, k) => Math.round(v + t * (stops[band + 1][k] - v)));
    spectral.ctx.fillStyle = `rgb(${rgb.join(",")})`;
    spectral.ctx.fillRect(c * spectral.width / cols, (rows-r-1) * spectral.height / rows, spectral.width / cols + 1, spectral.height / rows + 1);
  }
}

function renderReport() {
  const report = state.report?.mode === $("evaluation-mode").value ? state.report : null;
  $("quality-empty").hidden = !!report;
  $("quality-results").hidden = !report;
  if (!report) return;
  const metrics = [[pct(report.accuracy), "Точность (accuracy)"], [pct(report.balanced_accuracy), "Сбалансированная точность"], [report.macro_f1.toFixed(3), "Macro F1"], [String(report.n_test), "Новых проверочных записей"]];
  $("metrics").innerHTML = metrics.map(([value, label]) => `<div class="metric"><div class="value">${escapeHTML(value)}</div><div class="label">${label}</div></div>`).join("");
  $("confusion").innerHTML = `<table class="confusion-table"><thead><tr><th scope="col">Факт ↓ / Прогноз →</th>${report.labels.map(l => `<th scope="col">${labels[l]}</th>`).join("")}</tr></thead><tbody>${report.confusion.map((row, i) => `<tr><th scope="row">${labels[report.labels[i]]}</th>${row.map((v,j) => `<td class="${i===j ? "diagonal" : ""}">${v}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  $("report-passport").innerHTML = `<div class="report-meta"><p><strong>Режим:</strong> ${report.mode === "cascade" ? "Каскад 1 → 2, включая отказы" : "Только модель 2, только дерево и металл"}</p><p><strong>Материал:</strong> ${escapeHTML(report.model_id)}</p>${report.gate_id ? `<p><strong>Детектор:</strong> ${escapeHTML(report.gate_id)}</p>` : ""}<p><strong>Проверено:</strong> ${escapeHTML(prettyDate(report.created))}</p><p><strong>Сессии test:</strong> ${report.sessions.map(escapeHTML).join(", ")}</p><p><strong>95% интервал accuracy:</strong> ${pct(report.wilson95[0])} – ${pct(report.wilson95[1])}</p></div>`;
  $("gate-report").hidden = !report.gate;
  if (report.gate) {
    const g = report.gate, m = report.material_accepted;
    $("gate-report").innerHTML = `<h2>01 · Что пропускает детектор</h2><p>Принято настоящих стуков: ${g.n_knocks - g.false_rejects} из ${g.n_knocks} (${pct(g.knock_recall)}). Ошибочно пропущено неподходящих: ${g.false_accepts} из ${g.n_other} (${pct(g.false_accept_rate)}).</p><p>02 · Материал среди принятых настоящих стуков: ${m.accuracy === null ? "нет принятых стуков" : pct(m.accuracy) + " на " + m.n_test + " записях"}. Охват настоящих стуков: ${pct(m.coverage)}.</p>`;
  }
  $("evaluation-table").innerHTML = table(["Файл", "Материал", "Прогноз", "Результат"], report.examples.map(r => [r.path, labels[r.actual], labels[r.predicted], r.correct ? "✓ Верно" : "✕ Ошибка"]));
}

$("audio-file").addEventListener("change", e => {setFiles(e.target.files); e.target.value = "";});
$("analyze-btn").addEventListener("click", analyze);
$("batch-table").addEventListener("click", e => {
  const button = e.target.closest("button[data-batch-index]");
  if (!button || isBusy || recording || recordingPending) return;
  showBatchResult(Number(button.dataset.batchIndex), Number(button.dataset.hitIndex || 0));
  if (lastResult?.warnings.length) notify(lastResult.warnings.join(" "));
  else $("notice").hidden = true;
  if (innerWidth < 700) document.querySelector(".signal-panel").scrollIntoView({block: "start", behavior: "instant"});
});
const drop = $("drop-zone");
for (const eventName of ["dragenter", "dragover"]) drop.addEventListener(eventName, e => {e.preventDefault(); drop.classList.add("dragging");});
for (const eventName of ["dragleave", "drop"]) drop.addEventListener(eventName, e => {e.preventDefault(); drop.classList.remove("dragging");});
drop.addEventListener("drop", e => setFiles(e.dataTransfer.files));
$("dataset-form").addEventListener("submit", async e => {
  e.preventDefault();
  const files = Array.from($("dataset-files").files);
  const split = $("dataset-split").value, label = $("dataset-class").value, session = $("dataset-session").value;
  await busy($("save-data-btn"), "Добавляем…", async () => {
    let count = 0;
    const errors = [];
    for (const file of files) {
      $("import-status").textContent = `Обработка ${count + errors.length + 1} из ${files.length}: ${file.name}`;
      try {
        if (file.size > 12 * 1024 * 1024) throw new Error("Больше 12 МБ.");
        await api("/api/data", {...await encodeFile(file), split, label, session});
        count++;
      } catch (error) {errors.push(`${file.name}: ${error.message}`);}
    }
    resetBatchResults();
    await refresh();
    $("import-status").textContent = `Добавлено: ${count} из ${files.length}.`;
    notify(errors.length ? `Добавлено: ${count}. ${errors.join(" ")}` : `Добавлено записей: ${count}.`, !errors.length);
  });
});
for (const [id, stage] of [["train-btn", "both"], ["train-gate-btn", "gate"], ["train-material-btn", "material"]]) {
  $(id).addEventListener("click", () => busy($(id), "Обучаем…", async () => {
    state = await api("/api/train", {stage}); resetBatchResults(); renderState();
    notify(`${stage === "both" ? "Обе модели обучены" : stage === "gate" ? "Детектор обучен" : "Модель материала обучена"}. Качество ещё нужно проверить на test.`, true);
  }));
}
document.querySelectorAll('input[name="analysis-mode"]').forEach(input => input.addEventListener("change", () => {
  $("notice").hidden = true;
  $("mode-description").textContent = analysisMode() === "material" ? "Модель 1 пропущена. Речь и шум могут получить метку дерева или металла. Проверка тишины и повреждений файла сохраняется." : "Модель 1 отсекает неподходящие звуки. Модель 2 получает только прошедшие проверку записи.";
  resetBatchResults();
}));
$("evaluation-mode").addEventListener("change", () => {renderReport(); updateButtons();});
$("evaluate-btn").addEventListener("click", () => busy($("evaluate-btn"), "Проверяем…", async () => {
  await api("/api/evaluate", {mode: $("evaluation-mode").value}); await refresh();
  notify(`Проверка завершена: ${state.report.n_test} записей, точность ${pct(state.report.accuracy)}. Это измерение только для текущей выборки.`, true);
}));
$("refresh-data").addEventListener("click", () => busy($("refresh-data"), "Читаем…", async () => {
  resetBatchResults(); await refresh(); notify("Данные с диска обновлены.", true);
}));

// PCM WAV capture: avoids browser-specific MediaRecorder containers.
function wavBlob(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2), view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  text(0, "RIFF"); view.setUint32(4, buffer.byteLength - 8, true); text(8, "WAVE"); text(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); text(36, "data"); view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, sample)) * 32767, true));
  return new Blob([buffer], {type: "audio/wav"});
}

async function stopRecording() {
  if (!recording) return;
  const rec = recording; recording = null;
  clearTimeout(rec.timeout);
  rec.node.disconnect(); rec.source.disconnect(); rec.mute.disconnect();
  rec.stream.getTracks().forEach(track => track.stop());
  const rate = rec.context.sampleRate;
  await rec.context.close();
  const samples = new Float32Array(rec.chunks.reduce((n, chunk) => n + chunk.length, 0));
  let offset = 0; for (const chunk of rec.chunks) { samples.set(chunk, offset); offset += chunk.length; }
  $("record-btn").innerHTML = icon("mic") + "Записать звук";
  $("record-btn").classList.remove("recording");
  $("processing").hidden = true;
  if (samples.length < rate * .15) {notify("Запись слишком короткая. Повторите и сделайте один стук."); return;}
  setFile(new File([wavBlob(samples, rate)], `microphone_${Date.now()}.wav`, {type: "audio/wav"}));
  notify("Записано. Нажмите «Анализировать», чтобы проверить звук.", true);
}

$("record-btn").addEventListener("click", async () => {
  if (isBusy || recordingPending) return;
  if (recording) {await stopRecording(); return;}
  if (!navigator.mediaDevices?.getUserMedia) {notify("Микрофон недоступен в этом браузере. Загрузите WAV или MP3."); return;}
  const button = $("record-btn");
  recordingPending = true;
  updateButtons();
  let stream = null, context = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: false, noiseSuppression: false, autoGainControl: false}});
    context = new AudioContext(); await context.resume();
    const source = context.createMediaStreamSource(stream);
    const node = context.createScriptProcessor(4096, 1, 1), mute = context.createGain(); mute.gain.value = 0;
    const chunks = [];
    node.onaudioprocess = event => {chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));};
    source.connect(node); node.connect(mute); mute.connect(context.destination);
    recording = {stream, context, source, node, mute, chunks, timeout: setTimeout(stopRecording, 3000)};
    button.innerHTML = "■ Остановить запись"; button.classList.add("recording");
    $("processing").textContent = "Запись 3 секунды. Сделайте один мягкий стук рядом с микрофоном.";
    $("processing").hidden = false;
    $("notice").hidden = true;
  } catch (error) {
    stream?.getTracks().forEach(track => track.stop());
    if (context) await context.close();
    notify("Не удалось получить доступ к микрофону. Разрешите его в браузере или загрузите аудиофайл.");
  } finally {recordingPending = false; updateButtons();}
});
window.addEventListener("beforeunload", () => {recording?.stream.getTracks().forEach(track => track.stop());});
window.addEventListener("hashchange", navigate);
let resizeTimer;
window.addEventListener("resize", () => {clearTimeout(resizeTimer); resizeTimer = setTimeout(drawPlots, 100);});
navigate();
drawPlots();
refresh().catch(() => notify("Сервер анализа недоступен. Проверьте подключение и попробуйте ещё раз."));

$("multiple-impacts").addEventListener("change", () => {
  $("file-limit").textContent = $("multiple-impacts").checked ? "Отдельные удары с паузами · до 10 секунд · до 12 МБ каждый" : "Один стук на файл · до 10 секунд · до 12 МБ каждый";
  $("notice").hidden = true;
  resetBatchResults();
});
