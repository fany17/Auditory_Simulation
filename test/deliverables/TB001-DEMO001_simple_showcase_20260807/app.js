(() => {
  "use strict";

  const dataset = window.TB001_SHOWCASE;
  const frontendDataset = window.TB001_FRONTEND;
  const reference = window.TB001_REFERENCE;
  if (!dataset || !Array.isArray(dataset.conditions)) {
    throw new Error("真实 showcase 数据未载入");
  }
  if (!frontendDataset || !Array.isArray(frontendDataset.conditions)) {
    throw new Error("真实卷积前端数据未载入");
  }
  if (!reference || !reference.text) {
    throw new Error("数据集真实标注未载入");
  }

  const $ = id => document.getElementById(id);
  const defaultChoice = dataset.default_selection;
  const state = {
    conditionId: defaultChoice.condition_id,
    layerId: Number(defaultChoice.layer_id),
    alpha: Number(defaultChoice.alpha),
    mode: "layer",
    frontendVariantId: "frontend-100hz",
  };

  const LAYER_GUIDE = {
    1: {group: "前段 · 声学帧上下文化", title: "建立帧间联系", role: "让卷积前端产生的语音帧第一次彼此交换信息，建立最初的上下文化表示。"},
    2: {group: "前段 · 声学帧上下文化", title: "整合短邻域", role: "继续汇合相邻时刻的变化，使短时声学线索不再只依赖单个帧。"},
    3: {group: "前段 · 局部模式组织", title: "累积局部模式", role: "把多帧线索组合成更连续的局部模式，为后续层扩大上下文范围。"},
    4: {group: "前段 · 局部模式组织", title: "完成早期整理", role: "整理前段累计的局部声学模式，并把更稳定的表示送入网络中段。"},
    5: {group: "中段 · 上下文整合", title: "接入更宽上下文", role: "承接前段表示，让当前语音片段进一步结合前后信息；它处在声学表示向识别表示过渡的中段。"},
    6: {group: "中段 · 上下文整合", title: "延伸序列联系", role: "继续传播较长时间范围的信息，使不同语音片段能够在同一表示中相互影响。"},
    7: {group: "中段 · 表示重组", title: "重组中层表示", role: "重新组合已累积的局部线索与上下文，为后段面向文字解码的处理提供输入。"},
    8: {group: "中段 · 表示重组", title: "稳定上下文表示", role: "整合网络中段的累计结果，并将上下文更完整的表示交给后四层。"},
    9: {group: "后段 · 解码相关整合", title: "靠近解码边界", role: "开始更直接地影响送往 CTC 的类别分布；其作用仍由后续层继续重整。"},
    10: {group: "后段 · 解码相关整合", title: "强化输出方向", role: "在接近输出的位置继续整合上下文，使表示更适合被 CTC 头映射为字符类别。"},
    11: {group: "后段 · 输出前重整", title: "解码前重整", role: "对即将输出的表示再做一次上下文化重组，留给后续修正的层数已经很少。"},
    12: {group: "后段 · 输出前重整", title: "完成最终上下文化", role: "执行最后一次 Transformer 更新；它的输出随后经过最终归一化并进入 CTC 文字头。"},
  };

  const speedButtons = $("speedButtons");
  const layerStack = $("layerStack");
  const strengthButtons = Array.from(document.querySelectorAll("[data-alpha]"));
  const frontendButtons = Array.from(document.querySelectorAll("[data-frontend]"));

  function condition() {
    return dataset.conditions.find(item => item.id === state.conditionId);
  }

  function frontendCondition() {
    return frontendDataset.conditions.find(item => item.id === state.conditionId);
  }

  function selectedFrontendRun() {
    return frontendCondition().variants.find(item => item.frontend_variant_id === state.frontendVariantId);
  }

  function selectedRun() {
    if (state.mode === "frontend") return selectedFrontendRun();
    const current = condition();
    if (state.alpha === 1) return current.baseline;
    return current.runs.find(run => Number(run.layer_id) === state.layerId && Number(run.alpha) === state.alpha);
  }

  function runFor(current, layerId, alpha) {
    if (alpha === 1) return current.baseline;
    return current.runs.find(run => Number(run.layer_id) === layerId && Number(run.alpha) === alpha);
  }

  function buildControls() {
    dataset.conditions.forEach(item => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = item.speed_factor === 1 ? "原速" : `${item.speed_factor}×`;
      button.dataset.condition = item.id;
      button.addEventListener("click", () => {
        state.conditionId = item.id;
        render();
      });
      speedButtons.appendChild(button);
    });

    for (let layer = 1; layer <= 12; layer += 1) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "layer-button";
      button.dataset.layer = String(layer);
      const code = document.createElement("span");
      code.className = "layer-code";
      code.textContent = `L${layer}`;
      const short = document.createElement("span");
      short.className = "layer-short";
      short.textContent = LAYER_GUIDE[layer].title;
      const impact = document.createElement("span");
      impact.className = "layer-impact";
      impact.setAttribute("aria-hidden", "true");
      button.append(code, short, impact);
      button.addEventListener("click", () => {
        state.mode = "layer";
        state.layerId = layer;
        render();
      });
      layerStack.appendChild(button);
    }

    strengthButtons.forEach(button => {
      button.addEventListener("click", () => {
        state.mode = "layer";
        state.alpha = Number(button.dataset.alpha);
        render();
      });
    });

    frontendButtons.forEach(button => {
      button.addEventListener("click", () => {
        state.mode = "frontend";
        state.frontendVariantId = button.dataset.frontend;
        render();
      });
    });

    $("frontendNode").addEventListener("click", () => {
      state.mode = "frontend";
      render();
    });
  }

  function drawWaveform(values) {
    const canvas = $("waveformCanvas");
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(220, canvas.clientWidth);
    const height = 104;
    canvas.width = width * ratio;
    canvas.height = height * ratio;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "#aebdc3";
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
    const maxAbs = Math.max(...values.map(Math.abs), 1e-6);
    ctx.strokeStyle = "#166b8f";
    ctx.lineWidth = 1.35;
    ctx.beginPath();
    values.forEach((value, index) => {
      const x = index * width / Math.max(1, values.length - 1);
      const y = height / 2 - value / maxAbs * height * 0.41;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  function wordDiff(before, after) {
    const a = before.split(/\s+/).filter(Boolean);
    const b = after.split(/\s+/).filter(Boolean);
    const dp = Array.from({length: a.length + 1}, () => Array(b.length + 1).fill(0));
    for (let i = a.length - 1; i >= 0; i -= 1) {
      for (let j = b.length - 1; j >= 0; j -= 1) {
        dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
    const beforeTokens = [];
    const afterTokens = [];
    let i = 0;
    let j = 0;
    while (i < a.length || j < b.length) {
      if (i < a.length && j < b.length && a[i] === b[j]) {
        beforeTokens.push({text: a[i], type: "same"});
        afterTokens.push({text: b[j], type: "same"});
        i += 1;
        j += 1;
      } else if (j >= b.length || (i < a.length && dp[i + 1][j] >= dp[i][j + 1])) {
        beforeTokens.push({text: a[i], type: "removed"});
        i += 1;
      } else {
        afterTokens.push({text: b[j], type: "added"});
        j += 1;
      }
    }
    return {beforeTokens, afterTokens};
  }

  function characterEditDistance(before, after) {
    const previous = Array.from({length: after.length + 1}, (_, index) => index);
    for (let i = 1; i <= before.length; i += 1) {
      let diagonal = previous[0];
      previous[0] = i;
      for (let j = 1; j <= after.length; j += 1) {
        const above = previous[j];
        previous[j] = before[i - 1] === after[j - 1]
          ? diagonal
          : Math.min(diagonal, previous[j - 1], above) + 1;
        diagonal = above;
      }
    }
    return previous[after.length];
  }

  function candidateTokensAgainstTruth(candidate) {
    return wordDiff(reference.text, candidate).afterTokens.map(token => ({
      text: token.text,
      type: token.type === "added" ? "wrong" : "same",
    }));
  }

  function renderTokens(element, tokens) {
    element.replaceChildren();
    tokens.forEach((token, index) => {
      if (index > 0) element.append(" ");
      const span = document.createElement("span");
      span.textContent = token.text;
      if (token.type === "removed") span.className = "word-removed";
      if (token.type === "added") span.className = "word-added";
      if (token.type === "wrong") span.className = "word-wrong";
      element.appendChild(span);
    });
  }

  function renderLayerOverview(current) {
    const removalRuns = Array.from({length: 12}, (_, index) => runFor(current, index + 1, 0));
    const maxCtc = Math.max(...removalRuns.map(run => Number(run?.ctc_logit_divergence_from_baseline || 0)), 0);
    Array.from(layerStack.children).forEach((button, index) => {
      const layerId = index + 1;
      const run = removalRuns[index];
      const ctc = Number(run?.ctc_logit_divergence_from_baseline || 0);
      const edit = Number(run?.transcript_edit_distance_from_baseline || 0);
      const width = maxCtc > 0 ? Math.max(3, ctc / maxCtc * 100) : 0;
      button.style.setProperty("--impact", `${width}%`);
      button.setAttribute("aria-label", `L${layerId} ${LAYER_GUIDE[layerId].title}；直接去掉时字幕变化 ${edit}，CTC 分布变化 ${ctc.toFixed(4)}`);
    });
  }

  function renderLayerExplanation(run) {
    if (state.mode === "frontend") {
      const standard = frontendCondition().variants.find(item => item.frontend_variant_id === "frontend-50hz");
      const stride = Number(run.total_stride_samples);
      const rate = Number(run.actual_ctc_frame_rate_hz || run.nominal_feature_rate_hz || 0);
      const baselineCer = Number(standard.character_error_rate_vs_reference || 0);
      const currentCer = Number(run.character_error_rate_vs_reference || 0);
      const direction = currentCer < baselineCer ? "降低" : currentCer > baselineCer ? "升高" : "保持";
      const meaning = state.frontendVariantId === "frontend-50hz"
        ? "标准前端每约 320 个输入采样点产生一个时间步，约等于每 20 ms 一帧。"
        : state.frontendVariantId === "frontend-100hz"
          ? "只把最后一个卷积层的步幅从 2 改为 1，使时间帧数量约翻倍；权重不变，也没有重新训练。"
          : "把最后两个卷积层的步幅从 2 改为 1，使时间帧数量约增至四倍；更密不保证更准确。";

      $("layerGroup").textContent = "输入端 · 时间分辨率";
      $("layerTitle").textContent = `卷积前端 · ${run.frontend_label}`;
      $("layerPosition").textContent = "7 层卷积";
      $("layerRole").textContent = "卷积前端把 16 kHz 波形压缩成较低帧率的声学特征。步幅越大，计算越省，但快速语音中的多个短时变化更容易挤进同一个时间步。";
      $("residualEquation").classList.add("hidden");
      $("frontendEquation").classList.remove("hidden");
      $("frontendStrideValue").textContent = String(stride);
      $("frontendRateValue").textContent = `约 ${rate.toFixed(1)} 帧/s`;
      $("strengthMeaning").textContent = `${meaning} 本段语音相对参考文本的 CER 由 ${(baselineCer * 100).toFixed(1)}% ${direction}到 ${(currentCer * 100).toFixed(1)}%。`;
      $("layerEditValue").textContent = String(Number(run.transcript_edit_distance_from_baseline || 0));
      $("layerCtcValue").textContent = rate.toFixed(1);
      $("layerHiddenValue").textContent = `${(currentCer * 100).toFixed(1)}%`;
      $("layerEditLabel").textContent = "相对标准字幕变化";
      $("layerCtcLabel").textContent = "实际 CTC 帧/秒";
      $("layerHiddenLabel").textContent = "相对参考文本 CER";
      $("explanationBoundary").textContent = "前端加密不能恢复在语速变换中已经消失的声学细节；它只是让卷积阶段少做时间下采样。本实验未重训练，因此只能说明这条语音上的工程效果。";
      return;
    }

    const guide = LAYER_GUIDE[state.layerId];
    const alphaText = state.alpha === 1 ? "1.0" : state.alpha === 0.5 ? "0.5" : "0";
    const meaning = state.alpha === 1
      ? "α=1：保留本层计算出的全部新增量，模型按原样运行。"
      : state.alpha === 0.5
        ? "α=0.5：只保留本层新增量 Δh 的一半；不是把整层激活或模型权重除以二。"
        : "α=0：丢弃本层新增量，使输出直接等于输入，相当于旁路这一层；后续层仍照常运行。";

    $("layerGroup").textContent = guide.group;
    $("layerTitle").textContent = `L${state.layerId} · ${guide.title}`;
    $("layerPosition").textContent = `第 ${state.layerId} / 12 层`;
    $("layerRole").textContent = guide.role;
    $("residualEquation").classList.remove("hidden");
    $("frontendEquation").classList.add("hidden");
    $("alphaValue").textContent = alphaText;
    $("strengthMeaning").textContent = meaning;
    $("layerEditValue").textContent = String(Number(run.transcript_edit_distance_from_baseline || 0));
    $("layerCtcValue").textContent = Number(run.ctc_logit_divergence_from_baseline || 0).toFixed(4);
    $("layerHiddenValue").textContent = Number(run.final_hidden_cosine_distance_from_baseline || 0).toFixed(4);
    $("layerEditLabel").textContent = "字幕字符变化";
    $("layerCtcLabel").textContent = "CTC 分布变化";
    $("layerHiddenLabel").textContent = "最终表示变化";
    $("explanationBoundary").textContent = "上面的“处理重点”按网络深度解释信息流位置，不是该层独占的语言功能；数值才是本段语音上的实际消融结果。";
  }

  function render() {
    const current = condition();
    const run = selectedRun();
    if (!current || !run) throw new Error("当前组合没有真实推理结果");

    Array.from(speedButtons.children).forEach(button => {
      button.classList.toggle("active", button.dataset.condition === state.conditionId);
    });
    strengthButtons.forEach(button => {
      button.classList.toggle("active", Number(button.dataset.alpha) === state.alpha);
    });
    frontendButtons.forEach(button => {
      button.classList.toggle("active", button.dataset.frontend === state.frontendVariantId);
    });
    $("layerStrengthControl").classList.toggle("hidden", state.mode === "frontend");
    $("frontendControl").classList.toggle("hidden", state.mode !== "frontend");
    $("impactLegend").classList.toggle("hidden", state.mode === "frontend");
    $("frontendNode").classList.toggle("active", state.mode === "frontend");
    $("frontendStatus").textContent = state.mode === "frontend" ? selectedFrontendRun().frontend_label : "点击查看时间下采样";
    Array.from(layerStack.children).forEach(button => {
      const selected = Number(button.dataset.layer) === state.layerId;
      button.classList.toggle("selected", state.mode === "layer" && selected && state.alpha === 1);
      button.classList.toggle("half", state.mode === "layer" && selected && state.alpha === 0.5);
      button.classList.toggle("removed", state.mode === "layer" && selected && state.alpha === 0);
    });
    renderLayerOverview(current);
    renderLayerExplanation(run);

    const audio = $("audioPlayer");
    const nextSource = current.audio_source;
    if (!audio.src.endsWith(nextSource)) {
      audio.src = nextSource;
      audio.load();
    }
    drawWaveform(current.waveform);
    $("speedLabel").textContent = current.label;
    $("durationLabel").textContent = `${current.duration_seconds.toFixed(2)} 秒`;

    const action = state.mode === "frontend"
      ? run.frontend_label
      : state.alpha === 1 ? "保持原模型" : state.alpha === 0.5 ? `L${state.layerId} 只保留 50% 新增量` : `L${state.layerId} 新增量归零（旁路）`;
    $("interventionLabel").textContent = action;
    $("adjustedTitle").textContent = state.mode === "layer" && state.alpha === 1 ? "原模型输出" : action;
    const baselineRun = state.mode === "frontend"
      ? frontendCondition().variants.find(item => item.frontend_variant_id === "frontend-50hz")
      : current.baseline;
    $("baselineRunId").textContent = baselineRun.run_id;
    $("adjustedRunId").textContent = run.run_id;

    const baselineText = baselineRun.adjusted_transcript;
    const adjustedText = run.adjusted_transcript;
    $("referenceAudioId").textContent = reference.audio_id;
    $("referenceTranscript").textContent = reference.text;
    renderTokens($("baselineTranscript"), candidateTokensAgainstTruth(baselineText));
    renderTokens($("adjustedTranscript"), candidateTokensAgainstTruth(adjustedText));

    const editDistance = Number(run.transcript_edit_distance_from_baseline || 0);
    const baselineCer = characterEditDistance(reference.text, baselineText) / Math.max(1, reference.text.length);
    const adjustedCer = characterEditDistance(reference.text, adjustedText) / Math.max(1, reference.text.length);
    const cerDeltaPoints = (adjustedCer - baselineCer) * 100;
    $("editCount").textContent = String(editDistance);
    $("baselineCer").textContent = `${(baselineCer * 100).toFixed(1)}%`;
    $("adjustedCer").textContent = `${(adjustedCer * 100).toFixed(1)}%`;
    $("ctcDifference").textContent = Number(run.ctc_logit_divergence_from_baseline || 0).toFixed(4);
    $("cerSentence").textContent = Math.abs(cerDeltaPoints) < 0.05
      ? "相对真实文本，当前输出与标准模型的字符错误率相同。"
      : cerDeltaPoints < 0
        ? `相对真实文本，字符错误率下降 ${Math.abs(cerDeltaPoints).toFixed(1)} 个百分点。`
        : `相对真实文本，字符错误率上升 ${cerDeltaPoints.toFixed(1)} 个百分点。`;
    $("resultSentence").textContent = editDistance > 0
      ? `同一段 ${current.label} 输入，在${action}后产生了真实字幕差异。`
      : state.mode === "layer" && state.alpha === 1
        ? "当前显示同一输入的原模型结果。"
        : state.mode === "frontend" && state.frontendVariantId === "frontend-50hz"
          ? "当前显示预训练模型的标准卷积前端。"
          : "最终字幕保持一致，但内部时间分辨率和 CTC 输出仍发生变化。";
  }

  buildControls();
  window.addEventListener("resize", () => drawWaveform(condition().waveform));
  render();
})();
