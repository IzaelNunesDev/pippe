/**
 * Anim8 Studio Pro — Interactive Vector Editor & Engine Client
 * ============================================================
 * Handles real-time SSE progress, canvas mounting, Bézier geometric
 * construction overlay, split-screen video/vector slider, timeline
 * scrubbing, and dynamic CSS Variable recoloring at 60 FPS.
 */

document.addEventListener('DOMContentLoaded', () => {
  // -------------------------------------------------------------------
  // STATE MANAGEMENT
  // -------------------------------------------------------------------
  const state = {
    currentJobId: null,
    jobData: null,
    currentFrame: 0,
    totalFrames: 0,
    fps: 30,
    isPlaying: false,
    isLooping: true,
    playTimer: null,
    activeView: 'svg',
    lottieAnim: null,
    lottieSpeed: 1.0,
    originalPalette: [],
    currentPalette: {},
    selectedVideoFile: null,
    isScrubbing: false,
    splitPos: 50, // 0..100%
  };

  // -------------------------------------------------------------------
  // DOM ELEMENT REFERENCES
  // -------------------------------------------------------------------
  const el = {
    // Form & Controls
    form: document.getElementById('vectorize-form'),
    dropzone: document.getElementById('dropzone'),
    fileInput: document.getElementById('video-file-input'),
    dropzoneEmpty: document.getElementById('dropzone-empty'),
    dropzoneLoaded: document.getElementById('dropzone-loaded'),
    videoName: document.getElementById('selected-video-name'),
    videoMeta: document.getElementById('selected-video-meta'),
    btnClearVideo: document.getElementById('btn-clear-video'),
    sliderColors: document.getElementById('slider-colors'),
    valColors: document.getElementById('val-colors'),
    detailBtns: document.querySelectorAll('.segment-btn'),
    inputDetail: document.getElementById('input-detail'),
    valDetail: document.getElementById('val-detail'),
    sliderSmoothing: document.getElementById('slider-smoothing'),
    valSmoothing: document.getElementById('val-smoothing'),
    sliderBudget: document.getElementById('slider-budget'),
    valBudget: document.getElementById('val-budget'),
    sliderFps: document.getElementById('slider-fps'),
    valFps: document.getElementById('val-fps'),
    checkRemoveBg: document.getElementById('check-remove-bg'),
    labelBgState: document.getElementById('label-bg-state'),
    checkUseStacking: document.getElementById('check-use-stacking'),
    btnVectorize: document.getElementById('btn-vectorize'),

    // Top actions
    btnQuickDemo: document.getElementById('btn-quick-demo'),
    btnEmptyDemo: document.getElementById('btn-empty-demo'),
    btnCopySvg: document.getElementById('btn-copy-svg'),
    btnExportSvg: document.getElementById('btn-export-svg'),
    btnExportLottie: document.getElementById('btn-export-lottie'),

    // Viewport & Views
    stageEmpty: document.getElementById('stage-empty'),
    stageLoading: document.getElementById('stage-loading'),
    loadingStageText: document.getElementById('loading-stage-text'),
    loadingProgressBar: document.getElementById('loading-progress-bar'),
    loadingPctText: document.getElementById('loading-pct-text'),
    loadingSubhint: document.getElementById('loading-subhint'),

    viewTabs: document.querySelectorAll('.view-tab'),
    viewSvg: document.getElementById('view-svg-container'),
    viewConstruction: document.getElementById('view-construction-container'),
    viewSplit: document.getElementById('view-split-container'),
    viewLottie: document.getElementById('view-lottie-container'),

    svgMount: document.getElementById('svg-mount'),
    constructionSvgMount: document.getElementById('construction-svg-mount'),
    constructionCanvas: document.getElementById('construction-canvas'),

    splitWrapper: document.getElementById('split-wrapper'),
    splitVideo: document.getElementById('split-video'),
    splitVectorWrapper: document.getElementById('split-vector-wrapper'),
    splitSvgMount: document.getElementById('split-svg-mount'),
    splitDivider: document.getElementById('split-divider'),

    lottieMount: document.getElementById('lottie-mount'),
    btnSpeed05: document.getElementById('btn-lottie-speed-05'),
    btnSpeed1: document.getElementById('btn-lottie-speed-1'),
    btnSpeed2: document.getElementById('btn-lottie-speed-2'),

    // Timeline
    timelineBar: document.getElementById('timeline-bar'),
    btnPlayPause: document.getElementById('btn-play-pause'),
    btnStepPrev: document.getElementById('btn-step-prev'),
    btnStepNext: document.getElementById('btn-step-next'),
    btnLoopToggle: document.getElementById('btn-loop-toggle'),
    scrubber: document.getElementById('timeline-scrubber'),
    timecode: document.getElementById('timeline-timecode'),
    currentFrameIdx: document.getElementById('current-frame-idx'),
    totalFramesCount: document.getElementById('total-frames-count'),
    pillFrameShapes: document.getElementById('pill-frame-shapes'),
    pillTotalShapes: document.getElementById('pill-total-shapes'),

    // Sidebar Right
    paletteGrid: document.getElementById('palette-swatches-grid'),
    btnResetPalette: document.getElementById('btn-reset-palette'),
    shapesList: document.getElementById('shapes-list'),
    inspectorShapesCount: document.getElementById('inspector-shapes-count'),

    // Metrics
    metricFps: document.getElementById('metric-fps'),
    metricResolution: document.getElementById('metric-resolution'),
    metricSvgSize: document.getElementById('metric-svg-size'),
    metricLottieSize: document.getElementById('metric-lottie-size'),
    metricStacking: document.getElementById('metric-stacking'),

    toastContainer: document.getElementById('toast-container'),
  };

  // -------------------------------------------------------------------
  // INITIAL SETUP & SLIDERS
  // -------------------------------------------------------------------
  el.sliderColors.addEventListener('input', (e) => el.valColors.textContent = e.target.value);
  el.sliderSmoothing.addEventListener('input', (e) => el.valSmoothing.textContent = e.target.value);
  el.sliderBudget.addEventListener('input', (e) => el.valBudget.textContent = e.target.value);
  el.sliderFps.addEventListener('input', (e) => el.valFps.textContent = e.target.value);

  el.detailBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      el.detailBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const val = btn.dataset.val;
      el.inputDetail.value = val;
      el.valDetail.textContent = btn.textContent;
    });
  });

  el.checkRemoveBg.addEventListener('change', (e) => {
    el.labelBgState.textContent = e.target.checked
      ? 'Desligado (Remover fundo)'
      : 'Sobre (Manter fundo)';
  });

  // -------------------------------------------------------------------
  // FILE DROPZONE
  // -------------------------------------------------------------------
  el.dropzone.addEventListener('click', (e) => {
    if (e.target !== el.btnClearVideo) el.fileInput.click();
  });

  ['dragenter', 'dragover'].forEach(name => {
    el.dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      el.dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    el.dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      el.dropzone.classList.remove('dragover');
    });
  });

  el.dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  el.fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  el.btnClearVideo.addEventListener('click', (e) => {
    e.stopPropagation();
    state.selectedVideoFile = null;
    el.fileInput.value = '';
    el.dropzoneLoaded.classList.add('hidden');
    el.dropzoneEmpty.classList.remove('hidden');
  });

  function handleFileSelected(file) {
    if (!file.type.startsWith('video/')) {
      showToast('Por favor, selecione um arquivo de vídeo válido (.mp4, .webm).', 'error');
      return;
    }
    state.selectedVideoFile = file;
    el.videoName.textContent = file.name;
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    el.videoMeta.innerHTML = `<span class="chip">${sizeMb} MB</span><span class="chip">Pronto</span>`;
    el.dropzoneEmpty.classList.add('hidden');
    el.dropzoneLoaded.classList.remove('hidden');
    showToast(`Vídeo "${file.name}" carregado. Clique em "Vetorizar Vídeo".`, 'info');
  }

  // -------------------------------------------------------------------
  // DEMO & FORM SUBMISSION
  // -------------------------------------------------------------------
  el.btnQuickDemo.addEventListener('click', () => triggerDemoVectorization());
  el.btnEmptyDemo.addEventListener('click', () => triggerDemoVectorization());

  async function triggerDemoVectorization() {
    showLoading('Gerando mascote animado procedural no servidor...');
    try {
      const params = new URLSearchParams({
        num_colors: el.sliderColors.value,
        detail_level: el.inputDetail.value,
        smoothing: el.sliderSmoothing.value,
        path_budget: el.sliderBudget.value,
        target_fps: el.sliderFps.value,
        remove_background: el.checkRemoveBg.checked,
        use_stacking: el.checkUseStacking.checked,
        use_morphing: true,
      });

      const res = await fetch(`/api/demo?${params.toString()}`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Erro ao gerar demo.');

      trackJobProgress(data.job_id);
    } catch (err) {
      hideLoading();
      showToast(err.message, 'error');
    }
  }

  el.form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!state.selectedVideoFile) {
      showToast('Selecione um arquivo de vídeo ou use a Demo.', 'error');
      return;
    }

    showLoading('Enviando vídeo para o processamento...');
    const formData = new FormData();
    formData.append('file', state.selectedVideoFile);
    formData.append('num_colors', el.sliderColors.value);
    formData.append('detail_level', el.inputDetail.value);
    formData.append('smoothing', el.sliderSmoothing.value);
    formData.append('path_budget', el.sliderBudget.value);
    formData.append('target_fps', el.sliderFps.value);
    formData.append('remove_background', el.checkRemoveBg.checked);
    formData.append('use_stacking', el.checkUseStacking.checked);
    formData.append('use_morphing', true);

    try {
      const res = await fetch('/api/vectorize', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Erro ao iniciar vetorização.');

      trackJobProgress(data.job_id);
    } catch (err) {
      hideLoading();
      showToast(err.message, 'error');
    }
  });

  // -------------------------------------------------------------------
  // SERVER-SENT EVENTS (SSE) & JOB PROGRESS
  // -------------------------------------------------------------------
  function trackJobProgress(jobId) {
    state.currentJobId = jobId;
    const evtSource = new EventSource(`/api/jobs/${jobId}/events`);

    evtSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        updateLoadingUI(data.progress, data.stage);

        if (data.status === 'completed') {
          evtSource.close();
          fetchJobResult(jobId);
        } else if (data.status === 'failed') {
          evtSource.close();
          hideLoading();
          showToast(data.stage || 'Falha no processamento.', 'error');
        }
      } catch (err) {
        console.error('SSE Parse Error:', err);
      }
    };

    evtSource.onerror = () => {
      evtSource.close();
      // Fallback para polling se SSE falhar
      pollJobStatus(jobId);
    };
  }

  async function pollJobStatus(jobId) {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        const data = await res.json();
        updateLoadingUI(data.progress, data.stage);

        if (data.status === 'completed') {
          clearInterval(interval);
          fetchJobResult(jobId);
        } else if (data.status === 'failed') {
          clearInterval(interval);
          hideLoading();
          showToast(data.error || 'Falha no processamento.', 'error');
        }
      } catch (e) {
        clearInterval(interval);
        hideLoading();
      }
    }, 500);
  }

  function showLoading(msg = 'Iniciando...') {
    el.stageEmpty.classList.add('hidden');
    el.stageLoading.classList.remove('hidden');
    updateLoadingUI(5, msg);
    el.btnVectorize.disabled = true;
  }

  function updateLoadingUI(pct, text) {
    el.loadingProgressBar.style.width = `${pct}%`;
    el.loadingPctText.textContent = `${pct}%`;
    el.loadingStageText.textContent = text;
  }

  function hideLoading() {
    el.stageLoading.classList.add('hidden');
    el.btnVectorize.disabled = false;
  }

  // -------------------------------------------------------------------
  // JOB COMPLETION & RENDER
  // -------------------------------------------------------------------
  async function fetchJobResult(jobId) {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      const data = await res.json();
      if (!res.ok || !data.result) throw new Error('Resultado incompleto.');

      state.jobData = data.result;
      state.totalFrames = data.result.frames_processed;
      state.fps = data.result.target_fps;
      state.originalPalette = [...data.result.palette_hex];
      state.currentPalette = {};
      data.result.palette_hex.forEach((hex, i) => {
        state.currentPalette[i] = hex;
      });

      hideLoading();
      initEditorUI();
      showToast(`Vetorizado com sucesso! ${data.result.total_shapes} shapes gerados.`, 'success');
    } catch (err) {
      hideLoading();
      showToast(err.message, 'error');
    }
  }

  function initEditorUI() {
    const r = state.jobData;

    // Ativa botões de exportação
    el.btnCopySvg.disabled = false;
    el.btnExportSvg.disabled = false;
    el.btnExportLottie.disabled = false;

    // Configura Scrubber
    el.scrubber.max = state.totalFrames - 1;
    el.scrubber.value = 0;
    state.currentFrame = 0;
    el.totalFramesCount.textContent = state.totalFrames;
    el.currentFrameIdx.textContent = '0';
    el.pillTotalShapes.textContent = r.total_shapes;
    el.pillFrameShapes.textContent = r.frame_stats[0] ? r.frame_stats[0].shapes_count : 0;

    // Monta SVG
    mountSvgContent(r.svg_content);

    // Carrega vídeo original no Split
    el.splitVideo.src = `/api/jobs/${state.currentJobId}/video`;

    // Prepara Lottie
    initLottiePlayer(r.lottie_content);

    // Renderiza Paleta e Métricas
    renderColorPalette();
    renderInspectorShapes(0);
    updateMetrics();

    // Mostra View Padrão
    switchView(state.activeView || 'svg');

    // Inicia reprodução
    startPlayback();
  }

  // -------------------------------------------------------------------
  // SVG MOUNT & TIME SCRUBBING
  // -------------------------------------------------------------------
  function mountSvgContent(svgString) {
    el.svgMount.innerHTML = svgString;
    el.splitSvgMount.innerHTML = svgString;
    el.constructionSvgMount.innerHTML = svgString;

    // Ajusta viewBox e dimensões
    const svgEl = el.svgMount.querySelector('svg');
    if (svgEl) {
      applyPaletteToSvg(svgEl);
    }
    const splitSvg = el.splitSvgMount.querySelector('svg');
    if (splitSvg) applyPaletteToSvg(splitSvg);
    const constSvg = el.constructionSvgMount.querySelector('svg');
    if (constSvg) applyPaletteToSvg(constSvg);
  }

  function applyPaletteToSvg(svgEl) {
    Object.entries(state.currentPalette).forEach(([idx, hex]) => {
      svgEl.style.setProperty(`--color-${idx}`, hex);
    });
  }

  function renderFrame(frameIdx) {
    state.currentFrame = frameIdx;
    el.currentFrameIdx.textContent = frameIdx;
    el.scrubber.value = frameIdx;

    // Atualiza timecode
    const curSec = (frameIdx / state.fps).toFixed(2);
    const totSec = (state.totalFrames / state.fps).toFixed(2);
    el.timecode.textContent = `00:${curSec.padStart(5, '0')} / 00:${totSec.padStart(5, '0')}`;

    // Atualiza contagem de shapes do frame atual
    if (state.jobData && state.jobData.frame_stats[frameIdx]) {
      el.pillFrameShapes.textContent = state.jobData.frame_stats[frameIdx].shapes_count;
    }

    // Controle manual dos frames no SVG (desativa CSS animation nativa para controle quadro a quadro)
    [el.svgMount, el.splitSvgMount, el.constructionSvgMount].forEach(mount => {
      const svg = mount.querySelector('svg');
      if (!svg) return;
      const groups = svg.querySelectorAll('.anim8-frame');
      groups.forEach((g, idx) => {
        if (idx === frameIdx) {
          g.style.display = 'inline';
          g.style.animation = 'none';
        } else {
          g.style.display = 'none';
          g.style.animation = 'none';
        }
      });
    });

    // Se estiver no split, sincroniza o vídeo MP4 original
    if (el.splitVideo && el.splitVideo.duration) {
      el.splitVideo.currentTime = frameIdx / state.fps;
    }

    // Se a view de construção estiver ativa, redesenha os nós geométricos
    if (state.activeView === 'construction') {
      drawConstructionCanvas(frameIdx);
    }

    // Atualiza a lista do inspetor lateral
    renderInspectorShapes(frameIdx);
  }

  // -------------------------------------------------------------------
  // GEOMETRIC CONSTRUCTION & BÉZIER NODES OVERLAY
  // -------------------------------------------------------------------
  function drawConstructionCanvas(frameIdx) {
    const canvas = el.constructionCanvas;
    const mountBg = el.constructionSvgMount.querySelector('svg');
    if (!canvas || !mountBg || !state.jobData) return;

    const [w, h] = state.jobData.canvas_size;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    const frameShapes = state.jobData.shapes_by_frame[frameIdx] || [];

    frameShapes.forEach(shape => {
      const [bx, by, bw, bh] = shape.bbox;
      const [cx, cy] = shape.centroid;

      // 1. Círculos Geométricos Áureos (Harmonic circles)
      ctx.beginPath();
      ctx.arc(cx, cy, Math.max(bw, bh) * 0.45, 0, 2 * Math.PI);
      ctx.strokeStyle = 'rgba(139, 92, 246, 0.22)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // 2. Curvas de Bézier: Vértices (P0, P3) e Alças de Controle (P1, P2)
      shape.anchors.forEach(sub => {
        sub.forEach(seg => {
          const [p0, p1, p2, p3] = seg;

          // Linhas das alças de controle (P0 -> P1 e P3 -> P2)
          ctx.beginPath();
          ctx.moveTo(p0[0], p0[1]);
          ctx.lineTo(p1[0], p1[1]);
          ctx.moveTo(p3[0], p3[1]);
          ctx.lineTo(p2[0], p2[1]);
          ctx.strokeStyle = 'rgba(0, 240, 255, 0.45)';
          ctx.lineWidth = 1;
          ctx.stroke();

          // Alças de controle P1 e P2 (Pontos cianos)
          ctx.fillStyle = '#00F0FF';
          ctx.beginPath();
          ctx.arc(p1[0], p1[1], 2.2, 0, 2 * Math.PI);
          ctx.arc(p2[0], p2[1], 2.2, 0, 2 * Math.PI);
          ctx.fill();

          // Vértices âncora P0 e P3 (Pontos brancos com brilho)
          ctx.fillStyle = '#FFFFFF';
          ctx.shadowColor = '#FFFFFF';
          ctx.shadowBlur = 4;
          ctx.beginPath();
          ctx.arc(p0[0], p0[1], 3.2, 0, 2 * Math.PI);
          ctx.arc(p3[0], p3[1], 3.2, 0, 2 * Math.PI);
          ctx.fill();
          ctx.shadowBlur = 0;
        });
      });
    });
  }

  // -------------------------------------------------------------------
  // PLAYBACK & TIMELINE CONTROLS
  // -------------------------------------------------------------------
  el.btnPlayPause.addEventListener('click', togglePlayback);
  el.btnLoopToggle.addEventListener('click', () => {
    state.isLooping = !state.isLooping;
    el.btnLoopToggle.classList.toggle('active', state.isLooping);
  });

  el.btnStepPrev.addEventListener('click', () => {
    pausePlayback();
    const prev = (state.currentFrame - 1 + state.totalFrames) % state.totalFrames;
    renderFrame(prev);
  });

  el.btnStepNext.addEventListener('click', () => {
    pausePlayback();
    const next = (state.currentFrame + 1) % state.totalFrames;
    renderFrame(next);
  });

  el.scrubber.addEventListener('input', (e) => {
    pausePlayback();
    renderFrame(parseInt(e.target.value, 10));
  });

  // Tecla de Espaço para Play/Pause
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target.tagName !== 'INPUT' && state.totalFrames > 0) {
      e.preventDefault();
      togglePlayback();
    }
  });

  function togglePlayback() {
    if (state.isPlaying) {
      pausePlayback();
    } else {
      startPlayback();
    }
  }

  function startPlayback() {
    if (state.totalFrames <= 0) return;
    state.isPlaying = true;
    el.btnPlayPause.textContent = '⏸';
    el.btnPlayPause.classList.add('active');

    const frameDurationMs = 1000 / state.fps;
    clearInterval(state.playTimer);
    state.playTimer = setInterval(() => {
      let next = state.currentFrame + 1;
      if (next >= state.totalFrames) {
        if (state.isLooping) {
          next = 0;
        } else {
          pausePlayback();
          return;
        }
      }
      renderFrame(next);
    }, frameDurationMs);
  }

  function pausePlayback() {
    state.isPlaying = false;
    clearInterval(state.playTimer);
    el.btnPlayPause.textContent = '▶';
    el.btnPlayPause.classList.remove('active');
  }

  // -------------------------------------------------------------------
  // VIEW TABS & SWITCHING
  // -------------------------------------------------------------------
  el.viewTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      el.viewTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      switchView(tab.dataset.view);
    });
  });

  function switchView(viewName) {
    state.activeView = viewName;

    el.viewSvg.classList.add('hidden');
    el.viewConstruction.classList.add('hidden');
    el.viewSplit.classList.add('hidden');
    el.viewLottie.classList.add('hidden');

    if (viewName === 'svg') {
      el.viewSvg.classList.remove('hidden');
    } else if (viewName === 'construction') {
      el.viewConstruction.classList.remove('hidden');
      drawConstructionCanvas(state.currentFrame);
    } else if (viewName === 'split') {
      el.viewSplit.classList.remove('hidden');
      initSplitSlider();
    } else if (viewName === 'lottie') {
      el.viewLottie.classList.remove('hidden');
      if (state.lottieAnim) state.lottieAnim.play();
    }

    if (state.totalFrames > 0) {
      renderFrame(state.currentFrame);
    }
  }

  // -------------------------------------------------------------------
  // SPLIT-SCREEN COMPARISON SLIDER
  // -------------------------------------------------------------------
  function initSplitSlider() {
    let isDragging = false;

    function updateSplit(clientX) {
      const rect = el.splitWrapper.getBoundingClientRect();
      let offsetX = clientX - rect.left;
      let pct = (offsetX / rect.width) * 100;
      pct = Math.max(0, Math.min(100, pct));
      state.splitPos = pct;

      el.splitVectorWrapper.style.width = `${pct}%`;
      el.splitDivider.style.left = `${pct}%`;
    }

    el.splitDivider.onmousedown = (e) => {
      isDragging = true;
      e.preventDefault();
    };

    window.onmousemove = (e) => {
      if (isDragging) updateSplit(e.clientX);
    };

    window.onmouseup = () => {
      isDragging = false;
    };
  }

  // -------------------------------------------------------------------
  // LOTTIE INTEGRATION (lottie-web)
  // -------------------------------------------------------------------
  function initLottiePlayer(lottieJsonString) {
    if (!window.bodymovin) return;

    if (state.lottieAnim) {
      state.lottieAnim.destroy();
    }

    try {
      const animData = JSON.parse(lottieJsonString);
      el.lottieMount.innerHTML = '';

      state.lottieAnim = window.bodymovin.loadAnimation({
        container: el.lottieMount,
        renderer: 'svg',
        loop: true,
        autoplay: true,
        animationData: animData
      });
    } catch (e) {
      console.warn('Erro ao inicializar Lottie:', e);
    }
  }

  el.btnSpeed05.addEventListener('click', () => setLottieSpeed(0.5, el.btnSpeed05));
  el.btnSpeed1.addEventListener('click', () => setLottieSpeed(1.0, el.btnSpeed1));
  el.btnSpeed2.addEventListener('click', () => setLottieSpeed(2.0, el.btnSpeed2));

  function setLottieSpeed(speed, activeBtn) {
    [el.btnSpeed05, el.btnSpeed1, el.btnSpeed2].forEach(b => b.classList.remove('active'));
    activeBtn.classList.add('active');
    state.lottieSpeed = speed;
    if (state.lottieAnim) state.lottieAnim.setSpeed(speed);
  }

  // -------------------------------------------------------------------
  // COLOR PALETTE & INSTANT RECOLORING (60 FPS via CSS Variables)
  // -------------------------------------------------------------------
  function renderColorPalette() {
    el.paletteGrid.innerHTML = '';
    el.btnResetPalette.classList.remove('hidden');

    Object.entries(state.currentPalette).forEach(([idx, hex]) => {
      const card = document.createElement('div');
      card.className = 'color-swatch-card';
      card.title = `Trocar Cor #${idx} em toda a animação`;

      const box = document.createElement('div');
      box.className = 'swatch-color-box';
      box.style.backgroundColor = hex;

      const label = document.createElement('span');
      label.className = 'swatch-hex';
      label.textContent = hex.toUpperCase();

      const picker = document.createElement('input');
      picker.type = 'color';
      picker.value = hex;
      picker.className = 'hidden-color-picker';

      picker.addEventListener('input', (e) => {
        const newColor = e.target.value;
        box.style.backgroundColor = newColor;
        label.textContent = newColor.toUpperCase();
        state.currentPalette[idx] = newColor;

        // Atualização instantânea em 60 FPS direto no DOM de todos os SVGs!
        [el.svgMount, el.splitSvgMount, el.constructionSvgMount].forEach(mount => {
          const svg = mount.querySelector('svg');
          if (svg) svg.style.setProperty(`--color-${idx}`, newColor);
        });
      });

      card.appendChild(box);
      card.appendChild(label);
      card.appendChild(picker);
      el.paletteGrid.appendChild(card);
    });
  }

  el.btnResetPalette.addEventListener('click', () => {
    state.originalPalette.forEach((hex, i) => {
      state.currentPalette[i] = hex;
    });
    renderColorPalette();
    [el.svgMount, el.splitSvgMount, el.constructionSvgMount].forEach(mount => {
      const svg = mount.querySelector('svg');
      if (svg) applyPaletteToSvg(svg);
    });
    showToast('Paleta resetada para as cores originais da engine.', 'info');
  });

  // -------------------------------------------------------------------
  // SHAPE INSPECTOR LIST
  // -------------------------------------------------------------------
  function renderInspectorShapes(frameIdx) {
    if (!state.jobData || !state.jobData.shapes_by_frame[frameIdx]) {
      el.shapesList.innerHTML = '<div class="inspector-placeholder">Sem shapes neste frame</div>';
      el.inspectorShapesCount.textContent = '0';
      return;
    }

    const shapes = state.jobData.shapes_by_frame[frameIdx];
    el.inspectorShapesCount.textContent = shapes.length;
    el.shapesList.innerHTML = '';

    shapes.slice(0, 50).forEach((shape, sIdx) => {
      const card = document.createElement('div');
      card.className = 'shape-item-card';

      const colorHex = state.currentPalette[shape.color_index] || shape.hex_color;

      card.innerHTML = `
        <div class="shape-left">
          <span class="shape-color-dot" style="background-color: ${colorHex}"></span>
          <span class="shape-title">Track #${shape.track_id}</span>
        </div>
        <span class="shape-area">${Math.round(shape.area)} px²</span>
      `;

      // Hover para destacar o path no SVG
      card.addEventListener('mouseenter', () => {
        const pathEl = el.svgMount.querySelector(`#f${frameIdx}-s${sIdx}`);
        if (pathEl) {
          pathEl.style.stroke = '#00F0FF';
          pathEl.style.strokeWidth = '2.5px';
        }
      });
      card.addEventListener('mouseleave', () => {
        const pathEl = el.svgMount.querySelector(`#f${frameIdx}-s${sIdx}`);
        if (pathEl) {
          pathEl.style.stroke = 'none';
        }
      });

      el.shapesList.appendChild(card);
    });
  }

  // -------------------------------------------------------------------
  // METRICS SUMMARY
  // -------------------------------------------------------------------
  function updateMetrics() {
    const r = state.jobData;
    el.metricFps.textContent = `${r.target_fps} FPS`;
    el.metricResolution.textContent = `${r.canvas_size[0]} × ${r.canvas_size[1]}`;

    const svgBytes = new Blob([r.svg_content]).size;
    const lottieBytes = new Blob([r.lottie_content]).size;

    el.metricSvgSize.textContent = `${(svgBytes / 1024).toFixed(1)} KB`;
    el.metricLottieSize.textContent = `${(lottieBytes / 1024).toFixed(1)} KB`;
    el.metricStacking.textContent = r.use_stacking ? 'Ativado (Anti-Frestas)' : 'Desativado';
  }

  // -------------------------------------------------------------------
  // EXPORT & DOWNLOADS
  // -------------------------------------------------------------------
  el.btnExportSvg.addEventListener('click', () => {
    if (!state.currentJobId) return;
    window.location.href = `/api/jobs/${state.currentJobId}/svg`;
    showToast('Download do SVG animado iniciado.', 'success');
  });

  el.btnExportLottie.addEventListener('click', () => {
    if (!state.currentJobId) return;
    window.location.href = `/api/jobs/${state.currentJobId}/lottie`;
    showToast('Download do Lottie JSON iniciado.', 'success');
  });

  el.btnCopySvg.addEventListener('click', () => {
    if (!state.jobData) return;
    navigator.clipboard.writeText(state.jobData.svg_content)
      .then(() => showToast('Código SVG copiado para a área de transferência!', 'success'))
      .catch(() => showToast('Falha ao copiar código.', 'error'));
  });

  // -------------------------------------------------------------------
  // TOAST NOTIFICATIONS
  // -------------------------------------------------------------------
  function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ';
    toast.innerHTML = `<span>${icon}</span> <span>${message}</span>`;
    el.toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transition = 'opacity 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, 3800);
  }
});
