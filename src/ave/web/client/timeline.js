(function() {
  'use strict';

  window.AVE = window.AVE || {};

  var canvas = document.getElementById('timeline-canvas');
  var ctx = canvas.getContext('2d');

  var HEADER_HEIGHT = 28;
  var LAYER_HEIGHT = 40;
  var LAYER_GAP = 4;
  var EFFECT_DOT_RADIUS = 3;
  var MIN_TICK_SPACING = 80;

  var COLORS = {
    video: '#4a6fa5',
    audio: '#5a9977',
    av: '#6a5acd',
    clipBorder: '#ffffff22',
    highlight: '#8be9fd',
    playhead: '#ff5555',
    header: '#1a1a2e',
    headerText: '#888',
    layerBg: '#0f0f1a',
    layerAltBg: '#121225',
    effectDot: '#50fa7b',
    labelText: '#e0e0e0'
  };

  var state = null;
  var playheadNs = 0;
  var scrollX = 0;
  var pxPerNs = 0.0000001; // initial zoom
  var selectedClipId = null;
  var refreshTimer = null;

  function fetchTimeline() {
    fetch('/api/timeline')
      .then(function(resp) { return resp.json(); })
      .then(function(data) {
        state = data;
        render();
      })
      .catch(function(err) {
        console.warn('[timeline] fetch error:', err);
      });
  }

  function refresh() {
    if (refreshTimer) clearTimeout(refreshTimer);
    refreshTimer = setTimeout(function() {
      refreshTimer = null;
      fetchTimeline();
    }, 200);
  }

  function setPlayhead(ns) {
    playheadNs = ns;
    render();
  }

  function resizeCanvas() {
    var rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * devicePixelRatio;
    canvas.height = rect.height * devicePixelRatio;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    render();
  }

  function render() {
    var w = canvas.width / devicePixelRatio;
    var h = canvas.height / devicePixelRatio;

    ctx.clearRect(0, 0, w, h);

    renderHeader(w);
    renderLayers(w, h);
    renderPlayhead(w, h);
  }

  function renderHeader(w) {
    ctx.fillStyle = COLORS.header;
    ctx.fillRect(0, 0, w, HEADER_HEIGHT);

    // Adaptive tick marks
    var tickIntervalNs = getTickInterval(w);
    var startNs = Math.floor(scrollX / tickIntervalNs) * tickIntervalNs;

    ctx.fillStyle = COLORS.headerText;
    ctx.font = '11px monospace';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = '#ffffff11';
    ctx.lineWidth = 1;

    for (var ns = startNs; ; ns += tickIntervalNs) {
      var x = (ns - scrollX) * pxPerNs;
      if (x > w) break;
      if (x < 0) continue;

      ctx.beginPath();
      ctx.moveTo(x, HEADER_HEIGHT - 6);
      ctx.lineTo(x, HEADER_HEIGHT);
      ctx.stroke();

      var label = formatTickLabel(ns);
      ctx.fillText(label, x + 3, HEADER_HEIGHT / 2);
    }
  }

  function getTickInterval(w) {
    var totalVisibleNs = w / pxPerNs;
    var candidates = [
      100000000,    // 0.1s
      250000000,    // 0.25s
      500000000,    // 0.5s
      1000000000,   // 1s
      2000000000,   // 2s
      5000000000,   // 5s
      10000000000,  // 10s
      30000000000,  // 30s
      60000000000   // 1min
    ];
    for (var i = 0; i < candidates.length; i++) {
      var pxPerTick = candidates[i] * pxPerNs;
      if (pxPerTick >= MIN_TICK_SPACING) return candidates[i];
    }
    return 60000000000;
  }

  function formatTickLabel(ns) {
    var totalSec = ns / 1000000000;
    var min = Math.floor(totalSec / 60);
    var sec = totalSec % 60;
    if (min > 0) {
      return min + ':' + (sec < 10 ? '0' : '') + sec.toFixed(sec % 1 !== 0 ? 1 : 0);
    }
    return sec.toFixed(sec % 1 !== 0 ? 1 : 0) + 's';
  }

  function renderLayers(w, h) {
    if (!state || !state.layers) return;

    var visibleStartNs = scrollX;
    var visibleEndNs = scrollX + w / pxPerNs;

    for (var i = 0; i < state.layers.length; i++) {
      var layer = state.layers[i];
      var y = HEADER_HEIGHT + i * (LAYER_HEIGHT + LAYER_GAP);

      // Layer background
      ctx.fillStyle = i % 2 === 0 ? COLORS.layerBg : COLORS.layerAltBg;
      ctx.fillRect(0, y, w, LAYER_HEIGHT);

      // Clips
      for (var j = 0; j < layer.clips.length; j++) {
        var clip = layer.clips[j];

        // Viewport culling
        var clipEnd = clip.start_ns + clip.duration_ns;
        if (clipEnd < visibleStartNs || clip.start_ns > visibleEndNs) continue;

        renderClip(clip, y, w);
      }
    }
  }

  function renderClip(clip, layerY, canvasW) {
    var x = (clip.start_ns - scrollX) * pxPerNs;
    var clipW = clip.duration_ns * pxPerNs;

    // Clip color based on track type
    var color;
    if (clip.track_types === 6) color = COLORS.av;
    else if (clip.track_types === 4) color = COLORS.video;
    else if (clip.track_types === 2) color = COLORS.audio;
    else color = COLORS.av;

    // Draw clip rectangle
    ctx.fillStyle = color;
    ctx.fillRect(x, layerY + 1, clipW, LAYER_HEIGHT - 2);

    // Highlight border for selected clip
    if (clip.clip_id === selectedClipId) {
      ctx.strokeStyle = COLORS.highlight;
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, layerY + 2, clipW - 2, LAYER_HEIGHT - 4);
    }

    // Clip name label (clipped to clip width)
    if (clipW > 30) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(x, layerY, clipW, LAYER_HEIGHT);
      ctx.clip();

      ctx.fillStyle = COLORS.labelText;
      ctx.font = '11px sans-serif';
      ctx.textBaseline = 'middle';
      ctx.fillText(clip.name, x + 4, layerY + LAYER_HEIGHT / 2);

      ctx.restore();
    }

    // Effect indicator dots
    if (clip.effects && clip.effects.length > 0) {
      for (var e = 0; e < clip.effects.length; e++) {
        ctx.beginPath();
        ctx.arc(
          x + 6 + e * (EFFECT_DOT_RADIUS * 2 + 2),
          layerY + LAYER_HEIGHT - EFFECT_DOT_RADIUS - 2,
          EFFECT_DOT_RADIUS, 0, Math.PI * 2
        );
        ctx.fillStyle = COLORS.effectDot;
        ctx.fill();
      }
    }
  }

  function renderPlayhead(w, h) {
    var x = (playheadNs - scrollX) * pxPerNs;
    if (x < 0 || x > w) return;

    ctx.strokeStyle = COLORS.playhead;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();

    // Playhead triangle at top
    ctx.fillStyle = COLORS.playhead;
    ctx.beginPath();
    ctx.moveTo(x - 5, 0);
    ctx.lineTo(x + 5, 0);
    ctx.lineTo(x, 8);
    ctx.closePath();
    ctx.fill();
  }

  // --- Hit testing ---

  function hitTestClip(canvasX, canvasY) {
    if (!state || !state.layers) return null;
    for (var i = 0; i < state.layers.length; i++) {
      var y = HEADER_HEIGHT + i * (LAYER_HEIGHT + LAYER_GAP);
      if (canvasY < y || canvasY > y + LAYER_HEIGHT) continue;
      for (var j = 0; j < state.layers[i].clips.length; j++) {
        var clip = state.layers[i].clips[j];
        var x = (clip.start_ns - scrollX) * pxPerNs;
        var w = clip.duration_ns * pxPerNs;
        if (canvasX >= x && canvasX <= x + w) return clip;
      }
    }
    return null;
  }

  // --- Event handlers ---

  canvas.addEventListener('click', function(e) {
    var rect = canvas.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;

    // Click on header = seek
    if (y < HEADER_HEIGHT) {
      var ns = scrollX + x / pxPerNs;
      if (window.AVE.preview) {
        window.AVE.preview.seekTo(ns);
      }
      return;
    }

    // Click on clip = select
    var clip = hitTestClip(x, y);
    if (clip) {
      selectedClipId = clip.clip_id;
      render();
      if (window.AVE.chat) {
        window.AVE.chat.appendToInput('[clip: ' + clip.clip_id + ' "' + clip.name + '"] ');
      }
    } else {
      selectedClipId = null;
      render();
    }
  });

  canvas.addEventListener('wheel', function(e) {
    e.preventDefault();
    if (e.ctrlKey || e.metaKey) {
      // Zoom
      var zoomFactor = e.deltaY > 0 ? 0.85 : 1.18;
      var rect = canvas.getBoundingClientRect();
      var mouseX = e.clientX - rect.left;
      var nsAtMouse = scrollX + mouseX / pxPerNs;

      pxPerNs *= zoomFactor;
      pxPerNs = Math.max(1e-11, Math.min(1e-5, pxPerNs));

      scrollX = nsAtMouse - mouseX / pxPerNs;
      if (scrollX < 0) scrollX = 0;
    } else {
      // Horizontal scroll
      scrollX += e.deltaY / pxPerNs * 0.1;
      if (scrollX < 0) scrollX = 0;
    }
    render();
  }, { passive: false });

  // --- Resize handling ---

  var resizeDebounce = null;
  window.addEventListener('resize', function() {
    if (resizeDebounce) clearTimeout(resizeDebounce);
    resizeDebounce = setTimeout(resizeCanvas, 100);
  });

  // --- Init ---

  window.AVE.timeline = {
    refresh: refresh,
    setPlayhead: setPlayhead
  };

  resizeCanvas();
  fetchTimeline();
})();
