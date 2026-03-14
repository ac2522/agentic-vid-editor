(function() {
  'use strict';

  window.AVE = window.AVE || {};

  var canvas = document.getElementById('preview-canvas');
  var ctx = canvas.getContext('2d');
  var seekBar = document.getElementById('seek-bar');
  var timecodeEl = document.getElementById('timecode');
  var btnPlay = document.getElementById('btn-play');
  var btnPause = document.getElementById('btn-pause');
  var ws = null;
  var durationNs = 0;
  var currentNs = 0;
  var playing = false;

  function connect() {
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(proto + '//' + location.host + '/preview/ws');

    ws.onopen = function() {
      console.log('[preview] WebSocket connected');
    };

    ws.onmessage = function(evt) {
      if (typeof evt.data === 'string') {
        var msg = JSON.parse(evt.data);
        if (msg.type === 'frame') {
          showFrame(msg.data);
          if (msg.position_ns !== undefined) {
            currentNs = msg.position_ns;
            updateTimecode();
            updateSeekBar();
            if (window.AVE.timeline) {
              window.AVE.timeline.setPlayhead(currentNs);
            }
          }
        } else if (msg.type === 'state') {
          if (msg.duration_ns !== undefined) {
            setDuration(msg.duration_ns);
          }
          playing = !!msg.playing;
          updatePlayButtons();
        }
      }
    };

    ws.onclose = function() {
      console.log('[preview] WebSocket closed, reconnecting...');
      setTimeout(connect, 2000);
    };

    ws.onerror = function() {
      ws.close();
    };
  }

  function showFrame(b64) {
    var img = new Image();
    img.onload = function() {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
    };
    img.src = 'data:image/jpeg;base64,' + b64;
  }

  function formatTimecode(ns) {
    var totalMs = Math.floor(ns / 1000000);
    var ms = totalMs % 1000;
    var totalSec = Math.floor(totalMs / 1000);
    var sec = totalSec % 60;
    var totalMin = Math.floor(totalSec / 60);
    var min = totalMin % 60;
    var hrs = Math.floor(totalMin / 60);
    return pad(hrs, 2) + ':' + pad(min, 2) + ':' + pad(sec, 2) + '.' + pad(ms, 3);
  }

  function pad(n, width) {
    var s = String(n);
    while (s.length < width) s = '0' + s;
    return s;
  }

  function updateTimecode() {
    timecodeEl.textContent = formatTimecode(currentNs);
  }

  function updateSeekBar() {
    if (durationNs > 0) {
      seekBar.value = Math.round((currentNs / durationNs) * 1000);
    }
  }

  function updatePlayButtons() {
    btnPlay.disabled = playing;
    btnPause.disabled = !playing;
  }

  function sendCommand(cmd) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(cmd));
    }
  }

  function seekTo(ns) {
    currentNs = Math.max(0, Math.min(ns, durationNs));
    updateTimecode();
    updateSeekBar();
    sendCommand({ type: 'seek', position_ns: currentNs });
    if (window.AVE.timeline) {
      window.AVE.timeline.setPlayhead(currentNs);
    }
  }

  function setDuration(ns) {
    durationNs = ns;
    updateTimecode();
  }

  // Event listeners
  seekBar.addEventListener('input', function() {
    if (durationNs > 0) {
      var ns = Math.round((parseInt(seekBar.value, 10) / 1000) * durationNs);
      seekTo(ns);
    }
  });

  btnPlay.addEventListener('click', function() {
    sendCommand({ type: 'play' });
  });

  btnPause.addEventListener('click', function() {
    sendCommand({ type: 'pause' });
  });

  // Public API
  window.AVE.preview = {
    setDuration: setDuration,
    seekTo: seekTo
  };

  connect();
})();
