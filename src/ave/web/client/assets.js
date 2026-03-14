(function() {
  'use strict';

  window.AVE = window.AVE || {};

  var toggleBtn = document.getElementById('assets-toggle');
  var browserEl = document.getElementById('asset-browser');
  var gridEl = document.getElementById('asset-grid');

  function formatDuration(ns) {
    var totalSec = Math.floor(ns / 1000000000);
    var min = Math.floor(totalSec / 60);
    var sec = totalSec % 60;
    return min + ':' + (sec < 10 ? '0' : '') + sec;
  }

  function toggle() {
    var hidden = browserEl.classList.toggle('hidden');
    if (!hidden) {
      fetchAssets();
    }
  }

  function fetchAssets() {
    fetch('/api/assets')
      .then(function(resp) { return resp.json(); })
      .then(function(data) {
        renderAssets(data.assets || []);
      })
      .catch(function(err) {
        console.warn('[assets] fetch error:', err);
        gridEl.innerHTML = '';
      });
  }

  function renderAssets(assets) {
    gridEl.innerHTML = '';

    if (assets.length === 0) {
      var emptyEl = document.createElement('div');
      emptyEl.className = 'asset-card';
      emptyEl.style.padding = '16px';
      emptyEl.style.textAlign = 'center';
      emptyEl.style.color = '#888';
      emptyEl.textContent = 'No assets imported yet.';
      gridEl.appendChild(emptyEl);
      return;
    }

    for (var i = 0; i < assets.length; i++) {
      var asset = assets[i];
      var card = createCard(asset);
      gridEl.appendChild(card);
    }
  }

  function createCard(asset) {
    var card = document.createElement('div');
    card.className = 'asset-card';

    var img = document.createElement('img');
    img.src = asset.thumbnail_url || '';
    img.alt = asset.name;
    img.loading = 'lazy';
    img.onerror = function() {
      this.style.display = 'none';
    };
    card.appendChild(img);

    var info = document.createElement('div');
    info.className = 'asset-info';

    var name = document.createElement('div');
    name.className = 'asset-name';
    name.textContent = asset.name;
    name.title = asset.name;
    info.appendChild(name);

    var meta = document.createElement('div');
    meta.className = 'asset-meta';
    var parts = [];
    if (asset.duration_ns) parts.push(formatDuration(asset.duration_ns));
    if (asset.resolution) parts.push(asset.resolution);
    meta.textContent = parts.join(' | ');
    info.appendChild(meta);

    card.appendChild(info);

    card.addEventListener('click', function() {
      if (window.AVE.chat) {
        window.AVE.chat.appendToInput('Add "' + asset.name + '" to the timeline. ');
      }
      browserEl.classList.add('hidden');
    });

    return card;
  }

  toggleBtn.addEventListener('click', toggle);
})();
