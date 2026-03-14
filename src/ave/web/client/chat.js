(function() {
  'use strict';

  window.AVE = window.AVE || {};

  var messagesEl = document.getElementById('chat-messages');
  var inputEl = document.getElementById('chat-input');
  var sendBtn = document.getElementById('chat-send');
  var ws = null;
  var currentAgentBubble = null;
  var processing = false;

  function connect() {
    var proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    var url = proto + '//' + location.host + '/ws/chat';
    var token = sessionStorage.getItem('ave_session_token');
    if (token) {
      url += '?session=' + encodeURIComponent(token);
    }

    ws = new WebSocket(url);

    ws.onopen = function() {
      console.log('[chat] WebSocket connected');
    };

    ws.onmessage = function(evt) {
      var msg;
      try {
        msg = JSON.parse(evt.data);
      } catch (e) {
        return;
      }

      switch (msg.type) {
        case 'connected':
          if (msg.session_token) {
            sessionStorage.setItem('ave_session_token', msg.session_token);
          }
          break;

        case 'text_delta':
          if (!currentAgentBubble) {
            currentAgentBubble = addMessage('', 'agent');
          }
          currentAgentBubble.textContent += msg.text;
          scrollToBottom();
          break;

        case 'tool_start':
          addMessage('Running ' + msg.tool_name + '...', 'tool');
          break;

        case 'tool_done':
          // No-op for now; tool completion is implicit from next text/done
          break;

        case 'timeline_updated':
          if (window.AVE.timeline) {
            window.AVE.timeline.refresh();
          }
          break;

        case 'done':
          currentAgentBubble = null;
          setProcessing(false);
          break;

        case 'error':
          addMessage(msg.message || 'An error occurred.', 'error');
          currentAgentBubble = null;
          setProcessing(false);
          break;

        case 'busy':
          addMessage('Agent is busy, please wait...', 'busy');
          break;
      }
    };

    ws.onclose = function() {
      console.log('[chat] WebSocket closed, reconnecting in 2s...');
      setProcessing(false);
      setTimeout(connect, 2000);
    };

    ws.onerror = function() {
      ws.close();
    };
  }

  function addMessage(text, variant) {
    var el = document.createElement('div');
    el.className = 'msg ' + variant;
    el.textContent = text;
    messagesEl.appendChild(el);
    scrollToBottom();
    return el;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function setProcessing(busy) {
    processing = busy;
    sendBtn.disabled = busy;
    inputEl.disabled = busy;
    if (!busy) {
      inputEl.focus();
    }
  }

  function sendMessage() {
    var text = inputEl.value.trim();
    if (!text || processing) return;

    addMessage(text, 'user');
    inputEl.value = '';
    currentAgentBubble = null;
    setProcessing(true);

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'chat', text: text }));
    }
  }

  function appendToInput(text) {
    inputEl.value = text + inputEl.value;
    inputEl.focus();
  }

  // Event listeners
  sendBtn.addEventListener('click', sendMessage);

  inputEl.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Public API
  window.AVE.chat = {
    appendToInput: appendToInput
  };

  connect();
})();
