# Phase 8: Web UI + Integration — Design Specification

> Designed 2026-03-14. Builds on architecture-design-v2 and full-roadmap-design.
> Depends on: Phase 3 (Preview), Phase 4 (Agent/Tool Registry) — both complete.
> Revised after architectural review addressing: timeline data model, Agent SDK integration,
> concurrency, session recovery, route conflicts, and security.

---

## Goal

Provide a browser-based interface for non-technical users to edit video through natural language. The chat is the primary input — the agent does the editing. A read-only timeline visualization shows the current state. The existing preview system provides playback and scrubbing. An asset browser lets users see and reference ingested media.

This interface serves two user profiles equally:
- **Quick creator:** "Take my hiking clips, make a 15-second Instagram reel with captions"
- **Long-form editor:** "Import the interview, remove filler words, add B-roll, color correct, export for YouTube"

Both interact through chat. The timeline visualization helps users *see* what the agent built and reference specific elements.

---

## Architecture Decision: Chat-First with Inline Timeline

### Why not a full NLE-style UI

The project vision states: *"The interface follows the Claude Code CLI model: a chat/prompt interface with a lightweight preview panel."* A drag-and-drop NLE contradicts this — the agent handles edits, not the user's mouse. A full NLE would require a React/Vue SPA, a build toolchain, and weeks of frontend work for functionality the agent already provides.

### Why vanilla HTML/JS (no framework)

- The existing preview client (`src/ave/preview/client/index.html`) uses vanilla JS
- No build step required (no npm/webpack/vite)
- The UI is simple: 3 panels + a text input
- Fewer dependencies = fewer failure modes in a Docker environment
- The project's value is in the agent and tools, not the frontend framework

### Why a single aiohttp server

- The preview server already uses aiohttp
- Adding chat WebSocket + REST API to the same server avoids port management
- All state (session, timeline, assets) lives in one process
- Simplifies deployment: one Docker container, one port

### Security: localhost-only binding

The server binds to `127.0.0.1` by default. The agent can execute tools that read/write files and run FFmpeg commands — network exposure would be a security risk. The `--host` flag allows override for Docker (where `0.0.0.0` is needed inside the container, with Docker port mapping providing isolation).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser (vanilla JS)                      │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Chat Panel │  │ Preview      │  │ Asset Browser        │ │
│  │ (WebSocket)│  │ (WebSocket + │  │ (REST)               │ │
│  │            │  │  HTTP video) │  │                      │ │
│  └─────┬──────┘  └──────┬───────┘  └──────────┬───────────┘ │
│        │                │                      │             │
│  ┌─────┴────────────────┴──────────────────────┴───────────┐ │
│  │              Timeline Visualization (Canvas)             │ │
│  │              (REST, refreshed on timeline_updated)       │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    aiohttp server (127.0.0.1:8080)
                           │
        ┌──────────────────┼──────────────────────┐
        │                  │                      │
  ┌─────┴─────┐    ┌──────┴──────┐    ┌──────────┴─────────┐
  │ /ws/chat   │    │ /preview/ws │    │ /api/timeline      │
  │ ChatHandler│    │ (preview    │    │ /api/assets        │
  │            │    │  sub-app)   │    │ /api/assets/{id}/  │
  │ Anthropic  │    │             │    │   thumbnail        │
  │ SDK stream │    │             │    │                    │
  └─────┬──────┘    └─────────────┘    └──────────┬─────────┘
        │                                         │
  ┌─────┴──────────┐                    ┌─────────┴─────────┐
  │ EditingSession  │                    │ TimelineModel     │
  │ + Orchestrator  │                    │ (pure Python,     │
  │ (existing)      │                    │  no GES required) │
  └─────┬──────────┘                    └───────────────────┘
        │
  ┌─────┴──────────┐
  │ Tool Registry   │
  │ (11 domains,    │
  │  40+ tools)     │
  └────────────────┘
```

---

## Module Layout

```
src/ave/web/
├── __init__.py
├── app.py              # App factory: create_app(), route mounting
├── chat.py             # Chat WebSocket handler + Anthropic SDK bridge
├── api.py              # REST endpoints: timeline state, asset listing
├── timeline_model.py   # Pure Python timeline state (no GES dependency)
└── client/
    ├── index.html      # Single-page layout (3-panel)
    ├── styles.css      # Dark theme
    ├── chat.js         # Chat WebSocket client, streaming text display
    ├── timeline.js     # Canvas timeline renderer (read-only)
    ├── preview.js      # Preview player (refactored from existing client)
    └── assets.js       # Asset browser panel
```

---

## Component Details

### 1. TimelineModel — Pure Python Timeline State (`timeline_model.py`)

**This is the central data model the entire UI depends on.**

The existing `EditingSession` does not maintain a timeline model — it tracks tool call history and provisions, not clip positions. The `/api/timeline` endpoint needs a queryable state object.

`TimelineModel` is a pure Python class (no GES dependency) that accumulates timeline state from two sources:

1. **XGES file parsing:** On project load, parse the `.xges` XML file to extract layers, clips, effects. XGES is standard XML — no GES bindings needed. The `xml.etree.ElementTree` module handles this.

2. **Tool call results:** When tools modify the timeline (trim, split, concatenate, etc.), the `TimelineModel` is updated from the tool's return value. Each editing tool already returns computed parameters (new start, duration, etc.).

```python
@dataclass
class ClipState:
    """A clip on the timeline."""
    id: str
    name: str
    layer_index: int
    start_ns: int
    duration_ns: int
    in_point_ns: int
    effects: list[str]
    media_type: str  # "video", "audio", "av"

@dataclass
class LayerState:
    """A layer containing clips."""
    index: int
    clips: list[ClipState]

class TimelineModel:
    """Pure Python timeline state. No GES dependency."""

    def __init__(self, fps: float = 24.0) -> None:
        self._layers: list[LayerState] = []
        self._fps = fps

    def load_from_xges(self, xges_path: Path) -> None:
        """Parse XGES XML to populate timeline state."""
        # Uses xml.etree.ElementTree — no GES bindings needed

    def add_clip(self, clip: ClipState) -> None: ...
    def remove_clip(self, clip_id: str) -> None: ...
    def update_clip(self, clip_id: str, **kwargs) -> None: ...

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict for /api/timeline."""

    @property
    def duration_ns(self) -> int:
        """Total timeline duration (max clip end)."""
```

**Why not use GES objects directly:** GES requires GObject Introspection and `gi.repository.GES`, which is only available inside the Docker container. The web layer must work without GES for local development and testing. Parsing XGES XML is trivial (it's well-structured XML) and provides all the data needed for visualization.

**Sync strategy:** After each tool call that returns timeline-modifying results, the `ChatSession` calls `timeline_model.update_from_tool_result(tool_name, result)`. Additionally, the model can be re-synced from the XGES file at any time (idempotent).

### 2. App Factory (`app.py`)

Creates the unified aiohttp application:

```python
def create_app(
    project_dir: Path,
    xges_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> web.Application:
    """Create the AVE web application.

    Integrates: chat WebSocket, preview sub-app, REST API, static client.
    Binds to localhost by default for security.
    """
```

Responsibilities:
- Create `EditingSession` (existing class — loads all 11 domain tool registrations)
- Create `Orchestrator` (existing class — wraps session with meta-tools)
- Create `TimelineModel` (new — populated from XGES if provided)
- **Mount preview server as sub-app at `/preview/`** (avoids route collision on `/`)
- Mount chat WebSocket at `/ws/chat`
- Mount REST API at `/api/`
- Serve static client files at `/`
- Store shared state in `app` dict

**Route collision fix:** The existing `PreviewServer.create_app()` creates its own `web.Application` with routes at `/`, `/ws`, `/api/status`. Instead of mounting it at root (which conflicts with the new UI), mount it as a sub-application at `/preview/`. The preview WebSocket becomes `/preview/ws`, segments at `/preview/segments/`, status at `/preview/api/status`. The client's `preview.js` connects to these prefixed URLs.

### 3. Chat Handler (`chat.py`)

Bridges browser WebSocket ↔ Anthropic Python SDK streaming.

**Client → Server protocol:**
```json
{"type": "message", "text": "Remove all the umms from the interview"}
{"type": "cancel"}
```

**Server → Client protocol:**
```json
{"type": "text_delta", "text": "I'll search for filler words..."}
{"type": "tool_start", "tool_name": "search_tools", "tool_id": "tc_01"}
{"type": "tool_input_delta", "tool_id": "tc_01", "json_chunk": "..."}
{"type": "tool_done", "tool_id": "tc_01"}
{"type": "timeline_updated"}
{"type": "done", "turn_id": 1}
{"type": "error", "message": "Tool prerequisite not met: timeline_loaded"}
{"type": "busy", "message": "Agent is processing, please wait"}
```

#### Agent SDK Integration (concrete design)

The chat handler does NOT use `claude-agent-sdk`. Instead, it uses the **Anthropic Python SDK** (`anthropic` package) directly with streaming, which gives full control over the tool-use loop. This avoids dependency on a separate agent SDK and aligns with how the existing Orchestrator already works.

```python
class ChatSession:
    """Bridges WebSocket ↔ Anthropic API with tool-use loop."""

    def __init__(self, orchestrator: Orchestrator, timeline_model: TimelineModel):
        self._orchestrator = orchestrator
        self._timeline = timeline_model
        self._messages: list[dict] = []  # Conversation history
        self._processing = False
        self._cancel_event = asyncio.Event()

    def _get_tools_json(self) -> list[dict]:
        """Convert Orchestrator meta-tools to Anthropic API tool format."""
        return [
            {
                "name": mt.name,
                "description": mt.description,
                "input_schema": mt.parameters,
            }
            for mt in self._orchestrator.get_meta_tools()
        ]

    async def handle_message(self, ws: web.WebSocketResponse, text: str) -> None:
        """Process user message through the agentic loop."""
        if self._processing:
            await ws.send_json({"type": "busy", "message": "Agent is processing"})
            return

        self._processing = True
        self._cancel_event.clear()

        try:
            await self._agentic_loop(ws, text)
        finally:
            self._processing = False

    async def _agentic_loop(self, ws: web.WebSocketResponse, text: str) -> None:
        """Run the tool-use loop: send message → get response → execute tools → repeat."""
        self._messages.append({"role": "user", "content": text})

        loop = asyncio.get_running_loop()
        client = anthropic.AsyncAnthropic()

        while True:
            if self._cancel_event.is_set():
                await ws.send_json({"type": "done", "turn_id": self._orchestrator.turn_count})
                return

            # Stream the response
            tool_calls = []
            async with client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=self._orchestrator.get_system_prompt(),
                tools=self._get_tools_json(),
                messages=self._messages,
            ) as stream:
                async for event in stream:
                    if self._cancel_event.is_set():
                        break
                    await self._forward_event(ws, event, tool_calls)

            # Get the final message
            response = await stream.get_final_message()
            self._messages.append({"role": "assistant", "content": response.content})

            # If no tool use, we're done
            if response.stop_reason != "tool_use":
                await ws.send_json({"type": "done", "turn_id": self._orchestrator.turn_count})
                return

            # Execute tool calls and build tool results
            tool_results = []
            for tc in tool_calls:
                # Run synchronous tool handler in executor to avoid blocking event loop
                result_str = await loop.run_in_executor(
                    None,
                    self._orchestrator.handle_tool_call,
                    tc["name"],
                    tc["input"],
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result_str,
                })
                await ws.send_json({"type": "tool_done", "tool_id": tc["id"]})

                # Check if this tool modifies timeline
                if self._is_timeline_modifying(tc["name"], tc["input"]):
                    self._timeline.reload_from_xges()
                    await ws.send_json({"type": "timeline_updated"})

            self._messages.append({"role": "user", "content": tool_results})
            # Loop continues — agent sees tool results, may call more tools

    async def _forward_event(self, ws, event, tool_calls: list) -> None:
        """Forward streaming event to WebSocket client."""
        if hasattr(event, 'type'):
            if event.type == 'text':
                await ws.send_json({"type": "text_delta", "text": event.text})
            elif event.type == 'input_json':
                # Tool input streaming
                await ws.send_json({
                    "type": "tool_input_delta",
                    "tool_id": tool_calls[-1]["id"] if tool_calls else "",
                    "json_chunk": event.partial_json,
                })
            elif event.type == 'content_block_start':
                if hasattr(event, 'content_block') and event.content_block.type == 'tool_use':
                    tc = {"id": event.content_block.id, "name": event.content_block.name, "input": {}}
                    tool_calls.append(tc)
                    await ws.send_json({
                        "type": "tool_start",
                        "tool_name": event.content_block.name,
                        "tool_id": event.content_block.id,
                    })

    def _is_timeline_modifying(self, tool_name: str, tool_input: dict) -> bool:
        """Check if a meta-tool call modifies the timeline."""
        if tool_name != "call_tool":
            return False
        inner = tool_input.get("tool_name", "")
        modifying_domains = {"editing", "compositing", "motion_graphics", "scene"}
        schema = self._orchestrator.session.registry.get_tool_schema(inner)
        return schema.domain in modifying_domains if schema else False

    def cancel(self) -> None:
        """Signal cancellation of current processing."""
        self._cancel_event.set()
```

**Key design decisions:**
- Uses `anthropic.AsyncAnthropic` directly (not claude-agent-sdk) for full control
- Tool calls run in `run_in_executor()` since `Orchestrator.handle_tool_call()` is synchronous and some tools (FFmpeg, transcription) block for seconds/minutes
- Concurrency: `self._processing` flag rejects concurrent messages with `busy` event
- Cancellation: `asyncio.Event` checked between iterations
- Conversation history: `self._messages` list persists across turns within a session

#### Concurrency Control

The `ChatSession._processing` boolean ensures only one agentic loop runs at a time:
- If a message arrives while processing, the handler responds with `{"type": "busy"}`
- The client disables the input field on send and re-enables on `done` or `error`
- The `cancel` message sets `self._cancel_event`, which is checked between tool calls and during streaming

#### Session Recovery on Reconnect

`ChatSession` instances are stored in a server-side dict keyed by session token:

```python
# In app.py
app["sessions"] = {}  # session_token -> ChatSession

# In chat WebSocket handler
async def handle_ws_chat(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # Session token from query param, or generate new
    token = request.query.get("session", str(uuid.uuid4()))

    if token in request.app["sessions"]:
        session = request.app["sessions"][token]
    else:
        session = ChatSession(
            orchestrator=Orchestrator(request.app["editing_session"]),
            timeline_model=request.app["timeline_model"],
        )
        request.app["sessions"][token] = session

    await ws.send_json({"type": "connected", "session": token})
    # ... message loop ...
```

On reconnect:
- Client reconnects with the same session token (stored in sessionStorage)
- `ChatSession` is retrieved — conversation history (`self._messages`) is preserved
- The agent resumes with full context of prior turns
- `EditingSession` state (provisions, tool history) is app-level, always survives reconnects

For the long-form editor persona: 20 messages into a complex edit, WiFi blips, client reconnects with the same token, agent conversation history is intact.

### 4. REST API (`api.py`)

**`GET /api/timeline`**

Serializes `TimelineModel.to_dict()`:

```json
{
  "duration_ns": 60000000000,
  "fps": 24.0,
  "layers": [
    {
      "index": 0,
      "clips": [
        {
          "id": "clip_0001",
          "name": "interview_01.mxf",
          "start_ns": 0,
          "duration_ns": 15000000000,
          "in_point_ns": 5000000000,
          "effects": ["volume", "color_grade"],
          "type": "video"
        }
      ]
    }
  ]
}
```

When no project is loaded, returns `{"layers": [], "duration_ns": 0, "fps": 24.0}`.

**`GET /api/assets`**

Reads the asset registry JSON file (already exists at `project_dir/assets/registry.json`):

```json
{
  "assets": [
    {
      "id": "asset_001",
      "name": "interview_01.mp4",
      "duration_ns": 120000000000,
      "resolution": "1920x1080",
      "fps": 24.0,
      "thumbnail_url": "/api/assets/asset_001/thumbnail"
    }
  ]
}
```

**`GET /api/assets/{asset_id}/thumbnail`**

Serves the proxy file's first frame as JPEG. Falls back to a 1x1 gray pixel placeholder.

### 5. Timeline Visualization (`timeline.js`)

Pure canvas 2D rendering. No library dependencies.

**Features:**
- Horizontal time axis with adaptive tick marks (frames/seconds/minutes)
- One row per GES layer
- Clips rendered as colored rectangles with truncated name labels
- Effects shown as small colored dots on clip bottoms
- Vertical playhead line synced with preview position
- Horizontal scroll via mouse wheel or drag
- Zoom via Ctrl+scroll
- Click on clip: highlights it, copies clip ID into chat input as context prefix

**Rendering approach:**
- Re-fetch `/api/timeline` on `timeline_updated` WebSocket events
- Debounced: if multiple `timeline_updated` events arrive within 200ms (e.g., batch filler word removal), only one fetch fires
- Only render clips visible in the current viewport (viewport culling)
- For a 30-minute timeline with 100 clips: JSON is ~10KB, canvas rendering is <5ms
- requestAnimationFrame for smooth playhead animation during playback

**Not included (by design):**
- No drag-and-drop
- No clip resizing
- No context menus
- No property panels

These would be agent tasks, not manual tasks.

### 6. Preview Player (`preview.js`)

Refactored from the existing `src/ave/preview/client/index.html`:
- Video element for segment playback
- Canvas overlay for WebSocket frame scrubbing
- Transport controls (play/pause/seek)
- Timecode display
- WebSocket connects to `/preview/ws` (sub-app prefix)
- Synchronization: when playhead moves, timeline.js updates its playhead position

### 7. Asset Browser (`assets.js`)

Simple grid layout:
- Fetches `/api/assets` on load
- Displays thumbnails in a grid with name/duration labels
- Click sends: chat message "Add [asset_name] to the timeline" (user can edit before sending)
- Collapsible panel (hidden by default, toggled by button)

---

## Data Flow

### User sends a chat message

```
User types: "Cut the interview from 0:30 to 2:00 and add a fade"
    → chat.js sends: {"type": "message", "text": "..."}
    → WebSocket /ws/chat
    → ChatSession.handle_message() checks _processing flag
    → _agentic_loop begins:
        → anthropic.AsyncAnthropic().messages.stream(
            model, system_prompt, tools=[3 meta-tools], messages
          )
        → StreamEvent text → ws.send_json({"type": "text_delta"})
        → StreamEvent tool_use start → ws.send_json({"type": "tool_start"})
        → Response complete with stop_reason="tool_use"
        → For each tool call:
            → run_in_executor(orchestrator.handle_tool_call(name, input))
            → ws.send_json({"type": "tool_done"})
            → If timeline-modifying: timeline.reload_from_xges()
            → ws.send_json({"type": "timeline_updated"})
        → Append tool results to messages
        → Loop: stream next response (agent sees tool results)
        → Eventually stop_reason="end_turn"
    → ws.send_json({"type": "done"})

Meanwhile, client:
    → Receives text_delta → appends to chat bubble (streaming)
    → Receives tool_start → shows "[Searching tools...]" indicator
    → Receives tool_done → hides indicator
    → Receives timeline_updated → debounced GET /api/timeline → re-renders canvas
    → Receives done → re-enables input
```

### Preview sync

```
Timeline visualization: user clicks at position X
    → Calculates timestamp_ns from click position
    → Sends to preview.js: preview.seekTo(timestamp_ns)
    → preview.js sends WebSocket frame request to /preview/ws:
        {"type": "frame", "timestamp_ns": ...}
    → PreviewServer extracts frame, returns JPEG
    → preview.js renders on canvas
    → timeline.js updates playhead to match
```

---

## Error Handling

- **Agent errors:** Orchestrator catches exceptions, returns error string. ChatSession forwards as `{"type": "error", "message": "..."}`.
- **WebSocket disconnect:** Server retains ChatSession keyed by session token. Client shows reconnect UI with auto-retry, reconnects with same token.
- **No project loaded:** Timeline API returns empty state. Chat handler creates project on first ingest command.
- **Tool prerequisite failures:** Surfaced as agent text: "I need to load the project first. Let me do that..."
- **Concurrent messages:** Rejected with `{"type": "busy"}` — only one agentic loop runs at a time.
- **Blocking tools:** All tool calls run in `run_in_executor()` to prevent blocking the event loop during FFmpeg/transcription operations.

---

## Testing Strategy

### Unit tests (`tests/test_web/`)

**`test_timeline_model.py`** — pure Python timeline state:
- Empty model → correct JSON
- Add/remove/update clips
- Multi-layer ordering preserved
- Duration calculated from clip positions
- XGES parsing (from XML string, no GES needed)
- Idempotent reload

**`test_chat_protocol.py`** — message parsing, event formatting:
- Parse valid/invalid client messages
- Format all event types (text_delta, tool_start, tool_done, timeline_updated, done, error, busy)
- Handle malformed JSON gracefully

**`test_api_serialization.py`** — REST endpoint logic:
- Timeline model → JSON response
- Asset registry → JSON response with thumbnail URLs
- Missing project returns empty state

### Integration tests (`tests/test_web/`)

**`test_web_integration.py`** — aiohttp test client (no GES, no LLM):
- App factory creates valid application
- Static files served at `/`
- Preview sub-app mounted at `/preview/`
- REST endpoints return correct content types and JSON
- WebSocket chat connects and receives `connected` event
- WebSocket at `/preview/ws` connects

**`test_chat_integration.py`** — chat workflow with mock Anthropic client:
- Mock `anthropic.AsyncAnthropic` to return scripted responses
- Send message → receive text_delta events
- Tool execution → receive tool_start/tool_done events
- Timeline modification → receive timeline_updated event
- Cancel mid-stream → agent stops, receive done
- Busy rejection: send message while processing → receive busy
- Reconnect with same token → session preserved

**`test_chat_e2e.py`** (marked `@pytest.mark.llm`):
- Send real editing command → verify session history contains expected tool calls
- Multi-turn conversation with context preservation
- Agent uses progressive tool discovery (search → schema → call)

### Test markers

- Default tests: no external deps, run everywhere
- `@pytest.mark.llm`: requires ANTHROPIC_API_KEY (real Claude calls)
- `@requires_ffmpeg`: tests involving preview frame extraction

---

## Memory Efficiency

| Component | Memory profile | Mitigation |
|-----------|---------------|------------|
| EditingSession | ~1KB (function refs + history) | History is list of dicts, not media |
| TimelineModel | ~50KB for 100 clips | Pure dataclasses, re-synced from XGES |
| ChatSession messages | ~2KB per turn | Anthropic API manages context window |
| Preview frames | ~50KB per JPEG | Extracted one-at-a-time, not buffered |
| Browser DOM | Minimal (vanilla JS, canvas) | No virtual DOM, no framework overhead |
| Segment cache | ~5MB per 5s segment | Disk-backed, only viewport segments loaded |
| Session store | ~5KB per session | Dict of ChatSession objects, cleaned on app shutdown |

For a 30-minute video project, total server memory overhead from the web layer is <5MB (excluding segment cache which is disk-backed).

---

## Dependencies

```toml
[project.optional-dependencies]
web = ["aiohttp>=3.9", "anthropic>=0.43"]
```

The `anthropic` package provides the Anthropic Python SDK for streaming chat. No `claude-agent-sdk` dependency — we use the lower-level SDK for full control over the tool-use loop.

No frontend build dependencies. The client is static HTML/CSS/JS served directly.

---

## What Phase 8 Does NOT Include

Per the project's incremental design philosophy:
- **Color grading UI (8-4):** Deferred. Complex wheels/curves/scopes require a dedicated design. The agent handles color grading through chat commands today.
- **Drag-and-drop editing:** The agent handles edits. Users describe, agent executes.
- **Multi-user support:** Single user, single session. Multi-user is Phase 7+ concern.
- **Mobile responsive:** Desktop browser only for v1.
- **File upload:** Users volume-mount media directories into Docker. The agent ingests from local paths.

---

## Implementation Order

1. **timeline_model.py** — pure Python timeline state + XGES parser (testable independently)
2. **app.py + api.py** — app factory, REST endpoints, preview sub-app mounting
3. **client/index.html + styles.css** — layout skeleton
4. **timeline.js** — canvas timeline renderer (pure frontend, testable with mock data)
5. **preview.js** — refactored from existing client (connects to /preview/ws)
6. **chat.py** — chat WebSocket handler + Anthropic SDK agentic loop
7. **chat.js** — chat panel client with streaming display
8. **assets.js + asset API** — asset browser
9. **Integration tests** — full workflow verification
