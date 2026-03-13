# E2E Orchestrator Workflow Tests — Design Spec

## Purpose

Test that natural language video editing requests — phrased by amateurs and professionals — produce correct tool call sequences when routed through the Orchestrator with a real Claude API call.

## Approach

Each test scenario:
1. Sets up an `EditingSession` with a loaded project and appropriate state
2. Feeds a natural language prompt + system prompt + meta-tools to Claude via the Anthropic API
3. Runs a tool-use loop (max 20 turns) routing tool calls through `Orchestrator.handle_tool_call()`
4. Asserts on **outcomes**: which tools were called, what state was accumulated, and that results are valid

## Constraints

- One API call per scenario (single agentic loop, not separate calls)
- Skip if `ANTHROPIC_API_KEY` not set
- Use `claude-haiku-4-5-20251001` to minimize cost
- Mark all tests `@pytest.mark.slow` and `@pytest.mark.llm`
- Max 20 tool-use turns per scenario to bound cost
- Assert on outcomes (tools called, state), not exact sequences

## Scenarios

### 1. Instagram Reel Export
**Amateur:** "I have a 4K video that's about 2 minutes long. I want to take just the last 12 seconds and turn it into a vertical video for Instagram. Can you also add a background song?"
**Professional:** "Extract the tail 12s segment, reframe to 1080x1920 portrait, render with instagram_reel preset."

**Expected tools:** trim, render_with_preset (or equivalent). Amateur may also search for download/audio tools.
**Expected state:** `clip_trimmed`, possibly `preset_rendered`

### 2. Color Correction
**Amateur:** "The person's face is way too dark and the background is really bright. I just want to be able to see their face better."
**Professional:** "Lift shadows RGB to 0.15, reduce gain to 0.85 across channels, bump gamma green down to 0.95, saturation 1.1."

**Expected tools:** `color_grade`
**Expected state:** `color_graded`

### 3. Transcript Cleanup
**Amateur:** "There's a lot of ums and uhs in this video, can you clean those up so it sounds more professional?"
**Professional:** "Find all filler words in the transcript, compute text cuts for each filler region, preserve inter-segment continuity."

**Expected tools:** `find_fillers`, `text_cut`
**Expected state:** `fillers_found`, `text_cut_computed`

### 4. Highlight Reel / Rough Cut
**Amateur:** "This is a 10-minute video of a birthday party. Can you pick out the best moments and make a short highlight clip?"
**Professional:** "Detect scenes at content threshold 27, select scene indices 0, 2, 4, create rough cut in chronological order with 500ms inter-scene gaps."

**Expected tools:** `detect_scenes`, `create_rough_cut`
**Expected state:** `scenes_detected`, `rough_cut_created`

### 5. Audio Mixing
**Amateur:** "The background music is way too loud, I can barely hear the person talking. Fix the audio so the voice is clear."
**Professional:** "Normalize dialogue peak to -14dB target, set background music volume to -18dB, apply 2-second fade-in on music track."

**Expected tools:** `normalize` and/or `volume`, possibly `fade`
**Expected state:** `audio_normalized` or `volume_set`

## Test Architecture

```
tests/test_e2e/test_orchestrator_workflows.py
├── requires_anthropic marker (skip if no API key)
├── _run_agent_loop() helper — drives Claude tool-use loop
├── TestInstagramExport
│   ├── test_amateur_instagram_reel
│   └── test_professional_instagram_reel
├── TestColorCorrection
│   ├── test_amateur_color_fix
│   └── test_professional_color_grade
├── TestTranscriptCleanup
│   ├── test_amateur_remove_ums
│   └── test_professional_filler_cut
├── TestHighlightReel
│   ├── test_amateur_best_moments
│   └── test_professional_rough_cut
└── TestAudioMixing
    ├── test_amateur_fix_loud_music
    └── test_professional_audio_normalize
```

## Agent Loop Implementation

```python
def _run_agent_loop(orchestrator, user_prompt, max_turns=20):
    """Drive a Claude tool-use loop through the orchestrator."""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_prompt}]
    tools = [meta_tool_to_api_format(mt) for mt in orchestrator.get_meta_tools()]

    for _ in range(max_turns):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=orchestrator.get_system_prompt(),
            tools=tools,
            messages=messages,
        )
        if response.stop_reason == "end_turn":
            break
        # Process tool uses
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = orchestrator.handle_tool_call(block.name, block.input)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
        messages.append({"role": "user", "content": tool_results})

    return orchestrator.session.history, orchestrator.session.state
```
