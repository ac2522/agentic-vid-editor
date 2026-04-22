"""AVE Harness — trustworthy end-to-end evaluation of agent behaviour.

Three rungs of increasing cost and coverage:
- Rung A (this module at Phase 2): plan-level tool selection, no execution
- Rung B (Phase 3): real tool execution, state assertions
- Rung C (Phase 4): full render + VLM-judge

Built on Inspect AI (optional dependency — installed via the [harness] extra).
"""

from __future__ import annotations

__version__ = "0.1.0"
