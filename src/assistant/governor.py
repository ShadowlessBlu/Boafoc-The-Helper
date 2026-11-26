"""The Governor: the only path from an LLM proposal to a real action.

The LLM proposes {"name": ..., "args": {...}}. The Governor decides:
  ALLOW   - run it
  CONFIRM - ask the user first (sensitive actions)
  DENY    - never run it (unknown tool, bad arguments, forbidden, rate-limited)
Every decision goes to the audit callback.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class Risk(str, Enum):
    READ = "read"            # no side effects: auto-allow
    LOW = "low"              # harmless side effects: auto-allow, logged
    SENSITIVE = "sensitive"  # physical / irreversible / personal: needs confirmation
    FORBIDDEN = "forbidden"  # never, even if the user asks through the LLM


class Verdict(str, Enum):
    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool}


@dataclass
class Param:
    type: str = "string"
    required: bool = True
    enum: tuple | None = None
    max_len: int = 200


@dataclass
class ToolSpec:
    name: str
    description: str
    risk: Risk
    params: dict[str, Param] = field(default_factory=dict)
    rate_per_min: int | None = None
    func: Callable[..., Any] | None = None
    speak: Callable[[Any], str] | None = None  # result -> spoken sentence, skips a second LLM call
    # args -> the question asked before a SENSITIVE tool runs; None = nothing to confirm (run it,
    # it will say so). Without it the question is "Should I run <tool> with <args>?"
    confirm_text: Callable[[dict], str | None] | None = None


@dataclass
class Decision:
    verdict: Verdict
    reason: str
    tool: str
    args: dict


class Governor:
    def __init__(self, tools: dict[str, ToolSpec], default_rate_per_min: int = 10,
                 audit: Callable[[Decision], None] | None = None, clock: Callable[[], float] = time.monotonic):
        self.tools = tools
        self.default_rate = default_rate_per_min
        self.audit = audit or (lambda d: None)
        self.clock = clock
        self._calls: dict[str, deque] = defaultdict(deque)

    def _decide(self, proposal: Any) -> Decision:
        if not isinstance(proposal, dict):
            return Decision(Verdict.DENY, "proposal is not an object", "?", {})
        name = proposal.get("name")
        args = proposal.get("args") or {}
        if not isinstance(name, str) or name not in self.tools:
            return Decision(Verdict.DENY, f"unknown tool '{name}'", str(name), {})
        if not isinstance(args, dict):
            return Decision(Verdict.DENY, "args must be an object", name, {})
        spec = self.tools[name]
        if spec.risk is Risk.FORBIDDEN:
            return Decision(Verdict.DENY, "tool is forbidden by policy", name, args)

        # argument validation
        extra = set(args) - set(spec.params)
        note = ""
        if extra and spec.risk is Risk.READ:
            # Small models invent arguments (qwen3:1.7b called get_time with {"timezone": "local"}).
            # A read-only tool can't do harm, so drop them instead of refusing; anything with side
            # effects still gets denied, because there an unknown argument may change what happens.
            args = {k: v for k, v in args.items() if k in spec.params}
            note = f" (ignored unexpected arguments: {sorted(extra)})"
        elif extra:
            return Decision(Verdict.DENY, f"unexpected arguments: {sorted(extra)}", name, args)
        for pname, p in spec.params.items():
            if pname not in args:
                if p.required:
                    return Decision(Verdict.DENY, f"missing argument '{pname}'", name, args)
                continue
            v = args[pname]
            if isinstance(v, bool) and p.type != "boolean":
                return Decision(Verdict.DENY, f"'{pname}' must be {p.type}", name, args)
            if not isinstance(v, _TYPES[p.type]):
                return Decision(Verdict.DENY, f"'{pname}' must be {p.type}", name, args)
            if isinstance(v, str) and len(v) > p.max_len:
                return Decision(Verdict.DENY, f"'{pname}' is too long", name, args)
            if p.enum and v not in p.enum:
                return Decision(Verdict.DENY, f"'{pname}' must be one of {list(p.enum)}", name, args)

        # rate limit (sliding 60 s window)
        limit = spec.rate_per_min or self.default_rate
        now = self.clock()
        q = self._calls[name]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= limit:
            return Decision(Verdict.DENY, f"rate limit {limit}/min reached", name, args)
        q.append(now)

        if spec.risk is Risk.SENSITIVE:
            return Decision(Verdict.CONFIRM, "sensitive action needs confirmation", name, args)
        return Decision(Verdict.ALLOW, f"{spec.risk.value} action{note}", name, args)

    def check(self, proposal: Any) -> Decision:
        d = self._decide(proposal)
        self.audit(d)
        return d

    def execute(self, decision: Decision, confirmed: bool = False) -> Any:
        """Run an approved tool. CONFIRM decisions need confirmed=True."""
        if decision.verdict is Verdict.DENY:
            raise PermissionError(decision.reason)
        if decision.verdict is Verdict.CONFIRM and not confirmed:
            raise PermissionError("not confirmed by user")
        spec = self.tools[decision.tool]
        if spec.func is None:
            raise RuntimeError(f"tool '{spec.name}' has no implementation")
        return spec.func(**decision.args)
