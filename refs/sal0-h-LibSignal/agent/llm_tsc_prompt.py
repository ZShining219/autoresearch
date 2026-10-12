"""Prompt construction and response parsing for the LLM TSC controller.

The observation and four-signal vocabulary follow Traffic-R1 Appendix A.1.
The controller can use the same decision interface with a local Traffic-R1
checkpoint or a separate OpenAI-compatible model.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

# Paper action vocabulary (exactly four phases).
SIGNAL_ORDER = ("ETWT", "ELWL", "NTST", "NLSL")

SIGNAL_ALLOWED = {
    "ETWT": "Eastern and western through lanes",
    "ELWL": "Eastern and western left lanes",
    "NTST": "North and south through lanes",
    "NLSL": "North and south left lanes",
}

# (approach_from, movement) pairs served by each signal. Approach is where
# traffic comes from (N/E/S/W). Movement is through (T) or left (L).
SIGNAL_LANE_PAIRS = {
    "ETWT": (("E", "T"), ("W", "T")),
    "ELWL": (("E", "L"), ("W", "L")),
    "NTST": (("N", "T"), ("S", "T")),
    "NLSL": (("N", "L"), ("S", "L")),
}

APPROACH_WORD = {"N": "North", "S": "South", "E": "East", "W": "West"}

TASK_DESCRIPTION = (
    "The crossroad connects two roads: north-south and east-west, with the traffic "
    "light at their intersection. Each road is divided into two sections (e.g., north "
    "and south for the north-south road) and each section has two lanes: a through "
    "lane and a left-turn lane. Right turns are always permitted. Each lane is further "
    "divided into three segments. Segment 1 is the closest to the intersection. "
    "Segment 2 is in the middle. Segment 3 is the farthest. In a lane, there may be "
    "early queued vehicles and approaching vehicles traveling in different segments. "
    "Early queued vehicles have arrived at the intersection and await passage "
    "permission. Approaching vehicles will arrive at the intersection in the future."
)

FORMAT_INSTRUCTION = (
    "You can only choose one of the signals listed above. You FIRST think about the "
    "reasoning process for your choice as an internal monologue and then provide the "
    "final answer. Your think process MUST BE put in <think>...</think> tags. "
    "The final choice MUST BE put in \\boxed{}."
)

# Generic reasoning APIs need stronger output compliance than the fine-tuned
# Traffic-R1 checkpoint. The API backend applies this automatically.
DEFAULT_API_FORMAT_SUFFIX = (
    " Keep the think block to one or two short sentences (under 50 words). "
    "Do not repeat the task or discuss these instructions. "
    "Even if all queues are zero, you MUST still choose exactly one of "
    "ETWT, ELWL, NTST, NLSL. End immediately with exactly one literal final form: "
    "\\boxed{ETWT}, \\boxed{ELWL}, \\boxed{NTST}, or \\boxed{NLSL}. "
    "Use the chosen signal inside the box; never write the placeholder SIGNAL. "
    "Do not ask questions or continue thinking after the final box."
)

SYSTEM_PROMPT = "You are a helpful traffic control agent."

_BOXED_RE = re.compile(r"\\boxed\{\s*([A-Za-z]+)\s*\}")
_SIGNAL_TOKEN_RE = re.compile(r"\b(ETWT|ELWL|NTST|NLSL)\b", re.IGNORECASE)


class LLMParseError(ValueError):
    """Raised when a model response cannot be parsed into a valid signal."""


def build_observation_text(
    signal_stats: Dict[str, Dict[str, Dict[str, int]]],
    n_segments: int = 3,
) -> str:
    """Build the structured traffic observation block."""
    lines: List[str] = ["Structured Traffic Observation:"]
    for signal in SIGNAL_ORDER:
        pair = SIGNAL_LANE_PAIRS[signal]
        side_a = APPROACH_WORD[pair[0][0]]
        side_b = APPROACH_WORD[pair[1][0]]
        stats = signal_stats[signal]
        a = stats[side_a]
        b = stats[side_b]
        q_a, q_b = int(a["early_queued"]), int(b["early_queued"])
        lines.append(f"Signal: {signal}")
        lines.append(f"Allowed lanes: {SIGNAL_ALLOWED[signal]}")
        lines.append(
            f"- Early queued: {q_a} ({side_a}), {q_b} ({side_b}), "
            f"{q_a + q_b} (Total)"
        )
        segs_a = list(a["segments"]) + [0] * n_segments
        segs_b = list(b["segments"]) + [0] * n_segments
        for s in range(n_segments):
            ca, cb = int(segs_a[s]), int(segs_b[s])
            lines.append(
                f"- Segment {s + 1}: {ca} ({side_a}), {cb} ({side_b}), "
                f"{ca + cb} (Total)"
            )
    return "\n".join(lines)


def build_user_prompt(
    signal_stats: Dict[str, Dict[str, Dict[str, int]]],
    n_segments: int = 3,
) -> str:
    """Build the paper-derived user prompt without backend-specific text."""
    parts = [
        f"Task Description: {TASK_DESCRIPTION}",
        build_observation_text(signal_stats, n_segments=n_segments),
    ]
    parts.append(f"Format Instruction: {FORMAT_INSTRUCTION}")
    return "\n\n".join(parts)


def build_messages(
    signal_stats: Dict[str, Dict[str, Dict[str, int]]],
    n_segments: int = 3,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                signal_stats,
                n_segments=n_segments,
            ),
        },
    ]


def parse_signal(response: str, allowed: Sequence[str] = SIGNAL_ORDER) -> str:
    """Extract a valid signal from the final answer portion of a response."""
    if response is None:
        raise LLMParseError("Empty model response")
    text = str(response).strip()
    if not text:
        raise LLMParseError("Empty model response")

    allowed_set = {a.upper() for a in allowed}
    close = text.rfind("</think>")
    final_text = text[close + len("</think>") :] if close != -1 else text

    # Ignore boxed choices that occur only in the reasoning block. This keeps
    # a truncated chain-of-thought from becoming an accepted control action.
    boxed = _BOXED_RE.findall(final_text)
    if boxed:
        choice = boxed[-1].upper()
        if choice not in allowed_set:
            raise LLMParseError(
                f"Boxed signal {choice!r} not in allowed set {sorted(allowed_set)}"
            )
        return choice

    tokens = [t.upper() for t in _SIGNAL_TOKEN_RE.findall(final_text)]
    tokens = [t for t in tokens if t in allowed_set]
    if len(tokens) == 1:
        return tokens[0]
    if len(tokens) > 1:
        return tokens[-1]

    raise LLMParseError(
        "Could not parse a valid signal from model response "
        f"(expected \\boxed{{ETWT|ELWL|NTST|NLSL}}). "
        f"Response tail: {text[-400:]!r}"
    )


def empty_signal_stats(n_segments: int = 3) -> Dict[str, Dict[str, Dict[str, int]]]:
    stats: Dict[str, Dict[str, Dict[str, int]]] = {}
    for signal, pair in SIGNAL_LANE_PAIRS.items():
        stats[signal] = {}
        for approach, _movement in pair:
            side = APPROACH_WORD[approach]
            stats[signal][side] = {
                "early_queued": 0,
                "segments": [0] * n_segments,
            }
    return stats
