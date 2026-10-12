"""Translate Traffic-R1's observation into one native Jev Choice question."""

from __future__ import annotations

from typing import Any

from agent.llm_tsc_prompt import SIGNAL_ALLOWED, SIGNAL_ORDER


QUESTION_ID = "next_signal"


def build_jev_request(signal_stats: dict, n_segments: int = 3) -> dict[str, Any]:
    """Keep the paper's counts and signal vocabulary; compute all sums in Python.

    Jev receives structured state and a closed set of choices. It does not receive
    chat messages, reasoning instructions, or the Traffic-R1 output-format prompt.
    """
    if n_segments < 1:
        raise ValueError("JEV n_segments must be positive")

    signals = {}
    for signal in SIGNAL_ORDER:
        approaches = {
            side: {
                "early_queued": int(counts["early_queued"]),
                "segments": [int(value) for value in counts["segments"]],
            }
            for side, counts in signal_stats[signal].items()
        }
        if any(len(counts["segments"]) != n_segments for counts in approaches.values()):
            raise ValueError("JEV segment counts must match n_segments")
        signals[signal] = {
            "approaches": approaches,
            "totals": {
                "early_queued": sum(counts["early_queued"] for counts in approaches.values()),
                "segments": [
                    sum(counts["segments"][index] for counts in approaches.values())
                    for index in range(n_segments)
                ],
            },
        }

    return {
        "state": {
            "intersection": (
                "A four-way intersection of north-south and east-west roads. "
                "The controlled movements are through traffic and left turns "
                "from each approach. Right turns are outside this signal choice."
            ),
            "observation_fields": {
                "early_queued": "Stopped vehicles waiting for passage permission.",
                "segments": (
                    f"Approaching vehicle counts in {n_segments} equal lane-length "
                    "segments, ordered from nearest to farthest from the intersection. "
                    "These counts exclude early queued vehicles."
                ),
                "totals": "Counts summed across the two approaches served by a signal.",
            },
            "signals": signals,
        },
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": (
                    "Which signal should be green next to reduce vehicle waiting "
                    "at this intersection, given the queued and approaching vehicles "
                    "in `signals`? Select one signal even when all counts are zero."
                ),
                "criteria": dict(SIGNAL_ALLOWED),
            }
        },
    }
