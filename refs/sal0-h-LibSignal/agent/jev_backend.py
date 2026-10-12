"""Small HTTP client for OpenRouter's native Decisions API (not chat completions).

Contract: https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-questions-and-answers-request
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from agent.jev_prompt import QUESTION_ID
from agent.llm_tsc_prompt import SIGNAL_ORDER


ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL_ID = "typesafe/jev-1.13"
ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504, 524, 529}


class JEVError(RuntimeError):
    """An API failure or an invalid decision; never substitute another controller."""


@dataclass(frozen=True)
class JEVDecision:
    signal: str
    confidence: float | None
    probabilities: dict[str, float] | None
    model: str
    request_id: str | None
    usage: dict[str, int | float]
    elapsed_seconds: float
    attempts: int


def _probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JEVError(f"JEV {field} must be a number between 0 and 1")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise JEVError(f"JEV {field} must be a number between 0 and 1")
    return float(value)


def parse_decision(data: Any, expected_model: str, elapsed: float, attempts: int) -> JEVDecision:
    """Validate a typed answer before it can become a SUMO action.

    OpenRouter makes confidence and probabilities optional in its schema. Preserve
    absent values as None; absence must not be reported as zero confidence.
    """
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        raise JEVError("JEV response is missing its answers object")
    model = data.get("model")
    if not isinstance(model, str) or not (
        model == expected_model or model.startswith(expected_model + "-")
    ):
        raise JEVError("OpenRouter returned a different model from the requested JEV model")
    answer = data["answers"].get(QUESTION_ID)
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise JEVError("JEV response is missing the next_signal Choice answer")
    signal = answer.get("choice")
    if not isinstance(signal, str) or signal not in SIGNAL_ORDER:
        raise JEVError("JEV returned a signal outside the four allowed choices")

    confidence = answer.get("confidence")
    if confidence is not None:
        confidence = _probability(confidence, "confidence")
    probabilities = answer.get("probabilities")
    if probabilities is not None:
        if not isinstance(probabilities, dict) or set(probabilities) != set(SIGNAL_ORDER):
            raise JEVError("JEV probabilities must cover exactly the four allowed signals")
        probabilities = {
            name: _probability(value, "probability") for name, value in probabilities.items()
        }
        # Live OpenRouter responses round probabilities to two decimal places.
        # Keep those values intact, allowing half a rounding unit per option.
        rounding_tolerance = 0.005 * len(probabilities) + 1e-9
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=rounding_tolerance):
            raise JEVError("JEV probabilities do not sum to one")

    raw_usage = data.get("usage")
    if not isinstance(raw_usage, dict):
        raise JEVError("JEV response is missing usage")
    usage = {}
    for name in ("input_tokens", "output_tokens", "cost"):
        value = raw_usage.get(name)
        if value is None and name == "cost":
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise JEVError(f"JEV returned invalid usage.{name}")
        usage[name] = value
    request_id = data.get("id")
    if request_id is not None and not isinstance(request_id, str):
        raise JEVError("JEV returned an invalid request id")
    return JEVDecision(
        signal, confidence, probabilities, model, request_id, usage, elapsed, attempts
    )


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    """Honor Retry-After seconds or dates, but do not stall a run indefinitely."""
    delay = float(2 ** attempt)
    if retry_after:
        try:
            delay = float(retry_after)
        except ValueError:
            try:
                delay = parsedate_to_datetime(retry_after).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                pass
    if not math.isfinite(delay):
        delay = float(2 ** attempt)
    if delay > 30:
        raise JEVError("OpenRouter requested a retry delay over 30 seconds; try the run later")
    return max(0.0, delay)


class JEVClient:
    """One connection pool per intersection, reused across decision steps."""

    def __init__(self, param: dict[str, Any]):
        # Optional dependencies are loaded only when an active JEV junction calls us.
        import requests

        self.model = str(param.get("model_id", MODEL_ID)).strip()
        self.timeout = float(param.get("timeout", 30))
        self.request_attempts = int(param.get("request_attempts", 3))
        if not self.model or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("JEV requires a model_id and a positive finite timeout")
        if not 1 <= self.request_attempts <= 5:
            raise ValueError("JEV request_attempts must be between 1 and 5")

        self._api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not self._api_key:
            from dotenv import dotenv_values

            values = dotenv_values(ENV_FILE, interpolate=False)
            self._api_key = (values.get("OPENROUTER_API_KEY") or "").strip()
        if not self._api_key:
            raise ValueError("Set OPENROUTER_API_KEY in the environment or the repository .env")
        self._session = requests.Session()

    def _http_error(self, response) -> str:
        # Never log headers, credentials, request objects, or a raw response body.
        message = ""
        try:
            error = response.json().get("error", {})
            if isinstance(error, dict):
                message = str(error.get("message", ""))
        except (ValueError, AttributeError):
            pass
        message = message.replace(self._api_key, "[redacted]")[:400]
        return f"OpenRouter JEV HTTP {response.status_code}: {message}"

    def decide(self, request: dict[str, Any]) -> JEVDecision:
        """Retry transport failures only; typed answers need no text-repair loop."""
        import requests

        started = time.monotonic()
        payload = {
            "model": self.model,
            "state": request["state"],
            "questions": request["questions"],
        }
        for attempt in range(self.request_attempts):
            retry_after = None
            try:
                response = self._session.post(
                    ENDPOINT,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=self.timeout,
                    allow_redirects=False,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                error = f"OpenRouter JEV transport failure ({type(exc).__name__})"
            except requests.RequestException as exc:
                raise JEVError(f"OpenRouter JEV request failed ({type(exc).__name__})") from None
            else:
                if response.status_code == 200:
                    try:
                        data = response.json()
                    except ValueError:
                        raise JEVError("OpenRouter JEV returned invalid JSON") from None
                    return parse_decision(data, self.model, time.monotonic() - started, attempt + 1)
                error = self._http_error(response)
                if response.status_code not in RETRYABLE_STATUSES:
                    raise JEVError(error)
                retry_after = response.headers.get("Retry-After")

            if attempt + 1 == self.request_attempts:
                raise JEVError(f"{error}; failed after {self.request_attempts} attempts") from None
            time.sleep(_retry_delay(retry_after, attempt))
        raise AssertionError("Unreachable: request_attempts is positive")
