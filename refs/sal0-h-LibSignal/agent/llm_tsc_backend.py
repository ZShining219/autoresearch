"""Backends for the generic LLM traffic-signal controller."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

from agent.llm_tsc_prompt import DEFAULT_API_FORMAT_SUFFIX, SIGNAL_ORDER


_TRAFFIC_R1_STOP_STRINGS = tuple(
    f"\\boxed{{{signal}}}" for signal in SIGNAL_ORDER
)

_API_RETRY_INSTRUCTION = (
    "Your previous response did not provide a usable final action. "
    "Do not repeat the reasoning. Reply now with only one literal final form: "
    "\\boxed{ETWT}, \\boxed{ELWL}, \\boxed{NTST}, or \\boxed{NLSL}."
)


class LLMBackend:
    """Common interface: a message list produces normalized completion text."""

    def complete(self, messages: List[Dict[str, str]]) -> str:
        """Convenience wrapper around the canonical batch interface."""
        return self.complete_many([messages])[0]

    def complete_many(
        self,
        messages_list: Sequence[List[Dict[str, str]]],
    ) -> List[str]:
        """Complete requests in order; subclasses may provide concurrency."""
        return [self._complete_one(messages) for messages in messages_list]

    def _complete_one(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError

    def parse_attempts(self, configured_attempts: int) -> int:
        """Return meaningful parse attempts for this backend."""
        return max(1, int(configured_attempts))

    def retry_messages(
        self,
        messages: List[Dict[str, str]],
        response: str,
    ) -> List[Dict[str, str]]:
        """Build the next parse-retry request."""
        del response
        return messages


def create_backend(param: Dict[str, Any]) -> LLMBackend:
    backend = str(param.get("backend", "api")).lower()
    if backend == "local":
        return LocalHFBackend(param)
    if backend == "vllm":
        return VLLMBackend(param)
    if backend == "api":
        return OpenAICompatibleBackend(param)
    raise ValueError(
        f"Unknown LLM backend {backend!r}; expected 'local', 'vllm', or 'api'"
    )


def _resolve_hf_weights_dir(model_path: str, subfolder: Any) -> str:
    """Return a local directory vLLM can load (HF repo id or existing path)."""
    folder = None if subfolder is None or str(subfolder).strip() == "" else str(subfolder)
    if os.path.isdir(model_path):
        resolved = os.path.join(model_path, folder) if folder else model_path
    else:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download(model_path)
        resolved = os.path.join(snapshot, folder) if folder else snapshot
    if not os.path.isdir(resolved):
        raise FileNotFoundError(
            f"vLLM weights directory does not exist: {resolved!r} "
            f"(model_path={model_path!r}, subfolder={folder!r})"
        )
    return resolved


def _chat_tokenizer(tokenizer: Any) -> Any:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer
    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None and hasattr(inner, "apply_chat_template"):
        return inner
    raise TypeError("vLLM tokenizer does not expose apply_chat_template")


class OpenAICompatibleBackend(LLMBackend):
    """Chat Completions against an OpenAI-compatible HTTP endpoint."""

    def __init__(self, param: Dict[str, Any]):
        self.base_url = str(param.get("base_url", "")).rstrip("/")
        if not self.base_url:
            raise ValueError(
                "model.base_url must be set in the agent YAML for backend=api"
            )
        self.model = str(param.get("model_id", "")).strip()
        if not self.model:
            raise ValueError(
                "model.model_id must be set in the agent YAML for backend=api"
            )

        key_env = str(param.get("api_key_env", "OPENAI_API_KEY"))
        self.api_key_env = key_env
        self.api_key = os.environ.get(key_env, "")
        if not self.api_key:
            raise ValueError(
                f"API key env var {key_env!r} is unset. Set it before running "
                "the API-backed LLM controller."
            )

        self.temperature = float(param.get("temperature", 0.0))
        self.max_tokens = int(param.get("max_new_tokens", 512))
        self.timeout = float(param.get("timeout", 120.0))
        self.extra_body = param.get("extra_body") or {}
        self.parallelism = max(1, int(param.get("parallelism", 16)))
        self._client = None
        self._client_lock = threading.Lock()
        self._generation_calls = 0

    def _get_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    try:
                        from openai import OpenAI
                    except ImportError as exc:
                        raise ImportError(
                            "The 'openai' package is required for backend=api. "
                            "Install it with: pip install openai"
                        ) from exc
                    self._client = OpenAI(
                        base_url=self.base_url,
                        api_key=self.api_key,
                        timeout=self.timeout,
                    )
        return self._client

    @staticmethod
    def _prepare_messages(
        messages: List[Dict[str, str]],
        suffix: str = DEFAULT_API_FORMAT_SUFFIX,
    ) -> List[Dict[str, str]]:
        """Copy messages and apply the API-only output-compliance suffix."""
        prepared = [dict(message) for message in messages]
        for message in reversed(prepared):
            if message.get("role") == "user":
                message["content"] = (
                    str(message.get("content", "")) + suffix
                )
                break
        return prepared

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", None)
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(getattr(item, "text", item)))
            return "".join(parts)
        return str(content)

    def _extract_text(self, response: Any) -> str:
        if response is None:
            raise RuntimeError("LLM API returned an empty response")
        error = getattr(response, "error", None)
        if error:
            raise RuntimeError(f"LLM API error: {error}")
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("LLM API returned no choices")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise RuntimeError("LLM API returned a choice without a message")
        dump = message.model_dump() if hasattr(message, "model_dump") else {}
        content = self._message_text(message)
        reasoning = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or dump.get("reasoning_content")
            or dump.get("reasoning")
        )
        if reasoning and "<think>" not in content:
            content = f"<think>\n{reasoning}\n</think>\n{content}"
        return content

    def _complete_one(self, messages: List[Dict[str, str]]) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": self._prepare_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        response = self._get_client().chat.completions.create(**kwargs)
        return self._extract_text(response)

    def complete_many(
        self,
        messages_list: Sequence[List[Dict[str, str]]],
    ) -> List[str]:
        started = time.monotonic()
        if self.parallelism <= 1 or len(messages_list) <= 1:
            responses = [self._complete_one(messages) for messages in messages_list]
        else:
            with ThreadPoolExecutor(
                max_workers=min(self.parallelism, len(messages_list))
            ) as pool:
                futures = [
                    pool.submit(self._complete_one, messages)
                    for messages in messages_list
                ]
                responses = [future.result() for future in futures]

        self._generation_calls += 1
        print(
            f"[LLM API] generate call={self._generation_calls} "
            f"batch={len(messages_list)} elapsed={time.monotonic() - started:.2f}s",
            flush=True,
        )
        return responses

    def retry_messages(
        self,
        messages: List[Dict[str, str]],
        response: str,
    ) -> List[Dict[str, str]]:
        # A deterministic retry of the original prompt repeats the same
        # truncated reasoning. Preserve that response as context and ask the
        # API model only for the missing final action.
        return [
            *[dict(message) for message in messages],
            {"role": "assistant", "content": str(response)},
            {"role": "user", "content": _API_RETRY_INSTRUCTION},
        ]


class LocalHFBackend(LLMBackend):
    """Local transformers generation for the Traffic-R1 checkpoint."""

    def __init__(self, param: Dict[str, Any]):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.generation.stopping_criteria import (
                StopStringCriteria,
                StoppingCriteriaList,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for backend=local. "
                "Install with: pip install transformers accelerate"
            ) from exc

        self.torch = torch
        model_path = str(param.get("model_path", "Season998/Traffic-R1"))
        subfolder = param.get("model_subfolder", "huggingface")
        if subfolder is not None and str(subfolder).strip() == "":
            subfolder = None

        self.temperature = float(param.get("temperature", 0.0))
        self.max_new_tokens = int(param.get("max_new_tokens", 512))
        self.do_sample = bool(param.get("do_sample", False))
        self.top_p = param.get("top_p")
        if self.top_p is not None:
            self.top_p = float(self.top_p)
        self.top_k = param.get("top_k")
        if self.top_k is not None:
            self.top_k = int(self.top_k)
        self.chat_template_kwargs = dict(param.get("chat_template_kwargs") or {})

        load_kwargs: Dict[str, Any] = {}
        if subfolder:
            load_kwargs["subfolder"] = str(subfolder)

        device_map = param.get("device_map", "auto")
        torch_dtype = param.get("torch_dtype", "bfloat16")
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        elif torch_dtype == "float16":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            dtype = torch.float32

        print(
            f"[LLM] Loading local model from {model_path!r} "
            f"(subfolder={subfolder!r}, device_map={device_map!r}, dtype={torch_dtype!r}) ...",
            flush=True,
        )
        print("[LLM] Loading tokenizer ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, **load_kwargs
        )
        # Decoder-only batch generate needs a pad id and left padding.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # Traffic-R1 Appendix A.1 defines the final action as a boxed signal.
        # Build this once: StopStringCriteria preprocessing is too expensive to
        # repeat for every one of the 540 decision batches in a full run.
        self._answer_stopping_criteria = StoppingCriteriaList(
            [StopStringCriteria(self.tokenizer, _TRAFFIC_R1_STOP_STRINGS)]
        )
        self._generation_calls = 0
        print("[LLM] Tokenizer ready; loading weights ...", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map if torch.cuda.is_available() else None,
            **load_kwargs,
        )
        print("[LLM] from_pretrained returned; eval() ...", flush=True)
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")
        self.model.eval()
        print("[LLM] Local model ready.", flush=True)

    def parse_attempts(self, configured_attempts: int) -> int:
        # Greedy generation is deterministic, but retries are still useful when
        # the malformed response is included in a corrective continuation.
        return super().parse_attempts(configured_attempts)

    def retry_messages(
        self,
        messages: List[Dict[str, str]],
        response: str,
    ) -> List[Dict[str, str]]:
        # Do not resend an unchanged greedy prompt: it would reproduce the
        # same invalid action. Ask the model to emit only a valid signal box.
        return [
            *[dict(message) for message in messages],
            {"role": "assistant", "content": str(response)},
            {"role": "user", "content": _API_RETRY_INSTRUCTION},
        ]

    def _gen_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample and self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "stopping_criteria": self._answer_stopping_criteria,
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if self.top_k is not None:
                kwargs["top_k"] = self.top_k
        return kwargs

    def _complete_one(self, messages: List[Dict[str, str]]) -> str:
        return self.complete_many([messages])[0]

    def complete_many(
        self,
        messages_list: Sequence[List[Dict[str, str]]],
    ) -> List[str]:
        """Batched greedy/sampling generate; one forward stack for all prompts."""
        if not messages_list:
            return []

        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
            for messages in messages_list
        ]
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **self._gen_kwargs())
        elapsed = time.perf_counter() - started
        generated = output[:, prompt_len:]
        responses = [
            self.tokenizer.decode(
                generated[index], skip_special_tokens=True
            )
            for index in range(len(prompts))
        ]
        token_counts = [
            int(row.ne(self.tokenizer.pad_token_id).sum().item())
            for row in generated
        ]
        self._generation_calls += 1
        completed = sum(
            any(stop in response for stop in _TRAFFIC_R1_STOP_STRINGS)
            for response in responses
        )
        print(
            f"[LLM] generate call={self._generation_calls} batch={len(prompts)} "
            f"tokens={min(token_counts)}/{sum(token_counts) / len(token_counts):.1f}/"
            f"{max(token_counts)} (min/mean/max) boxed={completed}/{len(prompts)} "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )
        return responses


class VLLMBackend(LLMBackend):
    """In-process vLLM engine: greedy Traffic-R1 decode with per-sequence stop.

    HuggingFace generate() keeps the whole padded batch alive until the slowest
    sequence boxes (or hits max_new_tokens). vLLM finishes each of the 16
    intersections as soon as it emits a boxed signal, and prefix-caches the
    shared task text across steps.

    ``distributed_executor_backend=uni`` keeps the engine in this process so it
    can share the CUDA context LibSignal already created. Do not load
    backend=local in the same run: two copies of the weights will OOM.
    """

    def __init__(self, param: Dict[str, Any]):
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "The 'vllm' package is required for backend=vllm. Install it in "
                "the same env as torch/CUDA (H200 job), e.g. `pip install vllm`. "
                "backend=local still uses transformers.generate."
            ) from exc

        model_path = str(param.get("model_path", "Season998/Traffic-R1"))
        subfolder = param.get("model_subfolder", "huggingface")
        model_dir = _resolve_hf_weights_dir(model_path, subfolder)

        self.temperature = float(param.get("temperature", 0.0))
        self.max_new_tokens = int(param.get("max_new_tokens", 512))
        self.do_sample = bool(param.get("do_sample", False))
        self.top_p = param.get("top_p")
        if self.top_p is not None:
            self.top_p = float(self.top_p)
        self.top_k = param.get("top_k")
        if self.top_k is not None:
            self.top_k = int(self.top_k)
        self.chat_template_kwargs = dict(param.get("chat_template_kwargs") or {})

        torch_dtype = str(param.get("torch_dtype", "bfloat16"))
        if torch_dtype in ("bfloat16", "bf16"):
            dtype = "bfloat16"
        elif torch_dtype in ("float16", "fp16"):
            dtype = "float16"
        else:
            dtype = "auto"

        engine_kwargs: Dict[str, Any] = {
            "model": model_dir,
            "tokenizer": model_dir,
            "trust_remote_code": True,
            "dtype": dtype,
            "max_model_len": int(param.get("max_model_len", 4096)),
            "gpu_memory_utilization": float(param.get("gpu_memory_utilization", 0.85)),
            "max_num_seqs": int(param.get("max_num_seqs", 32)),
            "disable_log_stats": bool(param.get("disable_log_stats", False)),
        }
        if param.get("enable_prefix_caching", True):
            engine_kwargs["enable_prefix_caching"] = True
        executor = param.get("distributed_executor_backend", "uni")
        if executor:
            engine_kwargs["distributed_executor_backend"] = str(executor)

        print(
            f"[LLM] Loading vLLM from {model_dir!r} "
            f"(dtype={dtype!r}, max_model_len={engine_kwargs['max_model_len']}, "
            f"prefix_cache={engine_kwargs.get('enable_prefix_caching', False)}, "
            f"executor={engine_kwargs.get('distributed_executor_backend', 'default')}) ...",
            flush=True,
        )
        self.llm = self._build_llm(LLM, engine_kwargs)
        self.tokenizer = _chat_tokenizer(self.llm.get_tokenizer())
        self._sampling_params = self._build_sampling_params(SamplingParams)
        self._generation_calls = 0
        print("[LLM] vLLM engine ready.", flush=True)

    @staticmethod
    def _build_llm(llm_cls: Any, engine_kwargs: Dict[str, Any]) -> Any:
        """Drop newer engine kwargs if this vLLM build does not accept them."""
        attempts = [dict(engine_kwargs)]
        if "distributed_executor_backend" in engine_kwargs:
            stripped = dict(engine_kwargs)
            stripped.pop("distributed_executor_backend", None)
            attempts.append(stripped)
        if "enable_prefix_caching" in engine_kwargs:
            stripped = dict(attempts[-1])
            stripped.pop("enable_prefix_caching", None)
            attempts.append(stripped)
        last_error: Optional[BaseException] = None
        seen = []
        for kwargs in attempts:
            key = tuple(sorted(kwargs))
            if key in seen:
                continue
            seen.append(key)
            try:
                return llm_cls(**kwargs)
            except TypeError as exc:
                last_error = exc
                print(f"[LLM] vLLM rejected {sorted(kwargs.keys())}: {exc}", flush=True)
        assert last_error is not None
        raise last_error

    def _build_sampling_params(self, sampling_cls: Any) -> Any:
        greedy = not (self.do_sample and self.temperature > 0)
        kwargs: Dict[str, Any] = {
            "max_tokens": self.max_new_tokens,
            "stop": list(_TRAFFIC_R1_STOP_STRINGS),
            "skip_special_tokens": True,
        }
        if greedy:
            kwargs["temperature"] = 0.0
        else:
            kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            if self.top_k is not None:
                kwargs["top_k"] = self.top_k
        try:
            return sampling_cls(include_stop_str_in_output=True, **kwargs)
        except TypeError:
            print(
                "[LLM] Warning: this vLLM build has no include_stop_str_in_output; "
                "upgrade vllm so \\boxed{ETWT} remains in the text for the parser.",
                flush=True,
            )
            return sampling_cls(**kwargs)

    def retry_messages(
        self,
        messages: List[Dict[str, str]],
        response: str,
    ) -> List[Dict[str, str]]:
        return [
            *[dict(message) for message in messages],
            {"role": "assistant", "content": str(response)},
            {"role": "user", "content": _API_RETRY_INSTRUCTION},
        ]

    def _complete_one(self, messages: List[Dict[str, str]]) -> str:
        return self.complete_many([messages])[0]

    def complete_many(
        self,
        messages_list: Sequence[List[Dict[str, str]]],
    ) -> List[str]:
        if not messages_list:
            return []

        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
            for messages in messages_list
        ]
        started = time.perf_counter()
        outputs = self.llm.generate(
            prompts, self._sampling_params, use_tqdm=False
        )
        elapsed = time.perf_counter() - started
        responses = [output.outputs[0].text for output in outputs]
        token_counts = [len(output.outputs[0].token_ids) for output in outputs]
        finish = [output.outputs[0].finish_reason for output in outputs]
        self._generation_calls += 1
        completed = sum(
            any(stop in response for stop in _TRAFFIC_R1_STOP_STRINGS)
            for response in responses
        )
        n_stop = sum(reason == "stop" for reason in finish)
        n_length = sum(reason == "length" for reason in finish)
        print(
            f"[LLM vLLM] generate call={self._generation_calls} batch={len(prompts)} "
            f"tokens={min(token_counts)}/{sum(token_counts) / len(token_counts):.1f}/"
            f"{max(token_counts)} (min/mean/max) boxed={completed}/{len(prompts)} "
            f"finish=stop:{n_stop}/length:{n_length} elapsed={elapsed:.2f}s",
            flush=True,
        )
        return responses

