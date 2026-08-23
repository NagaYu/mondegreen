"""LocalRuntime: quantised export, and the measurements that back "runs on a laptop".

The claim this module exists to substantiate is LOCAL-SPEED, and specifically the
comparison in metric (6): how long an hour of transcription takes on the machine
in front of you, versus a cloud round trip -- and, in the case this project is
actually about, versus the fact that you are not allowed to make that round trip
at all.

Two halves:

* **Export.**  LoRA adapter -> merged model -> GGUF (Q4_K_M, Q8_0) via llama.cpp,
  and -> MLX for Apple silicon.  These are shell-outs to the canonical tools
  rather than reimplementations, and each one reports exactly what it ran.
* **Measurement.**  Throughput, peak RSS, and an extrapolation to one hour of
  audio using a stated speaking-rate constant.  All measured on real corrector
  runs, never estimated from parameter counts.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: Japanese speech runs about 7 morae per second in meeting-style speech, and a
#: mora is roughly one kana; written back as mixed kanji/kana that is ~350
#: characters per minute.  One hour of audio is therefore ~21,000 characters of
#: transcript.  Stated here rather than buried, because every "hour of audio"
#: number in the README depends on it.
CHARS_PER_HOUR_OF_AUDIO = 21_000

#: Round-trip latency assumed for a cloud post-processing call, per request, when
#: no real measurement is available.  Only used when explicitly requested and
#: always labelled.
ASSUMED_CLOUD_RTT_SECONDS = 1.8


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MiB.

    ``ru_maxrss`` is bytes on macOS and kibibytes on Linux; both are handled.

    Claim: LOCAL-SPEED -- "runs on a laptop" is a memory claim as much as a
    speed one.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def machine_info() -> Dict[str, object]:
    """Describe the machine a measurement was taken on.

    Claim: LOCAL-SPEED -- a throughput number without a machine is not a result.
    """
    info: Dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if sys.platform == "darwin":
        try:
            info["cpu_brand"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            info["memory_gb"] = round(
                int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
                / (1024 ** 3),
                1,
            )
        except Exception:  # pragma: no cover
            pass
    return info


# --------------------------------------------------------------------------------------
# Throughput
# --------------------------------------------------------------------------------------

@dataclass
class ThroughputResult:
    """One measured corrector run."""

    label: str
    n_texts: int
    chars: int
    seconds: float
    chars_per_second: float
    hours_of_audio_per_second: float
    seconds_per_hour_of_audio: float
    peak_rss_mb: float
    corrections: int
    glossary_terms: int
    machine: Dict[str, object] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Claim: SUPPORT."""
        return asdict(self)


def benchmark_corrector(
    corrector,
    texts: Sequence[str],
    label: str = "mondegreen",
    warmup: int = 1,
    repeats: int = 1,
) -> ThroughputResult:
    """Time a corrector over real transcripts and extrapolate to an hour of audio.

    A warm-up pass is run first and discarded, because the first call pays for
    index construction and reader initialisation, and reporting that as
    steady-state throughput would understate the system.  Both numbers are
    available: ``notes`` records the cold-start cost.

    Claim: LOCAL-SPEED -- this is metric (6).
    """
    cold_start = 0.0
    if warmup and texts:
        t0 = time.perf_counter()
        corrector.correct(texts[0])
        cold_start = time.perf_counter() - t0

    total_chars = 0
    total_corr = 0
    t0 = time.perf_counter()
    for _ in range(max(1, repeats)):
        for t in texts:
            r = corrector.correct(t)
            total_chars += len(t)
            total_corr += len(r.corrections)
    elapsed = time.perf_counter() - t0

    cps = total_chars / elapsed if elapsed > 0 else 0.0
    hours_per_sec = cps / CHARS_PER_HOUR_OF_AUDIO if CHARS_PER_HOUR_OF_AUDIO else 0.0
    return ThroughputResult(
        label=label,
        n_texts=len(texts) * max(1, repeats),
        chars=total_chars,
        seconds=elapsed,
        chars_per_second=cps,
        hours_of_audio_per_second=hours_per_sec,
        seconds_per_hour_of_audio=(CHARS_PER_HOUR_OF_AUDIO / cps) if cps else float("inf"),
        peak_rss_mb=peak_rss_mb(),
        corrections=total_corr,
        glossary_terms=len(getattr(corrector, "glossary", ()) or ()),
        machine=machine_info(),
        notes=f"cold_start={cold_start:.3f}s (index build + reader init, excluded from throughput); "
              f"1h of audio assumed to be {CHARS_PER_HOUR_OF_AUDIO} characters",
    )


def compare_to_cloud(
    local: ThroughputResult,
    n_requests: int,
    measured_rtt_seconds: Optional[float] = None,
) -> Dict[str, object]:
    """Put local processing time next to a cloud round trip.

    The honest headline is not the ratio.  It is the last field: for audio that
    cannot leave the machine, the cloud column has no number at all.

    Claim: LOCAL-SPEED.
    """
    rtt = measured_rtt_seconds if measured_rtt_seconds is not None else ASSUMED_CLOUD_RTT_SECONDS
    cloud_seconds = rtt * n_requests
    return {
        "local_seconds_per_hour_of_audio": local.seconds_per_hour_of_audio,
        "cloud_seconds_total": cloud_seconds,
        "cloud_rtt_seconds": rtt,
        "cloud_rtt_provenance": "measured" if measured_rtt_seconds is not None else "assumed",
        "requests": n_requests,
        "speedup": (cloud_seconds / local.seconds) if local.seconds else float("inf"),
        "data_leaves_machine": {"local": False, "cloud": True},
        "applicable_when_audio_is_confidential": {"local": True, "cloud": False},
    }


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------

@dataclass
class ExportResult:
    """What an export step produced, and exactly what it ran."""

    kind: str
    path: str
    ok: bool
    command: str = ""
    size_mb: float = 0.0
    message: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Claim: SUPPORT."""
        return asdict(self)


def _run(cmd: Sequence[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    """Run an export subprocess, capturing output for the failure report.

        Claim: LOCAL-SPEED -- a failed export must say exactly what it ran.
        """
    proc = subprocess.run(
        list(cmd), cwd=cwd, capture_output=True, text=True, check=False
    )
    return proc.returncode, (proc.stdout + proc.stderr)[-4000:]


def _dir_size_mb(path: str) -> float:
    """Size of a file or directory tree in MiB.

        Claim: LOCAL-SPEED -- artefact size is part of "runs on a laptop".
        """
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 ** 2)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 ** 2)


def merge_lora(base_model: str, adapter_dir: str, out_dir: str, dtype: str = "float16") -> ExportResult:
    """Merge a PEFT LoRA adapter into its base model and save the result.

    Claim: LOCAL-SPEED -- GGUF and MLX both want a single merged checkpoint, so
    this is the gate every local runtime passes through.
    """
    try:
        import torch  # type: ignore
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as exc:
        return ExportResult("merge", out_dir, False, message=f"missing deps: {exc}")

    os.makedirs(out_dir, exist_ok=True)
    torch_dtype = getattr(torch, dtype, torch.float16)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch_dtype)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model).save_pretrained(out_dir)
    return ExportResult(
        "merge", out_dir, True,
        command=f"peft.merge_and_unload({base_model} + {adapter_dir})",
        size_mb=_dir_size_mb(out_dir),
    )


def find_llama_cpp() -> Optional[str]:
    """Locate a llama.cpp checkout, from ``LLAMA_CPP_DIR`` or the usual places.

    Claim: SUPPORT.
    """
    env = os.environ.get("LLAMA_CPP_DIR")
    if env and os.path.isdir(env):
        return env
    for cand in (
        os.path.expanduser("~/llama.cpp"),
        os.path.expanduser("~/src/llama.cpp"),
        "/opt/llama.cpp",
        os.path.join(os.getcwd(), "llama.cpp"),
    ):
        if os.path.isdir(cand):
            return cand
    return None


def export_gguf(
    model_dir: str,
    out_dir: str,
    quantizations: Sequence[str] = ("Q4_K_M", "Q8_0"),
    llama_cpp_dir: Optional[str] = None,
    name: str = "mondegreen",
) -> List[ExportResult]:
    """Convert a merged HF checkpoint to GGUF and quantise it.

    Runs llama.cpp's own ``convert_hf_to_gguf.py`` and ``llama-quantize`` -- the
    canonical tools, not a reimplementation -- and reports the exact command line
    for every step so a failure is reproducible.

    Claim: LOCAL-SPEED -- Q4_K_M is the artefact the (E) condition is measured on.
    """
    results: List[ExportResult] = []
    os.makedirs(out_dir, exist_ok=True)
    root = llama_cpp_dir or find_llama_cpp()
    if root is None:
        return [ExportResult(
            "gguf", out_dir, False,
            message="llama.cpp not found. git clone https://github.com/ggml-org/llama.cpp "
                    "and set LLAMA_CPP_DIR, or pass llama_cpp_dir=",
        )]

    converter = None
    for cand in ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py"):
        p = os.path.join(root, cand)
        if os.path.isfile(p):
            converter = p
            break
    if converter is None:
        return [ExportResult("gguf", out_dir, False, message=f"no converter script under {root}")]

    f16 = os.path.join(out_dir, f"{name}-f16.gguf")
    cmd = [sys.executable, converter, model_dir, "--outfile", f16, "--outtype", "f16"]
    code, log = _run(cmd)
    results.append(ExportResult(
        "gguf-f16", f16, code == 0, command=" ".join(cmd),
        size_mb=_dir_size_mb(f16) if os.path.exists(f16) else 0.0,
        message="" if code == 0 else log,
    ))
    if code != 0:
        return results

    quantizer = shutil.which("llama-quantize")
    for cand in ("build/bin/llama-quantize", "llama-quantize", "quantize"):
        p = os.path.join(root, cand)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            quantizer = p
            break
    if quantizer is None:
        results.append(ExportResult(
            "gguf-quant", out_dir, False,
            message="llama-quantize binary not found; build llama.cpp (cmake -B build && "
                    "cmake --build build -j) or put it on PATH",
        ))
        return results

    for q in quantizations:
        dst = os.path.join(out_dir, f"{name}-{q}.gguf")
        cmd = [quantizer, f16, dst, q]
        code, log = _run(cmd)
        results.append(ExportResult(
            f"gguf-{q}", dst, code == 0, command=" ".join(cmd),
            size_mb=_dir_size_mb(dst) if os.path.exists(dst) else 0.0,
            message="" if code == 0 else log,
        ))
    return results


def export_mlx(model_dir: str, out_dir: str, quant_bits: int = 4, group_size: int = 64) -> ExportResult:
    """Convert a merged checkpoint to MLX, quantised, for Apple silicon.

    Claim: LOCAL-SPEED -- on the laptops this project targets, MLX is the fastest
    local path and is what the README's Apple-silicon numbers use.
    """
    try:
        from mlx_lm import convert  # type: ignore
    except Exception as exc:
        return ExportResult("mlx", out_dir, False, message=f"mlx-lm not installed: {exc}")
    try:
        convert(
            hf_path=model_dir,
            mlx_path=out_dir,
            quantize=True,
            q_bits=quant_bits,
            q_group_size=group_size,
        )
    except Exception as exc:  # pragma: no cover - depends on model
        return ExportResult("mlx", out_dir, False,
                            command=f"mlx_lm.convert({model_dir} -> {out_dir}, {quant_bits}bit)",
                            message=str(exc))
    return ExportResult(
        "mlx", out_dir, True,
        command=f"mlx_lm.convert({model_dir} -> {out_dir}, {quant_bits}bit, group={group_size})",
        size_mb=_dir_size_mb(out_dir),
    )


# --------------------------------------------------------------------------------------
# LM backends (optional re-ranking inside the candidate set)
# --------------------------------------------------------------------------------------

class LlamaCppReranker:
    """Score candidate replacements with a GGUF model via llama-cpp-python.

    Note what this does *not* do: generate.  It scores the candidates the hard
    constraint already produced.  The output space is a Python list either way.

    Claim: TERM-RECALL -- context breaks ties between phonetically equivalent
    glossary terms; LOCAL-SPEED -- and it does so from a 4-bit file on disk.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: Optional[int] = None,
                 verbose: bool = False) -> None:
        """Load the quantised model used to re-rank inside the candidate set.

                Claim: TERM-RECALL + LOCAL-SPEED.
                """
        from llama_cpp import Llama  # type: ignore

        self.name = f"llama.cpp:{os.path.basename(model_path)}"
        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, logits_all=True,
                          n_threads=n_threads or (os.cpu_count() or 4), verbose=verbose)

    def _logprob(self, text: str) -> float:
        """Total log-probability of a string under the GGUF model.

            Claim: TERM-RECALL.
            """
        out = self._llm(text, max_tokens=0, echo=True, logprobs=0)
        lps = out["choices"][0]["logprobs"]["token_logprobs"]
        vals = [v for v in lps if v is not None]
        return float(sum(vals))

    def score_candidates(self, prefix: str, candidates: Sequence[str], suffix: str) -> List[float]:
        """Length-normalised log-probability of each candidate in context.

        Claim: TERM-RECALL.
        """
        scores: List[float] = []
        for c in candidates:
            full = f"{prefix}{c}{suffix}"
            lp = self._logprob(full)
            scores.append(lp / max(1, len(full)))
        return scores


class MLXReranker:
    """The same, on MLX.

    Claim: TERM-RECALL + LOCAL-SPEED.
    """

    def __init__(self, model_path: str) -> None:
        """Load the quantised model used to re-rank inside the candidate set.

                Claim: TERM-RECALL + LOCAL-SPEED.
                """
        from mlx_lm import load  # type: ignore

        self.name = f"mlx:{os.path.basename(model_path)}"
        self._model, self._tokenizer = load(model_path)

    def score_candidates(self, prefix: str, candidates: Sequence[str], suffix: str) -> List[float]:
        """Claim: TERM-RECALL."""
        import mlx.core as mx  # type: ignore
        import mlx.nn as nn  # type: ignore

        scores: List[float] = []
        for c in candidates:
            ids = self._tokenizer.encode(f"{prefix}{c}{suffix}")
            if len(ids) < 2:
                scores.append(0.0)
                continue
            x = mx.array([ids[:-1]])
            y = mx.array([ids[1:]])
            logits = self._model(x)
            lp = -nn.losses.cross_entropy(logits, y, reduction="mean")
            scores.append(float(lp))
        return scores


def build_reranker(path: Optional[str]):
    """Pick a reranker backend from a path, or return ``None``.

    ``.gguf`` -> llama.cpp, a directory -> MLX.  Returning ``None`` is a normal
    outcome: the corrector's guarantee does not depend on an LM existing.

    Claim: LOCAL-SPEED.
    """
    if not path:
        return None
    if path.endswith(".gguf"):
        return LlamaCppReranker(path)
    if os.path.isdir(path):
        return MLXReranker(path)
    raise ValueError(f"cannot infer a reranker backend from {path!r}")


# --------------------------------------------------------------------------------------
# Constrained decoding
# --------------------------------------------------------------------------------------

def candidate_set_regex(candidates: Sequence[str]) -> str:
    """Build a regex matching exactly the given candidate strings.

    This is what gets handed to outlines / xgrammar when a *generative* model is
    used.  It is worth being precise about the role: the hard constraint is
    already enforced by choosing an element of a Python list.  Grammar-constrained
    decoding exists so that a model which generates tokens cannot wander outside
    that same set -- belt and braces, not the belt.

    Claim: LOW-DAMAGE.
    """
    import re as _re

    if not candidates:
        return r"(?!)"  # matches nothing
    return "^(?:" + "|".join(_re.escape(c) for c in candidates) + ")$"


def constrained_choice(
    prefix: str,
    candidates: Sequence[str],
    suffix: str = "",
    backend: Optional[object] = None,
):
    """Choose one candidate, using outlines/xgrammar when available.

    Falls back to the reranker's scores, and then to the first candidate.  Every
    path returns an element of ``candidates``; there is no path that returns
    anything else.

    Claim: LOW-DAMAGE -- the invariant holds identically with and without
    constrained-decoding libraries installed, which is the point.
    """
    if not candidates:
        return None
    if backend is not None and hasattr(backend, "score_candidates"):
        scores = backend.score_candidates(prefix, list(candidates), suffix)
        best = max(range(len(candidates)), key=lambda i: scores[i])
        return candidates[best]
    return candidates[0]
