from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from local_moe.performance import (
    BenchmarkCandidate,
    BenchmarkManifest,
    load_benchmark_manifest,
    render_markdown_report,
    summarize_benchmarks,
)
from local_moe.hardware import write_hardware_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local MLX model candidates.")
    parser.add_argument("--manifest", default="configs/model-benchmark.json")
    parser.add_argument("--out", default="outputs/performance-benchmark.json")
    parser.add_argument("--report", default="outputs/performance-decision.md")
    parser.add_argument("--include", default="", help="Comma-separated candidate ids.")
    parser.add_argument("--prompt-limit", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-kv-size", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--min-free-gb", type=float, default=25.0)
    parser.add_argument("--run-one", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    manifest = load_benchmark_manifest(args.manifest)
    prompts = manifest.prompts[: args.prompt_limit or None]

    if args.run_one:
        candidate = _find_candidate(manifest, args.run_one)
        print(json.dumps(run_one_candidate(candidate, prompts, args.max_tokens, args.max_kv_size), indent=2))
        return

    include = {item.strip() for item in args.include.split(",") if item.strip()}
    results: list[dict[str, Any]] = []
    for candidate in manifest.candidates:
        if include and candidate.id not in include:
            continue
        if candidate.runtime not in {"mlx_lm", "mlx_vlm"}:
            results.append(_skipped(candidate, f"unsupported runtime: {candidate.runtime}"))
            continue
        if not _has_disk_headroom(candidate.estimated_size_gb, args.min_free_gb):
            results.append(_skipped(candidate, "not enough disk headroom"))
            continue
        results.append(_run_isolated(args, candidate.id))

    hardware = write_hardware_report("outputs/hardware-profile.json")
    summary = summarize_benchmarks(manifest, results)
    summary["hardware"] = hardware.__dict__
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "manifest": args.manifest,
        "hardware": hardware.__dict__,
        "max_tokens": args.max_tokens,
        "max_kv_size": args.max_kv_size,
        "prompt_count": len(prompts),
        "results": results,
        "summary": summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown_report(summary), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "report": str(report_path), "results": len(results)}, indent=2))


def run_one_candidate(
    candidate: BenchmarkCandidate,
    prompts: list[Any],
    max_tokens: int,
    max_kv_size: int,
) -> dict[str, Any]:
    if candidate.runtime == "mlx_vlm":
        return _run_mlx_vlm_candidate(candidate, prompts, max_tokens, max_kv_size)
    return _run_mlx_lm_candidate(candidate, prompts, max_tokens, max_kv_size)


def _run_mlx_lm_candidate(
    candidate: BenchmarkCandidate,
    prompts: list[Any],
    max_tokens: int,
    max_kv_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from mlx_lm import load
        from mlx_lm.generate import stream_generate
        from mlx_lm.sample_utils import make_sampler
        import mlx.core as mx
    except Exception as exc:
        return _failed(candidate, "import_failed", str(exc), started)

    try:
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        load_started = time.perf_counter()
        model, tokenizer = load(candidate.repo)
        load_seconds = time.perf_counter() - load_started
    except Exception as exc:
        return _failed(candidate, "load_failed", str(exc), started)

    records = []
    sampler = make_sampler(temp=0.2, top_p=0.9)
    for prompt in prompts:
        prompt_started = time.perf_counter()
        try:
            formatted = _format_prompt(tokenizer, prompt.prompt)
            text_parts = []
            last_response = None
            for response in stream_generate(
                model,
                tokenizer,
                formatted,
                max_tokens=max_tokens,
                sampler=sampler,
                max_kv_size=max_kv_size,
            ):
                text_parts.append(response.text)
                last_response = response
            elapsed = time.perf_counter() - prompt_started
            records.append(
                {
                    "prompt_id": prompt.id,
                    "category": prompt.category,
                    "status": "ok",
                    "latency_seconds": round(elapsed, 3),
                    "prompt_tokens": getattr(last_response, "prompt_tokens", None),
                    "generation_tokens": getattr(last_response, "generation_tokens", None),
                    "prompt_tps": _round(getattr(last_response, "prompt_tps", None)),
                    "generation_tps": _round(getattr(last_response, "generation_tps", None)),
                    "peak_memory_gb": _round(getattr(last_response, "peak_memory", None)),
                    "content_excerpt": "".join(text_parts)[:500],
                }
            )
        except Exception as exc:
            records.append(
                {
                    "prompt_id": prompt.id,
                    "category": prompt.category,
                    "status": "failed",
                    "latency_seconds": round(time.perf_counter() - prompt_started, 3),
                    "error": str(exc),
                }
            )

    return {
        "candidate_id": candidate.id,
        "label": candidate.label,
        "repo": candidate.repo,
        "role": candidate.role,
        "runtime": candidate.runtime,
        "status": "ok" if any(item["status"] == "ok" for item in records) else "failed",
        "load_seconds": round(load_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
        "aggregate": _aggregate(records),
        "records": records,
    }


def _run_mlx_vlm_candidate(
    candidate: BenchmarkCandidate,
    prompts: list[Any],
    max_tokens: int,
    max_kv_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from mlx_vlm import generate, load
        import mlx.core as mx
    except Exception as exc:
        return _failed(candidate, "import_failed", str(exc), started)

    try:
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        load_started = time.perf_counter()
        model, processor = load(candidate.repo)
        load_seconds = time.perf_counter() - load_started
    except Exception as exc:
        return _failed(candidate, "load_failed", str(exc), started)

    records = []
    for prompt in prompts:
        prompt_started = time.perf_counter()
        try:
            formatted = _format_vlm_prompt(processor, prompt.prompt)
            response = generate(
                model,
                processor,
                formatted,
                max_tokens=max_tokens,
                temperature=0.2,
                max_kv_size=max_kv_size,
                verbose=False,
            )
            elapsed = time.perf_counter() - prompt_started
            generation_tokens = getattr(response, "generation_tokens", None)
            generation_tps = _round(getattr(response, "generation_tps", None))
            if not generation_tps and generation_tokens:
                generation_tps = _round(float(generation_tokens) / max(elapsed, 0.001))
            records.append(
                {
                    "prompt_id": prompt.id,
                    "category": prompt.category,
                    "status": "ok",
                    "latency_seconds": round(elapsed, 3),
                    "prompt_tokens": getattr(response, "prompt_tokens", None),
                    "generation_tokens": generation_tokens,
                    "prompt_tps": _round(getattr(response, "prompt_tps", None)),
                    "generation_tps": generation_tps,
                    "peak_memory_gb": _round(getattr(response, "peak_memory", None)),
                    "content_excerpt": str(getattr(response, "text", ""))[:500],
                }
            )
        except Exception as exc:
            records.append(
                {
                    "prompt_id": prompt.id,
                    "category": prompt.category,
                    "status": "failed",
                    "latency_seconds": round(time.perf_counter() - prompt_started, 3),
                    "error": str(exc),
                }
            )

    return {
        "candidate_id": candidate.id,
        "label": candidate.label,
        "repo": candidate.repo,
        "role": candidate.role,
        "runtime": candidate.runtime,
        "status": "ok" if any(item["status"] == "ok" for item in records) else "failed",
        "load_seconds": round(load_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
        "aggregate": _aggregate(records),
        "records": records,
    }


def _run_isolated(args: argparse.Namespace, candidate_id: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        __file__,
        "--manifest",
        args.manifest,
        "--prompt-limit",
        str(args.prompt_limit),
        "--max-tokens",
        str(args.max_tokens),
        "--max-kv-size",
        str(args.max_kv_size),
        "--run-one",
        candidate_id,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            timeout=args.timeout_seconds,
            env={**os.environ, "PYTHONPATH": "src"},
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "candidate_id": candidate_id,
            "status": "failed",
            "error_type": "timeout",
            "error": f"candidate benchmark timed out after {args.timeout_seconds} seconds",
            "stdout": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
            "total_seconds": round(time.perf_counter() - started, 3),
        }
    if completed.returncode != 0:
        return {
            "candidate_id": candidate_id,
            "status": "failed",
            "error_type": "subprocess_failed",
            "error": completed.stderr[-4000:],
            "stdout": completed.stdout[-2000:],
            "total_seconds": round(time.perf_counter() - started, 3),
        }
    parsed = _parse_json_stdout(completed.stdout)
    if parsed is not None:
        return parsed
    return {
        "candidate_id": candidate_id,
        "status": "failed",
        "error_type": "invalid_json",
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "total_seconds": round(time.perf_counter() - started, 3),
    }


def _parse_json_stdout(stdout: str) -> dict[str, Any] | None:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        pass

    marker = '{\n  "candidate_id"'
    start = stdout.find(marker)
    if start < 0:
        return None
    try:
        return json.loads(stdout[start:])
    except json.JSONDecodeError:
        return None


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [item for item in records if item.get("status") == "ok"]
    if not ok:
        return {"successful_prompts": 0, "failed_prompts": len(records)}
    return {
        "successful_prompts": len(ok),
        "failed_prompts": len(records) - len(ok),
        "latency_seconds_avg": _avg(item.get("latency_seconds") for item in ok),
        "generation_tps_avg": _avg(item.get("generation_tps") for item in ok),
        "prompt_tps_avg": _avg(item.get("prompt_tps") for item in ok),
        "peak_memory_gb": max(float(item.get("peak_memory_gb") or 0.0) for item in ok),
        "generation_tokens_total": sum(int(item.get("generation_tokens") or 0) for item in ok),
    }


def _format_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a concise local general-purpose assistant. response in English only for this benchmark.",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except Exception:
        try:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            return f"System: {messages[0]['content']}\nUser: {prompt}\nAssistant:"


def _format_vlm_prompt(processor: Any, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a concise local general-purpose assistant. response in English only for this benchmark.",
        },
        {"role": "user", "content": prompt},
    ]
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except Exception:
        try:
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except Exception:
            return f"System: {messages[0]['content']}\nUser: {prompt}\nAssistant:"


def _find_candidate(manifest: BenchmarkManifest, candidate_id: str) -> BenchmarkCandidate:
    for candidate in manifest.candidates:
        if candidate.id == candidate_id:
            return candidate
    raise SystemExit(f"Unknown candidate: {candidate_id}")


def _has_disk_headroom(model_size_gb: float, min_free_gb: float) -> bool:
    free_gb = shutil.disk_usage(Path.cwd()).free / 1024**3
    return free_gb - model_size_gb >= min_free_gb


def _skipped(candidate: BenchmarkCandidate, reason: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate.id,
        "label": candidate.label,
        "repo": candidate.repo,
        "role": candidate.role,
        "status": "skipped",
        "reason": reason,
    }


def _failed(candidate: BenchmarkCandidate, error_type: str, error: str, started: float) -> dict[str, Any]:
    return {
        "candidate_id": candidate.id,
        "label": candidate.label,
        "repo": candidate.repo,
        "role": candidate.role,
        "status": "failed",
        "error_type": error_type,
        "error": error,
        "total_seconds": round(time.perf_counter() - started, 3),
    }


def _avg(values: Any) -> float:
    parsed = [float(value) for value in values if value is not None]
    if not parsed:
        return 0.0
    return round(sum(parsed) / len(parsed), 3)


def _round(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
