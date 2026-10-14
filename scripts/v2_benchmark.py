"""Measure actual HTTP workloads; require supplied truth and thresholds for qualification."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Dict, List

from v2_http_client import V2Client


def json_pointer(value: Any, pointer: str) -> Any:
    """Read an RFC 6901 pointer from a returned result.

    Args:
        value: Parsed public JSON result from which the pointer is resolved.
        pointer: RFC 6901 pointer selecting an observed value from a public result.

    Returns:
        Selected nested value; an invalid or missing pointer raises a lookup error.
    """
    if not pointer:
        return value
    if not pointer.startswith("/"):
        raise ValueError("Observation paths must be JSON pointers starting with /")
    for component in pointer[1:].split("/"):
        component = component.replace("~1", "/").replace("~0", "~")
        value = value[int(component)] if isinstance(value, list) else value[component]
    return value


def percentile(values: List[float], fraction: float) -> float:
    """Report the nearest-rank percentile, including cold and warm calls.

    Args:
        values: Measured finite latency or quality observations to rank.
        fraction: Percentile fraction between zero and one, using nearest-rank selection.

    Returns:
        Nearest-rank percentile of the supplied observations.
    """
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def execute_case(client: V2Client, case: Dict[str, Any], repeat: int) -> Dict[str, Any]:
    """Measure one complete HTTP submission including input verification and polling.

    Args:
        client: HTTP client that submits and polls the public computation contract.
        case: Qualification case containing route, payload and named measurement pointers.
        repeat: Index of this complete HTTP execution within the repeated input case.

    Returns:
        One case execution with complete HTTP latency, operation state and observed values.
    """
    request = dict(case["request"])
    request.pop("idempotency_key", None)
    started = time.perf_counter()
    try:
        response = client.calculate(case["route"], request)
    except Exception as error:
        return {"name": case["name"], "repeat": repeat, "state": "transport_failed",
                "latency_ms": (time.perf_counter() - started) * 1000, "error": str(error)}
    result = response.get("result") or {}
    observations = {}
    for pointer in case.get("observations", []):
        try:
            observations[pointer] = json_pointer(result, pointer)
        except (KeyError, IndexError, TypeError, ValueError):
            observations[pointer] = None
    return {
        "name": case["name"], "repeat": repeat, "state": response["state"],
        "latency_ms": (time.perf_counter() - started) * 1000,
        "operation_id": response["operation_id"], "observations": observations,
        "target_count": len(result["targets"]) if "targets" in result else None,
        "diagnostics": response.get("diagnostics", {}), "error": response.get("error"),
        "result_provenance": result.get("provenance"), "source": result.get("source"),
        "metadata": result.get("metadata"), "measurement_valid": result.get("valid"),
        "measurement_status": result.get("measurement_status"),
    }


def summarize(case: Dict[str, Any], runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep unknown physical accuracy distinct from repeatability of fixed input bytes.

    Args:
        case: Qualification case containing route, payload and named measurement pointers.
        runs: Recorded HTTP timings, operation states and numerical observations for one
            case.

    Returns:
        Case latency and fixed-file consistency metrics with explicit unknown accuracy.
    """
    latency = [item["latency_ms"] for item in runs]
    observations = {}
    for pointer in case.get("observations", []):
        values = [item.get("observations", {}).get(pointer) for item in runs]
        numbers = [float(value) for value in values
                   if isinstance(value, (int, float)) and not isinstance(value, bool)
                   and math.isfinite(value)]
        info = {"values": values, "same_file_stddev": None,
                "absolute_error_max": None, "rms_error": None, "truth": None,
                "unit": case.get("observation_units", {}).get(pointer)}
        if numbers:
            info["same_file_stddev"] = statistics.pstdev(numbers)
            truth = case.get("truth", {}).get(pointer)
            if truth is not None:
                if not info["unit"] or truth.get("unit") != info["unit"]:
                    raise ValueError(
                        "Truth requires a matching explicit observation unit: " + pointer)
                expected = float(truth["value"])
                errors = [value - expected for value in numbers]
                info.update(truth=truth, absolute_error_max=max(map(abs, errors)),
                            rms_error=math.sqrt(sum(e * e for e in errors) / len(errors)))
        observations[pointer] = info
    peak = [item.get("diagnostics", {}).get("peak_worker_bytes") for item in runs]
    peak = [value for value in peak if isinstance(value, (int, float))]
    return {
        "name": case["name"], "route": case["route"],
        "input_provenance": case.get("input_provenance"),
        "request_sha256": hashlib.sha256(json.dumps(
            case["request"], sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "successes": sum(item["state"] == "succeeded" for item in runs),
        "attempts": len(runs), "latency_p50_ms": percentile(latency, 0.5),
        "latency_p95_ms": percentile(latency, 0.95), "latency_max_ms": max(latency),
        "first_call_ms": runs[0]["latency_ms"], "peak_worker_bytes": max(peak) if peak else None,
        "cache_activity": {key: sum(
            run.get("diagnostics", {}).get("cache", {}).get(key, 0) for run in runs)
            for key in ["hits", "misses", "evictions"]},
        "target_counts": [item.get("target_count") for item in runs],
        "observations": observations,
    }


def qualify(summary: List[Dict[str, Any]], thresholds: Dict[str, Any],
            concurrency: int) -> Dict[str, Any]:
    """Evaluate only caller-supplied thresholds and disclose incomplete evidence.

    Args:
        summary: Per-case observed timing and fixed-file consistency summaries.
        thresholds: Upstream-supplied limits for accuracy, repeatability, latency and
            throughput.
        concurrency: Number of parallel case executions used by the qualification run.

    Returns:
        Qualification decisions and disclosed evidence gaps for supplied thresholds.
    """
    checks = []
    checks.append({"metric": "all_operations_succeeded", "passed": all(
        case["successes"] == case["attempts"] for case in summary)})
    allowed = {"max_latency_p95_ms", "max_peak_worker_bytes", "max_same_file_stddev",
               "max_absolute_error", "min_target_count", "min_concurrency"}
    if set(thresholds) - allowed:
        raise ValueError("Unsupported qualification threshold: " + str(set(thresholds) - allowed))
    for metric, limit in thresholds.items():
        if metric == "min_concurrency":
            values = [concurrency]
        elif metric == "max_latency_p95_ms":
            values = [case["latency_p95_ms"] for case in summary]
        elif metric == "max_peak_worker_bytes":
            values = [case["peak_worker_bytes"] for case in summary]
        elif metric == "min_target_count":
            values = [value for case in summary for value in case["target_counts"]]
        else:
            key = "same_file_stddev" if metric == "max_same_file_stddev" else "absolute_error_max"
            values = [observation[key] for case in summary
                      for observation in case["observations"].values()]
        known = bool(values) and all(value is not None for value in values)
        passed = None if not known else all(
            value >= limit if metric.startswith("min_") else value <= limit for value in values)
        checks.append({"metric": metric, "limit": limit, "values": values, "passed": passed})
    state = "pending" if not thresholds or any(item["passed"] is None for item in checks) else (
        "pass" if all(item["passed"] for item in checks) else "fail")
    return {"status": state, "checks": checks,
            "scope": "Supplied thresholds and unchanged-file workloads only; capture repeatability "
                     "and physical accuracy require representative inputs and traceable truth."}


def main() -> None:
    """Execute qualification cases concurrently and save an auditable JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:9004")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--inventory", type=Path,
                        help="Deployment hardware/container/worker/cache settings JSON")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.concurrency < 1:
        parser.error("repeats and concurrency must be positive")
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not cases or len({case["name"] for case in cases}) != len(cases):
        parser.error("cases must have unique nonempty names")
    thresholds = json.loads(args.thresholds.read_text()) if args.thresholds else {}
    client = V2Client(args.base_url, "qualification")
    capabilities = client.request("/capabilities")
    tasks = [(case, repeat) for repeat in range(args.repeats) for case in cases]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        runs = list(pool.map(lambda task: execute_case(client, *task), tasks))
    summary = [summarize(case, [run for run in runs if run["name"] == case["name"]])
               for case in cases]
    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(), "concurrency": args.concurrency,
        "client_platform": platform.platform(), "client_machine": platform.machine(),
        "client_python": platform.python_version(),
        "deployment_inventory": (json.loads(args.inventory.read_text())
                                 if args.inventory else None),
        "service_version": {key: value for key, value in capabilities.items()
                            if key != "operations"},
        "service_container_peak_bytes": None, "thresholds": thresholds,
        "qualification": qualify(summary, thresholds, args.concurrency), "summary": summary,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["qualification"], indent=2))
    if report["qualification"]["status"] == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
