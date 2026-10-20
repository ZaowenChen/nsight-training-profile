#!/usr/bin/env python3
"""Capture and inspect training traces without an LLM service or training dependencies."""

import argparse
from contextlib import closing
import datetime as dt
import heapq
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys


PACK = (
    "profile_health_manifest", "top_kernels", "gpu_idle_gaps",
    "memory_transfers", "kernel_launch_overhead", "nvtx_layer_breakdown",
    "overlap_breakdown",
)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def tool(name):
    override = os.environ.get({"nsys": "NSIGHT_SYSTEMS_BIN", "nsys-ai": "NSYS_AI_BIN"}[name])
    if override:
        return str(Path(override).expanduser().resolve())
    local = Path.home() / ".local/bin" / name
    return str(local) if local.is_file() else (shutil.which(name) or name)


def environment():
    probes = {
        "nsys_version": [tool("nsys"), "--version"],
        "native_sampling": [tool("nsys"), "status", "--environment"],
        "gpu": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        "gpu_processes": ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
    }
    ai_python = Path(tool("nsys-ai")).resolve().parent / "python"
    if ai_python.is_file():
        probes["nsys_ai_version"] = [str(ai_python), "-c", "from importlib.metadata import version; print(version('nsys-ai'))"]
    result = {"tools": {name: tool(name) for name in ("nsys", "nsys-ai")}, "probes": {}}
    for name, command in probes.items():
        try:
            r = subprocess.run(command, capture_output=True, text=True, timeout=20)
            result["probes"][name] = {"exit_code": r.returncode, "text": (r.stdout + r.stderr)[-18000:]}
        except (OSError, subprocess.TimeoutExpired) as e:
            result["probes"][name] = {"error": str(e)}
    text = result["probes"]["native_sampling"].get("text", "")
    result["native_cpu_sampling_available"] = bool(re.search(
        r"CPU Profiling Environment\s*\(process-tree\)\s*:\s*(?:OK|Pass|Supported|Available)\b", text, re.I))
    paranoid = Path("/proc/sys/kernel/perf_event_paranoid")
    result["perf_event_paranoid"] = paranoid.read_text().strip() if paranoid.exists() else None
    return result


def new_output(path):
    out = Path(path).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def saved(command, output, timeout=180):
    output = Path(output)
    error = output.with_name(output.name + ".stderr.txt")
    result = {"command": command, "stdout": output.name, "stderr": error.name}
    env = dict(os.environ)
    env["NSYS_AI_CACHE_MODE"] = "direct"
    # Native exports made by upstream tools must also resolve the selected Nsight.
    env["PATH"] = str(Path(tool("nsys")).parent) + os.pathsep + env.get("PATH", "")
    try:
        with output.open("w") as stdout, error.open("w") as stderr:
            r = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout, env=env)
        result["exit_code"] = r.returncode
    except (OSError, subprocess.TimeoutExpired) as e:
        result.update(exit_code=None, error=str(e))
    if output.exists() and output.stat().st_size == 0:
        result["empty_output"] = True
    return result


def quote(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def cpu_sample_summary(conn, tables, columns):
    """Summarize periodic CPU samples, excluding scheduling-only callchains."""
    events, frames = "COMPOSITE_EVENTS", "SAMPLING_CALLCHAINS"
    if events not in tables:
        return {"status": "not_recorded", "periodic_cpu_samples": None,
                "reason": "No COMPOSITE_EVENTS table; native CPU sampling is not established"}
    if not {"id", "start", "globalTid", "cpuCycles"} <= columns[events]:
        return {"status": "unsupported_schema", "periodic_cpu_samples": None,
                "reason": "COMPOSITE_EVENTS lacks required sampling fields"}
    total = conn.execute('SELECT COUNT(*) FROM COMPOSITE_EVENTS').fetchone()[0]
    samples = conn.execute('SELECT COUNT(*) FROM COMPOSITE_EVENTS WHERE cpuCycles=1').fetchone()[0]
    result = {"status": "ok" if samples else "no_periodic_samples", "periodic_cpu_samples": samples,
              "other_composite_events": total - samples, "scope": "All recorded threads; periodic samples only (cpuCycles=1)",
              "metric": "Relative sample counts, not measured function milliseconds or CPU utilization"}
    if not samples:
        return result
    result["sample_window_ns"] = list(conn.execute('SELECT MIN(start),MAX(start) FROM COMPOSITE_EVENTS WHERE cpuCycles=1').fetchone())
    result["threads"] = [{"global_tid": tid, "samples": count}
                         for tid, count in conn.execute('SELECT globalTid,COUNT(*) FROM COMPOSITE_EVENTS WHERE cpuCycles=1 GROUP BY globalTid ORDER BY 2 DESC LIMIT 20')]
    if frames not in tables or not {"id", "symbol", "module", "stackDepth"} <= columns[frames]:
        result.update(status="samples_without_supported_callchains", samples_with_callchains=None)
        return result
    result["samples_with_callchains"] = conn.execute('SELECT COUNT(DISTINCT sc.id) FROM SAMPLING_CALLCHAINS sc JOIN COMPOSITE_EVENTS e ON sc.id=e.id WHERE e.cpuCycles=1').fetchone()[0]
    symbol, module, joins = "CAST(sc.symbol AS TEXT)", "CAST(sc.module AS TEXT)", ""
    if "StringIds" in tables and {"id", "value"} <= columns["StringIds"]:
        symbol = "COALESCE(s.value,CAST(sc.symbol AS TEXT),'<unresolved>')"
        module = "COALESCE(m.value,CAST(sc.module AS TEXT),'<unresolved>')"
        joins = " LEFT JOIN StringIds s ON sc.symbol=s.id LEFT JOIN StringIds m ON sc.module=m.id"
    for name, condition in (("self_hotspots", "AND sc.stackDepth=0"), ("inclusive_hotspots", "")):
        query = (f'SELECT {symbol},{module},COUNT(DISTINCT sc.id) FROM SAMPLING_CALLCHAINS sc '
                 f'JOIN COMPOSITE_EVENTS e ON sc.id=e.id{joins} WHERE e.cpuCycles=1 {condition} '
                 'GROUP BY sc.symbol,sc.module ORDER BY 3 DESC LIMIT 20')
        result[name] = [{"symbol": sym, "module": mod, "samples": count,
                         "share_of_all_periodic_samples": count / samples}
                        for sym, mod, count in conn.execute(query)]
    return result


def native_summary(path):
    """Stream a union of intervals per observed device; never sum as wall time."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        columns = {t: {r[1] for r in conn.execute(f"PRAGMA table_info({quote(t)})")} for t in tables}
        warnings, selected, unions = [], {}, []
        for kind, candidates in {
            "kernel": ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"),
            "memcpy": ("CUPTI_ACTIVITY_KIND_MEMCPY",),
            "memset": ("CUPTI_ACTIVITY_KIND_MEMSET",),
        }.items():
            found = [t for t in candidates if t in tables and {"start", "end"} <= columns[t]]
            t = next((t for t in found if conn.execute(f"SELECT 1 FROM {quote(t)} LIMIT 1").fetchone()), None)
            if not t:
                selected[kind] = None
                continue
            selected[kind] = t
            device = quote("deviceId") if "deviceId" in columns[t] else "NULL"
            if device == "NULL":
                warnings.append(f"{t} has no deviceId; device attribution is unavailable")
            unions.append(f'SELECT {device} AS device, start, "end" FROM {quote(t)} WHERE "end">start')
        states = {}
        if unions:
            query = "SELECT device,start,\"end\" FROM (" + " UNION ALL ".join(unions) + ") ORDER BY device,start,\"end\""
            for device, start, end in conn.execute(query):
                start, end = int(start), int(end)
                if device not in states:
                    states[device] = {"first": start, "lo": start, "hi": end, "busy": 0, "count": 1, "gaps": []}
                    continue
                state = states[device]
                state["count"] += 1
                if start <= state["hi"]:
                    state["hi"] = max(state["hi"], end)
                else:
                    state["busy"] += state["hi"] - state["lo"]
                    gap = (start - state["hi"], state["hi"], start)
                    heapq.heappush(state["gaps"], gap)
                    if len(state["gaps"]) > 20:
                        heapq.heappop(state["gaps"])
                    state["lo"], state["hi"] = start, end
        devices = []
        for device, state in states.items():
            busy = state["busy"] + state["hi"] - state["lo"]
            span = state["hi"] - state["first"]
            devices.append({
                "device_id": device, "first_event_ns": state["first"], "last_event_ns": state["hi"],
                "event_window_ns": span, "gpu_activity_union_ns": busy,
                "internal_idle_ns": span - busy, "activity_events": state["count"],
                "observed_activity_fraction": busy / span if span else None,
                "largest_internal_gaps": [{"duration_ns": d, "start_ns": a, "end_ns": b, "cause": "unattributed"}
                                          for d, a, b in sorted(state["gaps"], reverse=True)],
            })
        top = []
        kernel = selected.get("kernel")
        if kernel:
            c = columns[kernel]
            name_col = next((x for x in ("shortName", "demangledName", "nameId", "name") if x in c), None)
            device = 'k."deviceId"' if "deviceId" in c else "NULL"
            name = "'unknown'" if not name_col else f'CAST(k.{quote(name_col)} AS TEXT)'
            join = ""
            if name_col and "StringIds" in tables and {"id", "value"} <= columns["StringIds"]:
                join = f' LEFT JOIN StringIds s ON k.{quote(name_col)}=s.id'
                name = f'COALESCE(s.value,{name})'
            query = (f'SELECT {device},{name},COUNT(*),SUM(k."end"-k.start) FROM {quote(kernel)} k{join}'
                     f' WHERE k."end">k.start GROUP BY 1,2 ORDER BY 4 DESC LIMIT 20')
            top = [{"device_id": d, "name": n, "calls": count, "aggregate_kernel_ns": total}
                   for d, n, count, total in conn.execute(query)]
        coverage = {}
        for label, token in {"nvtx": "NVTX", "python": "PYTHON", "cpu_sampling": "SAMPL", "osrt": "OSRT", "scheduler": "SCHED"}.items():
            coverage[label] = [t for t in sorted(tables) if token in t.upper()
                               and not t.upper().startswith(("ENUM_", "TARGET_INFO_", "META_"))
                               and not t.upper().startswith("NVTX_PAYLOAD_")
                               and conn.execute(f"SELECT 1 FROM {quote(t)} LIMIT 1").fetchone()]
        if not devices:
            warnings.append("No supported GPU activity found; collect CUDA activity before diagnosing GPU bottlenecks")
        return {
            "status": "ok" if devices else "no_gpu_activity", "input": str(Path(path).resolve()),
            "units": "nanoseconds", "activity_tables": selected,
            "devices": devices, "top_kernels_all_devices": top, "nonempty_coverage_tables": coverage,
            "native_cpu_samples": cpu_sample_summary(conn, tables, columns),
            "warnings": warnings,
            "scope": "All observed processes and streams, separate per device; recognized coverage tables indicate presence, not quality; composite/encoded events are not decoded",
            "denominator": "First to last observed GPU event per device; excludes leading/trailing capture idle",
            "limitations": ["Activity fraction is not SM or Tensor Core utilization", "Kernel duration sums are overlapping aggregate work, not step time", "Uncaptured GPU workloads can affect scheduling", "Gaps alone do not establish a CPU cause"],
        }


def prepare_input(path, out, timeout):
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Report not found: {source}")
    if source.suffix == ".sqlite":
        return source, None
    if source.suffix != ".nsys-rep":
        raise ValueError("Expected a .nsys-rep or an exported .sqlite report")
    target = out / "trace.sqlite"
    collector = saved([tool("nsys"), "export", "--type=sqlite", f"--output={target}", str(source)], out / "export.txt", timeout)
    if collector.get("exit_code") != 0 or not target.is_file():
        write_json(out / "export-status.json", collector)
        raise ValueError(f"Nsight export failed; inspect {out / 'export.txt.stderr.txt'}")
    return target, collector


def analyze(path, out, timeout):
    db, export = prepare_input(path, out, timeout)
    summary = native_summary(db)
    write_json(out / "trace-summary.json", summary)
    collectors = {}
    if export:
        collectors["export"] = export
    collectors["doctor"] = saved([tool("nsys-ai"), "doctor", str(db), "--format", "json"], out / "doctor.json", timeout)
    for name in PACK:
        collectors[name] = saved([tool("nsys-ai"), "skill", "run", name, str(db), "--format", "json", "--max-rows", "20", "--no-cache"], out / f"{name}.json", timeout)
    for report in ("cuda_gpu_kern_sum", "cuda_api_sum"):
        collectors[report] = saved([tool("nsys"), "stats", "--report", report, "--format", "csv", str(db)], out / f"{report}.txt", timeout)
    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(), "input": str(Path(path).expanduser().resolve()),
        "sqlite": str(db), "environment": environment(), "collectors": collectors,
        "status": "ok" if summary["status"] == "ok" and all(x.get("exit_code") == 0 for x in collectors.values()) else "partial",
        "native_summary_status": summary["status"],
        "upstream_scope": "Whole profile, all observed devices/processes unless the result itself says otherwise",
        "summary": "trace-summary.json", "excluded_upstream_rules": ["cpu_gpu_pipeline", "tensor_core_usage"],
    }
    write_json(out / "manifest.json", manifest)
    return db, manifest


def capture(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("Provide the workload command after --")
    prefix = Path(args.output).expanduser().resolve()
    for suffix in (".nsys-rep", ".sqlite", ".capture.json", ".capture.stdout.txt", ".capture.stdout.txt.stderr.txt"):
        if Path(str(prefix) + suffix).exists():
            raise ValueError(f"Refusing to overwrite capture artifact: {prefix}{suffix}")
    env = environment()
    cpu = args.cpu == "on" or (args.cpu == "auto" and env["native_cpu_sampling_available"])
    flags = [tool("nsys"), "profile", "--trace=cuda,nvtx,osrt", "--kill=none", f"--duration={args.duration}", f"--output={prefix}"]
    flags += ["--sample=process-tree", "--cpuctxsw=process-tree"] if cpu else ["--sample=none", "--cpuctxsw=none"]
    if args.capture_range == "cudaProfilerApi":
        flags += ["--capture-range=cudaProfilerApi", "--capture-range-end=stop-shutdown"]
    else:
        flags.append(f"--delay={args.delay}")
    if args.python_sampling:
        flags.append("--python-sampling=true")
    full = flags + command
    record = {"command": full, "cwd": str(Path.cwd()), "environment": env, "native_cpu_sampling_requested": cpu,
              "collection_duration_seconds": args.duration, "workload_continues_after_collection": True,
              "note": "Nsight may wait for the training process to exit; duration limits collection, not workload runtime"}
    if args.dry_run:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(str(prefix) + ".capture.json", record)
    record["execution"] = saved(full, str(prefix) + ".capture.stdout.txt", timeout=None)
    write_json(str(prefix) + ".capture.json", record)
    print(json.dumps({"manifest": str(prefix) + ".capture.json", "exit_code": record["execution"].get("exit_code")}))
    return record["execution"].get("exit_code") or (0 if record["execution"].get("exit_code") == 0 else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("doctor", help="Inspect tools, GPU and native CPU sampling permissions")
    cap = sub.add_parser("capture", help="Capture a short window, preserving the training process")
    cap.add_argument("--output", required=True, help="New output prefix, without extension")
    cap.add_argument("--delay", type=float, default=30)
    cap.add_argument("--duration", type=float, default=10)
    cap.add_argument("--cpu", choices=("auto", "on", "off"), default="auto")
    cap.add_argument("--capture-range", choices=("time", "cudaProfilerApi"), default="time")
    cap.add_argument("--python-sampling", action="store_true")
    cap.add_argument("--dry-run", action="store_true")
    cap.add_argument("command", nargs=argparse.REMAINDER)
    ana = sub.add_parser("analyze", help="Export, independently summarize and collect bounded upstream evidence")
    ana.add_argument("profile")
    comp = sub.add_parser("compare", help="Save before/after evidence and an upstream deterministic diff")
    comp.add_argument("before")
    comp.add_argument("after")
    for p in (ana, comp):
        p.add_argument("--output", required=True)
        p.add_argument("--timeout", type=int, default=180, help="Timeout for each analysis subprocess")
    args = parser.parse_args()
    if args.action == "doctor":
        print(json.dumps(environment(), indent=2, ensure_ascii=False))
        return 0
    if args.action == "capture":
        if args.duration <= 0 or args.delay < 0:
            parser.error("duration must be positive and delay nonnegative")
        return capture(args)
    out = new_output(args.output)
    if args.action == "analyze":
        _, manifest = analyze(args.profile, out, args.timeout)
        print(json.dumps({"manifest": str(out / "manifest.json"), "status": manifest["status"]}))
    else:
        before, before_manifest = analyze(args.before, new_output(out / "before"), args.timeout)
        after, after_manifest = analyze(args.after, new_output(out / "after"), args.timeout)
        collector = saved([tool("nsys-ai"), "diff", str(before), str(after), "--no-ai", "--format", "json"], out / "diff.json", args.timeout)
        status = "ok" if collector.get("exit_code") == 0 and before_manifest["status"] == after_manifest["status"] == "ok" else "partial"
        write_json(out / "manifest.json", {"status": status, "before": "before/manifest.json", "after": "after/manifest.json", "diff": collector,
                                           "interpretation": "Confirm matched workload, aligned steps and profiler-free timings before claiming speedup"})
        print(json.dumps({"manifest": str(out / "manifest.json"), "diff_exit_code": collector.get("exit_code")}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
