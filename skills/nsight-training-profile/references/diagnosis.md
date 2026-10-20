# Evidence and interpretation

A report can support a precise observation without establishing its cause.
Treat missing tables, unsupported APIs, truncated results and failed queries
explicitly.

| Question | Evidence to combine | Candidate experiment |
| --- | --- | --- |
| Is the host starving the GPU? | Gaps across streams, outstanding work, launches, stacks and thread state during the gap | Isolate input preparation or simulation; remove an identified serial host stage |
| Does data loading limit training? | Actual batch-fetch range, worker/queue state, CPU stacks, storage events and GPU gaps | Change workers/prefetch or test a representative preloaded batch |
| Are transfers on the critical path? | Direction, bytes, duration, host blocking and useful overlap | Remove a proved round trip; test pinned/non-blocking transfer |
| Are synchronizations costly? | Blocking API, preceding device work, source stack and dependent host work | Reduce unnecessary scalar reads/log frequency while retaining dependencies |
| Is launch overhead material? | Frequent short kernels, launch cost, host gaps and dispatch rate | Test vectorization, supported compilation/fusion or graph capture |
| Is a kernel inefficient? | Critical-path importance and targeted hardware counters/roofline | Test an alternative kernel/layout/precision and validate outputs |
| Is communication exposed? | Aligned collectives, overlap and rank arrival times | Investigate a straggler or communication dependency |

## Known limitations

- `nsys-ai` 0.3.0's `cpu_gpu_pipeline` labels queue delays greater than 1 ms as
  `starvation_events`. That heuristic alone cannot identify CPU starvation:
  work can queue behind an already busy GPU. The default pack omits this rule.
  See [upstream source](https://github.com/GindaChen/nsys-ai/blob/main/src/nsys_ai/skills/builtins/cpu_gpu_pipeline.py).
- Tensor Core inference can depend on kernel classifications. Distinguish
  name-based hints from sampled hardware activity. Do not report the former as
  measured Tensor Core occupancy/utilization.
- Upstream “skills” are deterministic analysis units, different from this
  `SKILL.md`. Core analysis needs no model API key.
- NSYS schemas and upstream queries vary by version. The helper probes SQLite
  tables and records individual collector failures. Native `nsys stats` is a
  useful independent check. Missing or unknown data is not zero.
- The helper unions observed kernel/copy/memset intervals per device. Copies and
  kernels may overlap. Its denominator is the first-to-last GPU event window;
  it is not SM activity and does not prove coverage of unrelated processes or
  include leading/trailing capture idle time.
- The upstream pack is whole-profile unless explicitly filtered. Inspect a
  tool's parameter schema before using a device/rank/step filter; never relabel
  whole-profile output as scoped output.
- Long NCCL can reflect a late rank, and a long synchronization can reflect prior
  work. These may be symptoms. Warmup, instrumentation and other GPU processes
  can distort short captures; compare aligned steady-state iterations.

## Going deeper

Use `nsys-ai skill list --format json` and `nsys-ai skill run --help` to discover
installed parameters. Prefer bounded `--format json --max-rows` queries over
full trace dumps. The helper records installed versions for reproducibility.

Inspect `nsys stats --help-reports` and `nsys analyze --help` for native checks:

```bash
nsys stats --report cuda_gpu_kern_sum --format csv /absolute/profile.sqlite
nsys stats --report cuda_api_sum --format csv /absolute/profile.sqlite
```

For a dominant kernel, use a compatible Nsight Compute with a small filtered
capture. Replay/counter collection can change concurrency and runtime; confirm
the final result with ordinary training.

## Finding contract

Each finding contains the observation with units/device/process/time window,
evidence artifact and verified source location (or missing association), cause
as confirmed/hypothesis/unknown with rationale, proposed change with conditions,
and verification command/metric plus numerical or training-quality checks.
Avoid invented precision in confidence or expected speedup. Keep bulk tables in
the saved artifacts.
