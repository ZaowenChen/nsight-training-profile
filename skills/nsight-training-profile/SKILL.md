---
name: nsight-training-profile
description: 分析 NVIDIA Nsight Systems 训练报告（.nsys-rep/.sqlite），结合训练源码定位 CPU/GPU 等待、数据传输、同步、kernel 启动和通信瓶颈，设计并验证工程优化。也用于为 PyTorch 或强化学习训练采集短时 profile、比较优化前后结果；不用于游戏渲染分析或仅凭 GPU 利用率判断瓶颈。
---

# Training performance profiling

Use deterministic measurements to investigate training performance, then connect
the evidence to the user's code. Respond in the user's language. The installed
`nsys-ai` is an analysis dependency; the calling agent supplies the reasoning.
Its internal AI provider and MCP are optional, not prerequisites.

## Entry points

Resolve this skill directory from the location of this file. The bundled helper
uses Python 3.12+ and only the standard library:

```bash
python3 <skill-dir>/scripts/nsight_profile.py doctor
python3 <skill-dir>/scripts/nsight_profile.py analyze /absolute/run.nsys-rep --output /absolute/analysis
python3 <skill-dir>/scripts/nsight_profile.py compare /absolute/before.nsys-rep /absolute/after.nsys-rep --output /absolute/comparison
```

The output directory must be new or empty. Inputs remain unchanged. Analysis
exports a native report to SQLite when needed, calculates independent activity
intervals, and saves bounded `nsys-ai` results plus each collector's command and
status. Start with `manifest.json` and `trace-summary.json`; read other files as
needed. A failed optional collector is a coverage gap, not a measured zero.

If the user supplied a report, analyze it immediately. If no report is available,
inspect the actual launch command and environment, then follow
[references/capture.md](references/capture.md). Existing authorization to run a
training experiment applies; do not invent an additional approval step. Ask only
for missing information needed to identify the correct job or environment.

Native CPU capture is supported by `capture --cpu auto` or explicit `--cpu on`.
For permission requirements, temporary administrator setup and interpreting the
`native_cpu_samples` hotspots, read [references/native-cpu.md](references/native-cpu.md).

## Diagnosis

1. Establish scope: source revision, workload, device/rank/process coverage,
   training phase, effective batch, shapes, precision and compilation state.
   Inspect profile health, CPU sampling availability and initialization in the
   chosen window. Never substitute a guessed baseline.
2. Examine GPU work and gaps per device across all captured streams, memory
   transfers, host synchronization, frequent short kernels and communication
   overlap. Read [references/diagnosis.md](references/diagnosis.md) for evidence
   requirements and known upstream limitations.
3. Correlate important windows with runtime correlation IDs, NVTX, Python/native
   stacks, OS runtime or scheduler data, and actual source. Missing CPU samples
   limit CPU attribution; state a hypothesis and a targeted next capture instead
   of calling every GPU gap a DataLoader or Python problem.
4. Rank actionable opportunities by impact on the observed step's critical path.
   Give evidence, plausible alternatives, a concrete change and a verification
   experiment. Verify source paths/line numbers before citing them.
5. When optimization is requested, implement changes within the authorized
   project, keep an inspectable diff, and validate under equivalent conditions.
   Compare profiles to explain the mechanism and use profiler-free repeated
   measurements to establish the actual throughput improvement.

For RL, identify the stages present in that project: simulation/step, reset,
observation/reward, policy inference, rollout, advantage computation and policy
update. CPU-only and GPU simulation require different attribution. Read project
instructions before using project evaluation protocols. Do not assume an LLM
workload or change learning settings because a profiler suggests a larger batch.

## Measurement constraints

- Sum overlapping kernel durations only as aggregate work. The bundled summary
  unions kernel/copy/memset intervals per device; its denominator spans the first
  to last observed GPU event, not full training wall time or SM utilization.
  Leading/trailing idle time is excluded.
- CPU API time, enqueue delay, GPU kernel duration and NVTX host range duration
  are distinct. Long queue delay can mean a busy GPU.
- Separate observations, inferred causes and experiment results. Tensor Core
  utilization, bandwidth saturation and MFU require counters or a supported
  FLOP/hardware model. Kernel names alone are hints.
- Do not change system sampling permissions, drivers or GPU clock/power settings
  to make a report complete. Explain an actual limitation and use available
  evidence. A system permission change is a separate action.
- Capture short representative windows after warmup, retaining slow-step
  examples for intermittent issues. Avoid extra synchronization inside training
  loops. Do not collect CUDA with `torch.profiler` and Nsight concurrently in
  the same process.
- Validate numerical behavior and training quality when changing precision,
  effective batch, sequence length, environment count or learning parameters.
  For engineering-only changes, retain these conditions for comparisons.

## Deliverable

Explain the main limitation and a small ranked set of findings. Each finding
needs its device/process/time window, measured facts, source association (or
missing evidence), confidence with rationale, proposed change and runnable
verification. Link saved evidence/changes when useful. State whether speedup was
measured without profiling or is an untested expectation. Synthetic tool tests
do not establish training throughput or convergence.

Resolve tools through `NSIGHT_SYSTEMS_BIN` / `NSYS_AI_BIN`, then the user's
`~/.local/bin` and PATH. Use the workload's own Python environment. Keep profiler
dependencies isolated and do not install PyTorch into the profiler environment.
