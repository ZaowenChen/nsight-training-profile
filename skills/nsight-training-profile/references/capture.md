# Capturing training evidence

Run `scripts/nsight_profile.py doctor` first. Resolve the training command from
the current task, repository and environment; use that environment's interpreter.
Check for an existing job before launching a second copy.

## Short time-based capture

```bash
python3 <skill-dir>/scripts/nsight_profile.py capture \
  --output /absolute/profiles/baseline --delay 30 --duration 10 \
  -- /absolute/training-env/bin/python train.py <actual arguments>
```

Use `--dry-run` to inspect the command. The helper includes CUDA/NVTX/OSRT,
`--kill=none`, and a manifest. `--cpu auto` enables native sampling only when the
Nsight environment check succeeds; otherwise native sampling/context-switch
collection is disabled and the manifest records the missing coverage.
`--python-sampling` optionally adds Python stack sampling on supported Nsight
versions. It has overhead; enable it for a targeted CPU question.

`--duration` limits collection, not the training workload. With `--kill=none`,
training continues and the Nsight CLI can keep waiting for it to exit. Use a
short workload/its existing step-limit option for a bounded experiment, or keep
the command in a terminal/session while normal training proceeds. Do not stop a
user's ongoing job because the report has finished collecting.
Report finalization and symbol resolution can take longer than collection and
may delay return from the profiler stop API. Allow post-processing time; a tight
external wall-clock timeout can interrupt a valid workload during finalization.

## Exact steady-state range

For repeatable training, instrument representative iterations after model, data
pipeline, CUDA initialization and compilation have warmed up:

```python
import torch

# Existing training setup and warmup happen before this block.
torch.cuda.synchronize()  # boundary only; not on every iteration
torch.cuda.cudart().cudaProfilerStart()
for _ in range(number_of_profiled_steps):
    with torch.cuda.nvtx.range("train_step"):
        train_step()     # use the project's real implementation
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
```

```bash
python3 <skill-dir>/scripts/nsight_profile.py capture \
  --output /absolute/profiles/baseline --capture-range cudaProfilerApi \
  --duration 15 -- /absolute/training-env/bin/python train.py <actual arguments>
```

The helper uses `--capture-range-end=stop-shutdown --kill=none`. Duration is a
collection failsafe, not a guarantee that a marker will ever be reached.
Multi-process/distributed capture needs a job-appropriate protocol; one rank's
markers do not prove all ranks used aligned windows.

Annotate the stages that actually exist. Put data-fetch markers around
`next(iterator)` when testing input stalls: a `for batch in loader` body's marker
misses that wait. Use NVTX for Nsight; `record_function` does not automatically
substitute for NVTX. Host ranges enclose scheduling work; associate kernels by
CUDA correlation rather than treating host range duration as GPU execution.

## Run context and comparison

Save revision/diff, workload arguments, GPU/driver, framework/CUDA versions,
effective batch, shapes, precision, compile/graph state, rank count, DataLoader
settings, sampling window and step count. Retain logs, throughput and numerical
or quality checks. Record other active GPU processes where they can affect the
result. Do not dump the whole environment or credentials.

Measure profiler-free step time/throughput on repeated equivalent windows.
Instrumentation and initialization can alter timings. Capturing a different
number of steps cannot establish speedup from total kernel time.

## Native sampling prerequisites

If native CPU sampling fails, inspect `kernel.perf_event_paranoid` and
`nsys status --environment`. CUDA/NVTX/OSRT evidence can still be useful.
Follow [native-cpu.md](native-cpu.md) for process-scoped sampling requirements,
temporary administrator setup, and interpreting missing data.

## References

- [Nsight Systems user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA analysis and expert rules](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
- [nsys-ai documentation](https://github.com/GindaChen/nsys-ai/tree/main/docs)
