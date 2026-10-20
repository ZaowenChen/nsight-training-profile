# Native CPU sampling

The helper supports native CPU sampling with `capture --cpu auto` (the default)
or an explicit `--cpu on`. This feature relies on Linux perf permissions. A
failed environment check means a capture prerequisite is missing; it does not
mean the CPU analysis feature was omitted.

Check `nsys status --environment` and `/proc/sys/kernel/perf_event_paranoid`.
For Nsight's process-tree mode, level 2 allows user-mode CPU IP/backtrace
sampling and process-tree context switches; kernel-mode sampling requires
broader permissions. Level 3 or higher prevents ordinary unprivileged sampling.
A level of 4 can cause perf_event_open to return EACCES for ordinary users.
Verify actual permissions and capture results on the target host.

When the user explicitly wants to enable this feature, explain the concrete
system requirement. An administrator can temporarily run:

```bash
sudo sysctl -w kernel.perf_event_paranoid=2
nsys status --environment
```

Record the previous value and restore it after the test. Do not silently write
permanent sysctl configuration. If sudo requires a password the agent cannot
supply, have the user run the administrator step or a reviewable test-and-restore
script in their terminal. A password is not an agent input.

On AMD CPUs, Intel LBR being unavailable is expected and does not by itself
rule out CPU sampling. For useful native stacks,
use supported frame pointers or DWARF unwind data; `--backtrace=dwarf` is an
option when appropriate. Do not infer Python source-line attribution from a
native CPython interpreter frame: combine it with Python sampling/cProfile,
NVTX markers and source evidence.

`trace-summary.json` includes `native_cpu_samples`. It probes
COMPOSITE_EVENTS/SAMPLING_CALLCHAINS, counts periodic events only when
`cpuCycles=1`, and excludes scheduling-only callchains from CPU hotspots.
Self hotspots use stackDepth=0; inclusive hotspots count a function at most
once per sample even under recursion. Report sample shares as sample shares,
not exact function milliseconds, per-core CPU utilization or critical-path time.
Missing tables/fields and missing callchains remain explicit statuses.

After changing permissions, validate both the environment and the resulting
report. Nsight can return exit 0 while disabling unsupported CPU collection.
Positive periodic sample counts plus usable callchains are stronger evidence
than CLI success. Symbol resolution and report finalization can take longer
than the short capture itself.

Sources:
- [NVIDIA CPU profiling requirements](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
- [NVIDIA sample query semantics](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
- [Linux perf access control](https://docs.kernel.org/admin-guide/perf-security.html)
