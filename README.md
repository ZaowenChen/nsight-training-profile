# nsight-training-profile

面向 coding agent 的训练性能分析 skill：用 NVIDIA Nsight Systems 和
`nsys-ai` 检查 CPU/GPU 等待、传输、同步与通信，再结合训练源码提出并验证工程优化。

An agent skill for evidence-based CPU/GPU training performance analysis, backed
by NVIDIA Nsight Systems and nsys-ai. Includes deterministic trace summaries,
native CPU hotspot extraction, before/after comparison, and a controlled
PyTorch demo.

## 能做什么

- 检查 Nsight、GPU 和原生 CPU 采样权限。
- 采集预热后的短时训练窗口，并记录启动命令和环境。
- 分析 `.nsys-rep` / `.sqlite`，输出有证据的 JSON 和查询日志。
- 按设备合并重叠的 GPU 活动区间，避免把并行 kernel 耗时误加成训练时间。
- 提取原生 CPU 的 self / inclusive 热点；区分周期采样和调度事件。
- 整理 kernel、数据传输、launch 开销、NVTX 与计算/通信重叠证据。
- 比较优化前后报告，并要求使用关闭 profiler 的测量验证真实吞吐变化。

适用于 PyTorch、常规深度学习和强化学习训练。Agent 负责阅读项目源码和形成诊断；
本项目的脚本负责可复核的数据提取。核心功能不需要额外的模型 API key 或 MCP 服务。

## 安装

需要 Linux、Python 3.12+。采集 CUDA 数据需要受对应 Nsight 版本支持的
NVIDIA GPU、驱动和训练环境。已有 SQLite 的独立汇总不需要 GPU；
转换 `.nsys-rep` 和原生统计需要安装 Nsight。

先获取源码：

```bash
git clone https://github.com/ZaowenChen/nsight-training-profile.git
cd nsight-training-profile
```

1. 从 [NVIDIA 官方页面](https://developer.nvidia.com/nsight-systems/get-started)
   安装适用于你的平台和驱动的 **Nsight Systems**。本仓库不包含 NVIDIA 安装包。
2. 单独安装分析依赖，保持训练环境独立：

   ```bash
   uv tool install --python 3.12 'nsys-ai==0.3.0'
   ```

   也可以使用独立虚拟环境：

   ```bash
   python3.12 -m venv .venv
   .venv/bin/python -m pip install -r requirements-tools.txt
   export NSYS_AI_BIN="$PWD/.venv/bin/nsys-ai"
   ```

3. 安装 skill：

   ```bash
   python3 scripts/install_skill.py
   ```

安装器默认写入 `${CODEX_HOME:-$HOME/.codex}/skills/nsight-training-profile`。
可以用 `--skills-dir /your/skills` 指定其他位置；如果目标已存在，安装器会保留它并退出，
请先自行备份或使用另一个目标目录。

可设置 `NSIGHT_SYSTEMS_BIN` 和 `NSYS_AI_BIN` 指向自定义安装的可执行文件。
默认也会检查 `~/.local/bin` 和 PATH。运行训练时始终使用原训练环境的 Python。

## 交给 agent

```text
使用 $nsight-training-profile 分析 /absolute/path/run.nsys-rep，
结合当前项目源码，定位 CPU/GPU 瓶颈，给出有证据的优化和验证方案。
```

也可以要求 agent 先采集预热后的少量训练步骤，再检查主要瓶颈。
[SKILL.md](skills/nsight-training-profile/SKILL.md) 定义了诊断方法、证据要求和报告约定。

## 直接使用脚本

以下命令从仓库根目录运行。输出目录必须是新目录或空目录。

```bash
# 工具、GPU 与 CPU 采样权限检查
python3 skills/nsight-training-profile/scripts/nsight_profile.py doctor

# 查看采集命令；train-env 和 train.py 替换为实际训练路径
python3 skills/nsight-training-profile/scripts/nsight_profile.py capture \
  --output profiles/baseline --delay 30 --duration 10 --dry-run \
  -- /absolute/train-env/bin/python train.py

# 去掉 --dry-run 后执行采集
# --duration 限制采集时长；--kill=none 会让训练继续，CLI 可能等待训练退出。

# 分析一份报告
python3 skills/nsight-training-profile/scripts/nsight_profile.py analyze \
  profiles/baseline.nsys-rep --output analysis/baseline

# 比较两份报告
python3 skills/nsight-training-profile/scripts/nsight_profile.py compare \
  profiles/baseline.nsys-rep profiles/optimized.nsys-rep --output analysis/comparison
```

先阅读 `manifest.json` 和 `trace-summary.json`，再按问题读取其他结果。
失败或不支持的收集器保留状态和 stderr；缺失数据不等于测得的零值。
`compare` 的 CLI 退出成功只表示流程完成，是否改善需要查看 `diff.json` 和实验条件。

## 原生 CPU 采样

`capture --cpu auto` 会按环境检查结果启用采样；`--cpu on` 显式请求开启。
Nsight 的 Linux 进程级用户态 CPU 采样一般需要
`kernel.perf_event_paranoid <= 2`，也可能受容器权限或内核能力影响。

如果确实需要调整，管理员可以记录原值、临时设为 2，测试后恢复。
脚本不自动修改系统设置。完整说明见
[原生 CPU 采样](skills/nsight-training-profile/references/native-cpu.md)。

汇总会读取 `COMPOSITE_EVENTS` / `SAMPLING_CALLCHAINS`：
只将 `cpuCycles=1` 的周期事件用于 CPU 热点；递归函数在同一条样本中只计一次。
样本占比不是精确的函数执行毫秒数，也不是整机 CPU 利用率。

## 可复现演示

`examples/cpu_gpu_demo.py` 构造一个有 CPU 预处理开销的小型训练任务。
两种模式计算完全相同的整数结果，保持模型、输入、精度和优化器一致，
仅将 CPU 预处理从 Python 循环换成 NumPy 向量化。

在已有 PyTorch CUDA + NumPy 的训练环境中运行：

```bash
mkdir -p results
python examples/cpu_gpu_demo.py --mode baseline --output results/baseline.json
python examples/cpu_gpu_demo.py --mode optimized --output results/optimized.json
```

它会在预热后测量多个窗口，窗口边界同步 GPU，输出每步中位耗时。
使用 `--capture` 可发出 CUDA profiler API 范围标记，配合
`--capture-range=cudaProfilerApi` 采集；使用 `--cpu-profile results/cpu.txt`
可单独检查 Python 函数热点。

此演示用于验证工具和诊断方法，不代表实际训练任务的瓶颈或预期加速。

## 测试

测试仅使用 Python 标准库，无需 GPU、Nsight 或网络：

```bash
python3 -m unittest discover -s tests -v
```

覆盖多 stream 重叠、多设备、拷贝区间、缺失采样、调度事件排除、递归去重、
只读输入、已有输出保护和 skill 安装行为。GitHub Actions 在 Python 3.12–3.14 上运行。

## 解释结果时的边界

- GPU 活动区间占比不同于 SM / Tensor Core 利用率。
- GPU 空闲只能定位等待，CPU 原因还需要调用栈、NVTX、线程状态或源码证据。
- 主机 NVTX 范围不同于 GPU 执行时间；并行 kernel 总耗时不同于训练 step 耗时。
- Profiler 会改变执行开销。最终吞吐与数值/训练质量验证应在等价条件下完成。
- 生成的报告可能含本地路径、命令、进程和模型信息。仓库忽略常见原始报告与日志，
  分享分析结果前仍需检查内容。

## 致谢与许可

- [nsys-ai](https://github.com/GindaChen/nsys-ai)：核心分析依赖，MIT。
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)：采集、导出与原生统计工具，
  按 NVIDIA 自身许可单独安装。
- [NVIDIA 分析文档](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)：事件结构与采样解释参考。

本项目代码和 skill 使用 [MIT License](LICENSE)。依赖项保留各自的许可；
本项目与 NVIDIA 及 nsys-ai 维护者没有官方隶属关系。
