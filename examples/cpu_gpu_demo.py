#!/usr/bin/env python3
"""Controlled CPU/GPU training demo, not a benchmark of the user's real model.

The two modes compute identical integer preprocessing results. Only that CPU
implementation changes; model, input, precision and optimizer stay the same.
Requires the existing PyTorch + NumPy training environment.
"""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import statistics
import time

import numpy as np
import torch


def python_prepare(count):
    return sum(i * i for i in range(count))


def vector_prepare(count):
    values = np.arange(count, dtype=np.int64)
    return int(np.sum(values * values, dtype=np.int64))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('baseline', 'optimized'), required=True)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--count', type=int, default=120000)
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--cpu-profile', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.count <= 1000000 or args.steps <= 0 or args.repeats <= 0:
        parser.error('Require positive steps/repeats and count in [1, 1000000]')
    expected = (args.count - 1) * args.count * (2 * args.count - 1) // 6
    assert python_prepare(args.count) == vector_prepare(args.count) == expected
    prepare = python_prepare if args.mode == 'baseline' else vector_prepare
    if args.cpu_profile:
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(args.steps):
            assert prepare(args.count) == expected
        profiler.disable()
        with args.cpu_profile.open('w') as file:
            pstats.Stats(profiler, stream=file).sort_stats('cumulative').print_stats(20)
        args.output.write_text(json.dumps({'mode':args.mode, 'cpu_profile':str(args.cpu_profile),
            'checksum':expected, 'note':'cProfile instrumentation is excluded from throughput measurement'},indent=2)+'\n')
        return

    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.ReLU(), torch.nn.Linear(256, 16)).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    target = torch.zeros(64, 16, device='cuda')
    cpu_times = []

    def step():
        with torch.cuda.nvtx.range('train_step'):
            with torch.cuda.nvtx.range('cpu_prepare'):
                start = time.perf_counter()
                checksum = prepare(args.count)
                cpu_times.append(time.perf_counter() - start)
                cpu_batch = torch.full((64, 256), (checksum % 997) / 997.0)
            with torch.cuda.nvtx.range('h2d'):
                batch = cpu_batch.to('cuda')
            with torch.cuda.nvtx.range('forward'):
                loss = torch.nn.functional.mse_loss(model(batch), target)
            with torch.cuda.nvtx.range('backward'):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
            with torch.cuda.nvtx.range('optimizer'):
                optimizer.step()
        return loss

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    cpu_times.clear()
    if args.capture:
        torch.cuda.cudart().cudaProfilerStart()
    windows = []
    for _ in range(1 if args.capture else args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            loss = step()
        torch.cuda.synchronize()
        windows.append((time.perf_counter() - start) * 1000 / args.steps)
    if args.capture:
        torch.cuda.cudart().cudaProfilerStop()
    final_loss = loss.item()
    assert np.isfinite(final_loss)
    result = {
        'controlled_demo':True, 'mode':args.mode, 'profiled':args.capture,
        'steps_per_window':args.steps, 'repeats':len(windows),
        'step_ms_by_window':windows, 'median_step_ms':statistics.median(windows),
        'cpu_prepare_mean_ms':statistics.mean(cpu_times)*1000, 'checksum':expected,
        'final_loss':final_loss, 'torch':torch.__version__, 'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),
        'note':'Throughput claims use only profiled=false runs; this synthetic demo does not measure the user training project',
    }
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
