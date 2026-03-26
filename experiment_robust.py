import time
import csv
import gc
from statistics import mean, median, stdev

import numpy as np
from memory_profiler import memory_usage
import naga


LENGTH = 0.5
DT = 0.01

ITERATIONS_LIST = [300, 600, 900, 1200, 1500]
LINK_COUNTS = [10, 20, 30, 40, 50]

TARGET_SUCCESSES = 100
MAX_ATTEMPTS_FACTOR = 5
WARMUPS = 2
BASE_SEED = 0

OUTPUT_CSV = "benchmark_randomized.csv"


def make_random_inputs(iteration_count, rng):
    T = np.arange(DT, iteration_count * DT + DT, DT)

    # Mild randomization, still structured
    ax = rng.uniform(0.8, 1.2)
    ay = rng.uniform(0.8, 1.2)
    az = rng.uniform(0.8, 1.2)

    wy = rng.uniform(8.0, 16.0)
    wz = rng.uniform(16.0, 24.0)

    phiy = rng.uniform(0.0, 2 * np.pi)
    phiz = rng.uniform(0.0, 2 * np.pi)

    dx = np.ones(iteration_count, dtype=float) * (ax * DT)
    dy = ay * np.cos(wy * T + phiy) * DT
    dz = -az * np.sin(wz * T + phiz) * DT

    meta = {
        "ax": ax,
        "ay": ay,
        "az": az,
        "wy": wy,
        "wz": wz,
        "phiy": phiy,
        "phiz": phiz,
    }
    return dx, dy, dz, meta


def make_random_config(link_count, length):
    return naga.configuration_multilink_random_planar(count=link_count, length=length)


def run_once(iteration_count, link_count, rng):
    dx, dy, dz, meta = make_random_inputs(iteration_count, rng)
    config = make_random_config(link_count, LENGTH)
    naga.calculate_kinematics(config, dx, dy, dz, iteration_count)
    return meta


def measure_once(iteration_count, link_count, rng):
    gc.collect()

    meta_holder = {}

    def wrapped():
        meta = run_once(iteration_count, link_count, rng)
        meta_holder.update(meta)

    t0 = time.perf_counter()
    mem_trace = memory_usage((wrapped,), interval=0.001, retval=False, max_usage=False)
    t1 = time.perf_counter()

    runtime_sec = t1 - t0
    peak_mem_mib = max(mem_trace)
    min_mem_mib = min(mem_trace)
    mem_span_mib = peak_mem_mib - min_mem_mib

    return runtime_sec, peak_mem_mib, mem_span_mib, meta_holder


def benchmark():
    rows = []

    for iteration_count in ITERATIONS_LIST:
        for link_count in LINK_COUNTS:
            print(f"\nCase: iterations={iteration_count}, links={link_count}")

            # Warmup on random feasible/infeasible attempts; failures are okay
            warmup_rng = np.random.default_rng(BASE_SEED + 10_000 * iteration_count + link_count)
            for _ in range(WARMUPS):
                try:
                    _ = measure_once(iteration_count, link_count, warmup_rng)
                except ValueError:
                    pass

            successes = 0
            attempts = 0
            max_attempts = MAX_ATTEMPTS_FACTOR * TARGET_SUCCESSES

            runtimes_success = []
            mem_success = []

            while successes < TARGET_SUCCESSES and attempts < max_attempts:
                seed = BASE_SEED + 1_000_000 * iteration_count + 10_000 * link_count + attempts
                rng = np.random.default_rng(seed)

                attempts += 1

                try:
                    runtime_sec, peak_mem_mib, mem_span_mib, meta = measure_once(
                        iteration_count, link_count, rng
                    )

                    successes += 1
                    runtimes_success.append(runtime_sec)
                    mem_success.append(peak_mem_mib)

                    row = {
                        "iterations": iteration_count,
                        "links": link_count,
                        "attempt": attempts,
                        "success": 1,
                        "runtime_sec": runtime_sec,
                        "peak_mem_mib": peak_mem_mib,
                        "mem_span_mib": mem_span_mib,
                        "ax": meta["ax"],
                        "ay": meta["ay"],
                        "az": meta["az"],
                        "wy": meta["wy"],
                        "wz": meta["wz"],
                        "phiy": meta["phiy"],
                        "phiz": meta["phiz"],
                    }
                    rows.append(row)

                    print(
                        f"  success {successes:03d}/{TARGET_SUCCESSES} "
                        f"(attempt {attempts:03d}) "
                        f"time={runtime_sec:.6f}s "
                        f"peak={peak_mem_mib:.3f} MiB"
                    )

                except ValueError:
                    row = {
                        "iterations": iteration_count,
                        "links": link_count,
                        "attempt": attempts,
                        "success": 0,
                        "runtime_sec": np.nan,
                        "peak_mem_mib": np.nan,
                        "mem_span_mib": np.nan,
                        "ax": np.nan,
                        "ay": np.nan,
                        "az": np.nan,
                        "wy": np.nan,
                        "wz": np.nan,
                        "phiy": np.nan,
                        "phiz": np.nan,
                    }
                    rows.append(row)
                    print(f"  infeasible at attempt {attempts:03d}")

            success_rate = successes / attempts if attempts > 0 else 0.0
            print(
                f"Finished case iterations={iteration_count}, links={link_count}: "
                f"successes={successes}, attempts={attempts}, success_rate={success_rate:.3f}"
            )

            if runtimes_success:
                print(
                    f"  runtime mean={mean(runtimes_success):.6f}s, "
                    f"median={median(runtimes_success):.6f}s, "
                    f"std={stdev(runtimes_success) if len(runtimes_success) > 1 else 0.0:.6f}s"
                )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "iterations", "links", "attempt", "success",
                "runtime_sec", "peak_mem_mib", "mem_span_mib",
                "ax", "ay", "az", "wy", "wz", "phiy", "phiz",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    benchmark()