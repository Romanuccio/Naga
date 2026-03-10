import naga
import time
import numpy as np
import G3C_extension as cga
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

length = 0.5
iterations = [300, 600, 900]
link_counts = [10, 20, 30, 40]
dt = 0.01
times = defaultdict(float)
# set seed for repeatibility
np.random.seed(0)

initial_configurations = [naga.configuration_multilink_random_planar(count=link_count, length=length) for link_count in link_counts]
trials = 300

def run_trial(iteration_count, link_count, config, trial_idx):
    start = time.perf_counter()
    T = np.arange(dt, iteration_count * dt + dt, dt)
    dx = np.ones(iteration_count) * dt
    dy = np.cos(12 * T) * dt
    dz = -np.sin(20 * (T + np.pi / 3.0)) * dt
    naga.calculate_kinematics(config, dx, dy, dz, iteration_count)
    end = time.perf_counter()
    if trial_idx == 0:
        return None
    return (f'iterations:{iteration_count}, links:{link_count}', end - start)

def main():
    times = defaultdict(float)
    counts = defaultdict(int)

    with ProcessPoolExecutor() as executor:
        futures = []
        for iteration_count in iterations:
            for i, link_count in enumerate(link_counts):
                config = initial_configurations[i]
                for j in range(trials + 1):
                    futures.append(
                        executor.submit(run_trial, iteration_count, link_count, config, j)
                    )

        for f in as_completed(futures):
            result = f.result()
            if result is None:
                continue
            key, t = result
            times[key] += t
            counts[key] += 1

    # average
    times = {key: times[key] / counts[key] for key in times}

    with open("time_results.json", "w") as f:
        json.dump(times, f, indent=4)

if __name__ == "__main__":
    main()
