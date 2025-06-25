import numpy as np
import naga
import time

total_time = 0
iterations = 50
for i in range(iterations + 1):
    start = time.perf_counter()

    count = 30
    length = 0.5
    initial_configuration = naga.configuration_multilink_random_planar(
        count=count, length=length
    )
    iterations = 600
    dt = 0.01
    T = np.arange(dt, iterations * dt + dt, dt)
    # constant velocity in x
    dx = np.ones(iterations) * dt
    # cos and shifted sin in y and z
    dy = np.cos(12 * T) * dt
    dz = -np.sin(20 * (T + np.pi / 3.0)) * dt
    configurations = naga.calculate_kinematics(
        initial_configuration, dx, dy, dz, iterations
    )

    end = time.perf_counter()
    # print(f"Elapsed time: {end - start} seconds")
    if i != 0:
        total_time += end - start

print(f"Total mean elapsed time over 50 iterations: {total_time/50} seconds")
