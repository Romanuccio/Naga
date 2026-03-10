import naga
import time
import numpy as np
import G3C_extension as cga
import json
from collections import defaultdict

length = 0.5
iterations = [300, 600, 900]
link_counts = [10, 20, 30, 40]
dt = 0.01
times = defaultdict(float)
# set seed for repeatibility
np.random.seed(0)

initial_configurations = [naga.configuration_multilink_random_planar(count=link_count, length=length) for link_count in link_counts]
trials = 100

for iteration_count in iterations:
    for i, link_count in enumerate(link_counts):
        for j in range(trials+1):
            start = time.perf_counter()
            initial_configuration = initial_configurations[i]
            T = np.arange(dt, iteration_count*dt+dt, dt)
            # constant velocity in x
            dx = np.ones(iteration_count)*dt
            # cos and shifted sin in y and z
            dy = np.cos(12*T)*dt
            dz = -np.sin(20*(T+np.pi/3.))*dt
            naga.calculate_kinematics(initial_configuration, dx, dy, dz, iteration_count)
            
            end = time.perf_counter()
            
            if j == 0:
                continue
            
            current_iteration_time = end - start
            times[f'iterations:{iteration_count}, links:{link_count}'] += current_iteration_time

times = {key: value / trials for key, value in times.items()}

with open('time_results.json', 'w') as f:
    json.dump(times, f, indent=4)