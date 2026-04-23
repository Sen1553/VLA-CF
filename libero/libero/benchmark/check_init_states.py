import os
import numpy as np
import torch

from libero.libero import benchmark

benchmark_dict = benchmark.get_benchmark_dict()
benchmark_instance = benchmark_dict["libero_cf"]()
init_states = benchmark_instance.get_task_init_states(0)
print(init_states.shape)

print(init_states[0])