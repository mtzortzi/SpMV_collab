import numpy as np
import torch

def generate_random_int(min : int, max : int) -> int:
    assert max > min
    range = max//10 * 10
    r = np.random.random()
    generated_number = (r*range + min)%range
    return int(generated_number)
