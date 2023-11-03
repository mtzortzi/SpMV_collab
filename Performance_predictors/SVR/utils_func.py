import numpy as np
import torch
from torch.utils.data import Dataset

def generate_random_int(min : int, max : int) -> int:
    assert max > min
    range = max//10 * 10
    r = np.random.random()
    generated_number = (r*range + min)%range
    return int(generated_number)

def MAPELoss(output, target):
    return torch.mean(torch.abs((target - output) / target))

