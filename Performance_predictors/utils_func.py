import numpy as np
import torch
from torch.utils.data import DataLoader
from dataReader import SparseMatrixDataset

def generate_random_int(min : int, max : int) -> int:
    assert max > min
    range = max//10 * 10
    r = np.random.random()
    generated_number = (r*range + min)%range
    return int(generated_number)

def MAPELoss(output, target):
    return torch.mean(torch.abs((target - output) / target))

def get_implementations_list(dataset:SparseMatrixDataset):
    implementation_list : list = list()
    inv_mappings = {v: k for k, v in dataset.mappings.items()}

    for idx in range(len(dataset)):
        implementation = dataset[idx][0][-1].tolist()
        implementation_list.append(inv_mappings[implementation])
    return implementation_list
    