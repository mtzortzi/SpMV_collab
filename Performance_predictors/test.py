from utils_func import *
import dataReader
from torch.utils.data import DataLoader
import os



dataset = dataReader.SparseMatrixDataset("./depos/Performance_predictors/Dataset/data/all_format/all_format_AMD-EPYC-24_larger_than_cache.csv", False)
loader = DataLoader(dataset)
print(get_implementations_list(loader, dataset))