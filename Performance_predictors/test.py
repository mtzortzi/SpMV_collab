import utils_func
import torch

tensor1 = torch.tensor([179.3824, 0.7218])
tensor2 = torch.tensor([165.4110, 0.6730])

print(utils_func.MAPELoss(tensor1, tensor2))