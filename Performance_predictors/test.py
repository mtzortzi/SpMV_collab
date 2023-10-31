import utils
import torch

tensor1 = torch.tensor([179.3824, 0.7218])
tensor2 = torch.tensor([165.4110, 0.6730])

print(utils.MAPELoss(tensor1, tensor2))