import torch

nb_epochs = 100
lr = 0.01
loss_fn = torch.nn.MSELoss()
activation_fn = torch.nn.Sigmoid()
final_activation_fn = torch.nn.Identity()
nb_hidden_layers = 2
in_dimension = 7 # CHANGE FROM 7 TO 6 IF YOU MAKE MODEL FOR EACH MATRIX IMPLEMENTATION
out_dimension = 2
hidden_size = 20