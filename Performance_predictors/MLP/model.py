import torch
import torch.nn.functional as F
import MLP.globals as globals
from tqdm import tqdm

class MlpPredictor(torch.nn.Module):
    def __init__(self, activation_fn, 
                 nb_hidden_layers,
                 in_dimension, 
                 out_dimension, 
                 hidden_size
                 ) -> None:
        super(MlpPredictor, self).__init__()
        self.input_layer = torch.nn.Linear(in_dimension, hidden_size)

        f1 = lambda x : torch.nn.Linear(hidden_size, hidden_size)
        f2 = lambda x : activation_fn
        
        self.hiddenLayers = [f(_) for _ in range(nb_hidden_layers) for f in (f1, f2)]
        self.finalLayer = torch.nn.Linear(hidden_size, out_dimension)


        self.allLayers = torch.nn.Sequential(self.input_layer, *self.hiddenLayers, self.finalLayer)

    def forward(self, x):
        return self.allLayers(x)
    

def train(model : MlpPredictor, epoch, train_dataset, loss_fn, optimizer):
    costTbl = []
    count_tbl = []
    c = 0
    
    for j in tqdm(range(len(train_dataset)), ncols=75):
        #Todo : retrive data from dataset
        (x, y) = train_dataset[j]
        optimizer.zero_grad()
        y_pred = model(x)
        cost = loss_fn(y_pred, y)
        cost.backward()
        optimizer.step()

        if j%500 == 0:
            c += 1000 + (epoch-1)*len(train_dataset)
            print("\r\t\t\t\t\t\t\t\t\t   Cost at iteration %i = %f" %(epoch, cost.detach().item()), end="")
            costTbl.append(cost.item())
            count_tbl.append(c)
        
    return costTbl, count_tbl

def test(model : MlpPredictor, test_dataset, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x, y in test_dataset:
            y_pred = model(x)
            test_loss += loss_fn(y_pred, y)
    test_loss /= len(test_dataset)
    print('Test set: Avg. loss: {:.4f}\n'.format(test_loss, end=""))
    return test_loss

def fit(tbl_test_losses, tbl_train_losses, tbl_train_counter, model, train_set, test_set, optimizer, loss_fn):
    test_losses = test(model, test_set, loss_fn)
    tbl_test_losses.append(test_losses)
    for epoch in range(1, globals.nb_epochs + 1):
        #Training the model
        (train_losses, train_counter) = train(model, epoch, train_set, loss_fn, optimizer)
        tbl_train_losses.append(train_losses)
        tbl_train_counter.append(train_counter)

        #Test
        test_losses = test(model, test_set, loss_fn)
        tbl_test_losses.append(test_losses)

    return (tbl_train_counter, tbl_train_losses, tbl_test_losses)