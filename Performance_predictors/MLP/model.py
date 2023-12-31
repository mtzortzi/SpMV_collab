import torch
import torch.nn.functional as F
from globals import MODEL_PATH
from torch.utils.data import DataLoader, Dataset
import MLP.globals as globals
from tqdm import tqdm
import numpy as np
import model_runners as runners
import os

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

    def forward(self, x) -> torch.Tensor:
        return self.allLayers(x)
    

def reshapeFromShape(lst, shape):
    new_shape = 0
    for i in range(len(shape)):
        new_shape = new_shape+shape[i]
    return np.array(lst).reshape(new_shape)

def getShape(list):
    return [len(a) for a in list]

def train(model : MlpPredictor, epoch, train_loader : DataLoader, loss_fn, optimizer):
    costTbl = []
    count_tbl = []
    c = 0
    j = 0

    for batch in tqdm(train_loader, ncols=75):
        (x, y) = batch
        optimizer.zero_grad()
        y_pred = model(x)
        cost = loss_fn(y_pred, y)
        cost.backward()
        optimizer.step()
        j += 1
        if j%500 == 0:
            c += 1000 + (epoch-1)*len(train_loader)
            print("\r\t\t\t\t\t\t\t\t\t   Cost at iteration %i = %f" %(epoch, cost.detach().item()), end="")
            costTbl.append(cost.item())
            count_tbl.append(c)
        
    return costTbl, count_tbl

def test(model : MlpPredictor, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, ncols=75):
            (x, y) = batch
            y_pred = model(x)
            test_loss += loss_fn(y_pred, y)
    test_loss /= len(test_loader)
    print('Test set: Avg. loss: {:.4f}\n'.format(test_loss, end=""))
    return test_loss

def fit(tbl_test_losses : list, 
        tbl_train_losses : list, 
        tbl_train_counter : list, 
        model : MlpPredictor, 
        train_loader : DataLoader, 
        test_loader : DataLoader, 
        prediction_dataset : Dataset,
        optimizer, 
        loss_fn, 
        system : str,
        implementation : str,
        cache : str):
    
    test_losses = test(model, test_loader, loss_fn)
    tbl_test_losses.append(test_losses)
    for epoch in range(1, globals.nb_epochs + 1):
        #Training the model
        (train_losses, train_counter) = train(model, epoch, train_loader, loss_fn, optimizer)
        tbl_train_losses.append(train_losses)
        tbl_train_counter.append(train_counter)

        #Test
        test_losses = test(model, test_loader, loss_fn)
        tbl_test_losses.append(test_losses)

        #Validation
        if epoch%10 == 0:
            
            if implementation == "None":
                if not(os.path.exists(MODEL_PATH + "{}/mlp/{}".format(system, epoch))):
                    os.makedirs(MODEL_PATH + "{}/mlp/{}".format(system, epoch))
                path = MODEL_PATH + "{}/mlp/{}".format(system, epoch)
                saved_model_path = MODEL_PATH + "{}/mlp/{}/mlp_{}epochs".format(system, epoch, epoch)
            else:
                if not(os.path.exists(MODEL_PATH + "{}/mlp/{}/{}".format(system, epoch, implementation))):
                    os.makedirs(MODEL_PATH + "{}/mlp/{}/{}".format(system, epoch, implementation))
                path = MODEL_PATH + "{}/mlp/{}/{}".format(system, epoch, implementation)
                saved_model_path = MODEL_PATH + "{}/mlp/{}/{}/mlp_{}epochs_{}".format(system, epoch, implementation, epoch, implementation)
            
            
            name_prediction = "mlp_{}epochs_prediciton".format(epoch)
            runners.plot_prediction_dispersion_mlp(model, prediction_dataset, name_prediction, path, implementation, cache)
            torch.save(model.state_dict(), saved_model_path)
    
    s = getShape(tbl_train_counter)
    tbl_train_counter = reshapeFromShape(tbl_train_counter, s)
    s = getShape(tbl_train_losses)
    tbl_train_losses = reshapeFromShape(tbl_train_losses, s)

    return (tbl_train_counter, tbl_train_losses, tbl_test_losses)
