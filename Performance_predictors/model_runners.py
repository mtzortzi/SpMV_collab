import MLP.globals as MLP_globals
import MLP.model as MLP_model
import SVR.globals as SVR_globals
import SVR.model as SVR_model

import dataReader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from globals import MODEL_PATH
from globals import models
import dataReader as db

def run_mlp(activation_function,
            nb_hidden_layers,
            in_dimension,
            out_dimension,
            hidden_size,
            csv_path,
            system):
    print("running MLP")

    mlp_model = MLP_model.MlpPredictor(activation_function,
                                       nb_hidden_layers,
                                       in_dimension,
                                       out_dimension,
                                       hidden_size,)
    
    dataset = dataReader.SparseMatrixDataset(csv_path)
    optimizer = torch.optim.SGD(mlp_model.parameters(), lr=MLP_globals.lr)

    validation_split = 0.2
    shuffle_dataset = True
    random_seed = 42
    n_iteration = MLP_globals.nb_epochs

    # Creating data indices for training and validation splits
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating dataset for train and validation
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset,sampler=train_sampler)
    validation_loader = DataLoader(dataset, sampler=valid_sampler)

    (tbl_train_counter, tbl_train_losses, tbl_test_losses) = MLP_model.fit([],
                                                                           [],
                                                                           [],
                                                                           mlp_model,
                                                                           train_loader,
                                                                           validation_loader,
                                                                           optimizer,
                                                                           MLP_globals.loss_fn)
    
    saved_model_path = MODEL_PATH + "{}/mlp_{}epochs".format(system, n_iteration)
    torch.save(mlp_model.state_dict(), saved_model_path)
    idx_test_counter = (len(tbl_train_counter)-1)//n_iteration
    test_counter = [tbl_train_counter[idx_test_counter*i] for i in range(n_iteration + 1)]

    plt.plot(tbl_train_counter, tbl_train_losses, 'bo')
    plt.scatter(test_counter, tbl_test_losses, color ='red', zorder=2)
    plt.legend(['Train Loss', 'Test Loss'], loc = 'upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    saved_figure_path = MODEL_PATH + "/{}/mlp_{}_{}epochs.png".format(system, system, n_iteration)
    plt.savefig(saved_figure_path)

def predict(model, input, dataset : db.SparseMatrixDataset):
    #TODO: get random X and Y to do a prediction

    output = model(input)
    gflops_unscaled = dataset.scaler_gflops.inverse_transform(output[0])
    energy_efficiency_unscaled = dataset.scaler_energy_efficiency.inverse_transform(output[1])
    prediction = torch.cat((gflops_unscaled, energy_efficiency_unscaled), 1)
    return prediction

def run_svr(kernel, C, epsilon, gamma, csv_path):
    dataset = dataReader.SparseMatrixDataset(csv_path)
    svr_model = SVR_model.SvrPredictor(kernel, C, epsilon, gamma)
    SVR_model.train_SVR(svr_model, dataset)

def load_mlp_model(activation_fn, 
                 nb_hidden_layers,
                 in_dimension, 
                 out_dimension, 
                 hidden_size,
                 name,
                 system):
    print("loading mlp model")
    model = MLP_model.MlpPredictor(activation_fn, 
                                   nb_hidden_layers,
                                   in_dimension,
                                   out_dimension,
                                   hidden_size)
    model_path = MODEL_PATH + "/{}/{}".format(system, name)
    model.load_state_dict(torch.load(model_path))
    return model

def load_svr_model(kernel, C, epsilon, gamma, name, system):
    print("loading svr model")
    model_path = MODEL_PATH + "/{}/{}".format(system, name)
    model = SVR_model.SvrPredictor(kernel, C, epsilon, gamma)
    model.load_state_dict(torch.load(model_path))
    return model
