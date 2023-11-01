import MLP.globals as MLP_globals
import MLP.model as MLP_model
import SVR.globals as SVR_globals
import SVR.model as SVR_model

import dataReader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from globals import MODEL_PATH, DATA_PATH
import dataReader as db
from sklearn import preprocessing
import utils_func
import seaborn as sns

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
    test_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset,sampler=train_sampler)
    test_loader = DataLoader(dataset, sampler=test_sampler)

    csv_path_validation = DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system)
    validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation)

    #Fitting the neural network
    (tbl_train_counter, tbl_train_losses, tbl_test_losses) = MLP_model.fit([],
                                                                           [],
                                                                           [],
                                                                           mlp_model,
                                                                           train_loader,
                                                                           test_loader,
                                                                           validation_dataset,
                                                                           optimizer,
                                                                           MLP_globals.loss_fn,
                                                                           system)
    
    #Saving the last model
    saved_model_path = MODEL_PATH + "{}/mlp_{}epochs".format(system, n_iteration)
    torch.save(mlp_model.state_dict(), saved_model_path)

    #Ploting prediction scaterring
    name = "mlp_{}epochs".format(MLP_globals.nb_epochs)
    path = MODEL_PATH + "{}".format(system)
    plot_prediction_dispersion(mlp_model, validation_dataset, name, path)

    #Ploting loss history
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

def predict(model, input, scaler_gflops:preprocessing.MinMaxScaler, scaler_energy_efficiency:preprocessing.MinMaxScaler):
    Y_pred = model(input.float())

    # loss = torch.nn.MSELoss()

    # print("Prediction (scaled) : {}".format(Y_pred))
    # print("Expected (scaled) : {}".format(Y))
    # print("Loss (scaled) : {}".format(loss(Y_pred, Y)))
    # print("Loss % (scaled) : {}".format(utils_func.MAPELoss(Y_pred, Y)))
          
    
    gflops_predicted_unscaled = torch.tensor(scaler_gflops.inverse_transform(Y_pred[0].detach().view(1, -1)))
    energy_efficiency_predicted_unscaled = torch.tensor(scaler_energy_efficiency.inverse_transform(Y_pred[1].detach().view(1, -1)))
    prediction = torch.cat((gflops_predicted_unscaled, energy_efficiency_predicted_unscaled), 1)


    # gflops_unscaled = torch.tensor(dataset.scaler_gflops.inverse_transform(Y[0].view(1, -1)))
    # energy_efficiency_unscaled = torch.tensor(dataset.scaler_energy_efficiency.inverse_transform(Y[0].view(1, -1)))
    # expected = torch.cat((gflops_unscaled, energy_efficiency_unscaled), 1)    

    # print("Prediction : {}".format(prediction))
    # print("Expected : {}".format(expected))
    # print("Loss : {}".format(loss(prediction, expected)))
    # print("Loss % : {}".format(utils_func.MAPELoss(prediction, expected)))
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


def plot_prediction_dispersion(model:torch.nn.Module, 
                               validation_dataset:db.SparseMatrixDataset,
                               name:str, 
                               path:str):
    
    
    length_dataset = len(validation_dataset)
    predictions = []
    expectations = []
    for idx in range(length_dataset):
        (X, Y) = validation_dataset[idx]
        prediction = predict(model, X, validation_dataset.scaler_gflops, validation_dataset.scaler_energy_efficiency)
        gflops_unscaled = torch.tensor(validation_dataset.scaler_gflops.inverse_transform(Y[0].view(1, -1)))
        energy_efficiency_unscaled = torch.tensor(validation_dataset.scaler_energy_efficiency.inverse_transform(Y[1].view(1, -1)))
        expectation = torch.cat((gflops_unscaled, energy_efficiency_unscaled), 1) 
        predictions.append(prediction.numpy().flatten().tolist())
        expectations.append(expectation.numpy().flatten().tolist())

    gflops_predictions = [val[0] for val in predictions[:]]
    energy_efficiency_predictions = [val[1] for val in predictions[:]]

    gflops_expectations = [val[0] for val in expectations[:]]
    energy_efficiency_expectations = [val[1] for val in expectations[:]]

    # print("gflops predictions :", gflops_predictions)
    # print("gflops_exepectations :", gflops_expectations)
    # print("energy efficiency predictions :", energy_efficiency_predictions)
    # print("energy efficiency expectations :", energy_efficiency_expectations)

    identity_gflops = np.arange(min(gflops_expectations), max(gflops_expectations), 10)
    identity_energy_efficiency = np.arange(min(energy_efficiency_expectations), max(energy_efficiency_expectations), 0.01)

    implementations = validation_dataset.dataframe["implementation"]
    
    sns.regplot(x=gflops_predictions, y=gflops_expectations, scatter=False, fit_reg=True, color="blue")
    sns.scatterplot(x=gflops_predictions, y=gflops_expectations, hue=implementations)
    plot = sns.lineplot(x=identity_gflops, y=identity_gflops, color="red")
    plot.get_figure().savefig("{}/gflops_scattering_{}.png".format(path, name))

    plt.clf()

    sns.regplot(x=energy_efficiency_predictions, y=energy_efficiency_expectations, scatter=False, fit_reg=True, color="blue")
    sns.scatterplot(x=energy_efficiency_predictions, y=energy_efficiency_expectations, hue=implementations)
    plot = sns.lineplot(x=identity_energy_efficiency, y=identity_energy_efficiency, color="red")
    plot.get_figure().savefig("{}/energy_efficiency_scattering_{}.png".format(path, name))
    plt.clf()