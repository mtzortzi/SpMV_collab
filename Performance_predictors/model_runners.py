import MLP.globals as MLP_globals
import MLP.model as MLP_model
import SVR.model as SVR_model
import Tree.model as Tree_model

import dataReader
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchmetrics.regression import MeanAbsolutePercentageError
import matplotlib.pyplot as plt
import numpy as np
from globals import MODEL_PATH, DATA_PATH, models
import dataReader as db
from sklearn import preprocessing
import seaborn as sns
from sklearn.tree import plot_tree
from Tree.model import TreePredictor
from utils_func import MAPELoss

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

    csv_path_validation = DATA_PATH + "validation/all_format/all_format_{}.csv".format(system)
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

    return mlp_model

def predict_mlp(model, input, scaler_gflops:preprocessing.MinMaxScaler, scaler_energy_efficiency:preprocessing.MinMaxScaler):
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

def run_svr(kernel, C, epsilon, gamma, csv_path, system, out_feature):
    dataset = dataReader.SparseMatrixDataset(csv_path)

    svr_model = SVR_model.SvrPredictor(kernel, C, epsilon, gamma)
    # SVR_model.train_SVR_Nystroem(svr_model, dataset)
    # SVR_model.train_LinearSVR(svr_model, dataset)
    SVR_model.train_usualSVR(svr_model, dataset, out_feature)
    #Saving the last model
    if out_feature == 0:  
        saved_model_path = MODEL_PATH + "{}/svr/svr_gflops".format(system)
    elif out_feature == 1:
        saved_model_path = MODEL_PATH + "{}/svr/svr_energy_efficiency".format(system)
    
    torch.save(svr_model.state_dict(), saved_model_path)
    return svr_model 

def run_tree(max_depth, csv_path, system, out_feature):
    dataset = dataReader.SparseMatrixDataset(csv_path)
    tree_model : torch.nn.Module = Tree_model.TreePredictor(max_depth)
    Tree_model.train_TreePredictor(tree_model, dataset)
    if out_feature == 0:  
        saved_model_path = MODEL_PATH + "{}/tree/tree_gflops".format(system)
    elif out_feature == 1:
        saved_model_path = MODEL_PATH + "{}/tree/tree_energy_efficiency".format(system)
    torch.save(tree_model.state_dict(), saved_model_path)
    return tree_model

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
    model_path = MODEL_PATH + "/{}/mlp/{}".format(system, name)
    model.load_state_dict(torch.load(model_path))
    return model

def load_svr_model(kernel, C, epsilon, gamma, name, system):
    print("loading svr model")
    model_path = MODEL_PATH + "/{}/svr/{}".format(system, name)
    model = SVR_model.SvrPredictor(kernel, C, epsilon, gamma)
    model.load_state_dict(torch.load(model_path))
    return model

def load_tree_model(max_depth, name, system):
    print("loading tree model")
    model_path = MODEL_PATH + "/{}/tree/{}".format(system, name)
    model = Tree_model.TreePredictor(max_depth)
    model.load_state_dict(torch.load(model_path))
    return model

def plot_prediction_dispersion_sklearn(model:torch.nn.Module,
                                   validation_dataset:db.SparseMatrixDataset,
                                   name:str,
                                   path:str,
                                   out_feature:int,
                                   model_name:str):
    
    assert model_name in models
    length_dataset = len(validation_dataset)
    predictions = []
    expectations = []
    for idx in range(length_dataset):
        (X, Y) = validation_dataset[idx]
        input = np.array([X.numpy()])
        y_pred = model(input)
        if out_feature == 0:
            y_pred_unscaled = torch.tensor(validation_dataset.scaler_gflops.inverse_transform(y_pred.reshape(-1, 1)))
            expectation = torch.tensor(validation_dataset.scaler_gflops.inverse_transform(Y[out_feature].view(-1, 1)))
        elif out_feature == 1:
            y_pred_unscaled = torch.tensor(validation_dataset.scaler_energy_efficiency.inverse_transform(y_pred.reshape(-1, 1)))
            expectation = torch.tensor(validation_dataset.scaler_energy_efficiency.inverse_transform(Y[out_feature].view(-1, 1)))
        
        predictions.append(y_pred_unscaled.numpy().tolist()[0][0])
        expectations.append(expectation.numpy().tolist()[0][0])
    
    plt.clf()
    implementations = validation_dataset.dataframe["implementation"]
    if out_feature == 0:
        identity = np.arange(min(expectations), max(expectations))
    elif out_feature == 1:
        identity = np.arange(min(expectations), max(expectations), 0.1)
    
    sns.regplot(x=predictions, y=expectations, scatter=False, fit_reg=True, color="Blue")
    sns.scatterplot(x=predictions, y=expectations, hue=implementations)
    
    plot = sns.lineplot(x=identity, y=identity, color="Red")
    plt.xlabel("Predictions")
    plt.ylabel("Expectations")

    plot_title = ""
    if out_feature == 0:
        plot_title = "gflops_scattering"
    elif out_feature == 1:
        plot_title = "energy_effiency_scattering"
    
    plt.title(plot_title)
    plot.get_figure().savefig("{}/{}/scaterring_{}.png".format(path, model_name, name))
    
    if model_name == "tree":
        plt.clf()
        features = validation_dataset.features
        features.append("implementation")
        plot_tree(model.tree, filled=True, feature_names=features)
        plt.savefig("{}/tree/{}.pdf".format(path, name))

def plot_prediction_dispersion_mlp(model:torch.nn.Module, 
                                   validation_dataset:db.SparseMatrixDataset,
                                   name:str, 
                                   path:str):
    
    length_dataset = len(validation_dataset)
    predictions = []
    expectations = []
    for idx in range(length_dataset):
        (X, Y) = validation_dataset[idx]
        prediction = predict_mlp(model, X, validation_dataset.scaler_gflops, validation_dataset.scaler_energy_efficiency)
        gflops_unscaled = torch.tensor(validation_dataset.scaler_gflops.inverse_transform(Y[0].view(1, -1)))
        energy_efficiency_unscaled = torch.tensor(validation_dataset.scaler_energy_efficiency.inverse_transform(Y[1].view(1, -1)))
        expectation = torch.cat((gflops_unscaled, energy_efficiency_unscaled), 1) 
        predictions.append(prediction.numpy().flatten().tolist())
        expectations.append(expectation.numpy().flatten().tolist())

    gflops_predictions = [val[0] for val in predictions[:]]
    energy_efficiency_predictions = [val[1] for val in predictions[:]]

    gflops_expectations = [val[0] for val in expectations[:]]
    energy_efficiency_expectations = [val[1] for val in expectations[:]]

    identity_gflops = np.arange(min(gflops_expectations), max(gflops_expectations), 10)
    identity_energy_efficiency = np.arange(min(energy_efficiency_expectations), max(energy_efficiency_expectations), 0.01)

    implementations = validation_dataset.dataframe["implementation"]
    
    sns.regplot(x=gflops_predictions, y=gflops_expectations, scatter=False, fit_reg=True, color="blue")
    sns.scatterplot(x=gflops_predictions, y=gflops_expectations, hue=implementations)
    plot = sns.lineplot(x=identity_gflops, y=identity_gflops, color="red")
    plt.xlabel("Predictions")
    plt.ylabel("Expectations")
    plt.title("gflops_scattering")
    plot.get_figure().savefig("{}/gflops_scattering_{}.png".format(path, name))

    plt.clf()

    sns.regplot(x=energy_efficiency_predictions, y=energy_efficiency_expectations, scatter=False, fit_reg=True, color="blue")
    sns.scatterplot(x=energy_efficiency_predictions, y=energy_efficiency_expectations, hue=implementations)
    plot = sns.lineplot(x=identity_energy_efficiency, y=identity_energy_efficiency, color="red")
    plt.xlabel("Predictions")
    plt.ylabel("Expectations")
    plt.title("energy_efficieny_scattering")
    plot.get_figure().savefig("{}/energy_efficiency_scattering_{}.png".format(path, name))
    plt.clf()

def average_loss_mlp(model:torch.nn.Module, validation_dataset:db.SparseMatrixDataset, out_feature:int):
    print("Average loss mlp")
    length_dataset = len(validation_dataset)
    avg_loss_lst : list = []
    loss_fnc = MeanAbsolutePercentageError()
    for idx in range(length_dataset):
        (X, Y) = validation_dataset[idx]
        prediction = predict_mlp(model, X, validation_dataset.scaler_gflops, validation_dataset.scaler_energy_efficiency)
        gflops_unscaled = torch.tensor(validation_dataset.scaler_gflops.inverse_transform(Y[0].view(1, -1)))
        energy_efficiency_unscaled = torch.tensor(validation_dataset.scaler_energy_efficiency.inverse_transform(Y[1].view(1, -1)))
        expectation = torch.cat((gflops_unscaled, energy_efficiency_unscaled), 1)
        loss = loss_fnc(prediction, expectation)
        avg_loss_lst.append(loss)
    return sum(avg_loss_lst)/len(avg_loss_lst)

def average_loss_sklearn(model:torch.nn.Module, validation_dataset:db.SparseMatrixDataset, out_feature:int):
    print("Average loss sklearn")
    length_dataset = len(validation_dataset)
    avg_loss_lst : list = []
    loss_fnc = MeanAbsolutePercentageError()
    for idx in range(length_dataset):
        (X, Y) = validation_dataset[idx]
        input = np.array([X.numpy()])
        y_pred = model(input)
        if out_feature == 0:
            y_pred_unscaled = torch.tensor(validation_dataset.scaler_gflops.inverse_transform(y_pred.reshape(-1, 1)))
            expectation = torch.tensor(validation_dataset.scaler_gflops.inverse_transform(Y[out_feature].view(-1, 1)))
        elif out_feature == 1:
            y_pred_unscaled = torch.tensor(validation_dataset.scaler_energy_efficiency.inverse_transform(y_pred.reshape(-1, 1)))
            expectation = torch.tensor(validation_dataset.scaler_energy_efficiency.inverse_transform(Y[out_feature].view(-1, 1)))
        loss = loss_fnc(y_pred_unscaled, expectation)
        avg_loss_lst.append(loss)
    return sum(avg_loss_lst)/len(avg_loss_lst)