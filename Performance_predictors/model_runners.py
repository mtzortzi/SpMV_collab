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
from matplotlib.cbook import boxplot_stats
import numpy as np
from globals import MODEL_PATH, DATA_PATH, models
import dataReader as db
from sklearn import preprocessing
import seaborn as sns
from sklearn.tree import plot_tree
from Tree.model import TreePredictor
from utils_func import get_implementations_list
import os
from joblib import dump, load
import re
import pandas as pd
from tqdm import tqdm

def run_mlp(activation_function,
            nb_hidden_layers,
            in_dimension,
            out_dimension,
            hidden_size,
            csv_path,
            system,
            implementation,
            cache : str):

    mlp_model = MLP_model.MlpPredictor(activation_function,
                                       nb_hidden_layers,
                                       in_dimension,
                                       out_dimension,
                                       hidden_size)
    
    if implementation != "None":  
        dataset = dataReader.SparseMatrixDataset(csv_file=csv_path, using_implementation_split=True)
    else :
        dataset = dataReader.SparseMatrixDataset(csv_file=csv_path, using_implementation_split=False)
    optimizer = torch.optim.SGD(mlp_model.parameters(), lr=MLP_globals.lr)

    test_split = 0.2
    validation_split = 0.25 # 5% of the 20% remaining
    shuffle_dataset = True
    random_seed = 42
    n_iteration = MLP_globals.nb_epochs

    # Creating data indices for training and validation splits
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    split_2 = int(np.floor(validation_split * len(test_indices)))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(test_indices)

    test_indices, validation_indices = test_indices[split_2:], test_indices[:split_2]

    # Creating dataset for train and validation
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(dataset,sampler=train_sampler)
    test_loader = DataLoader(dataset, sampler=test_sampler)
    validation_dataset : db.SparseMatrixDataset = db.SparseMatrixDataset(dataframe=dataset.dataframe.iloc[validation_indices], using_implementation_split=(implementation!="None"))

    

    if cache != "None" :
        if implementation == "None":
            csv_path_validation = DATA_PATH + "validation/all_format/all_format_{}_{}_than_cache.csv".format(system, cache)
            prediction_dataset = dataReader.SparseMatrixDataset(csv_file=csv_path_validation, using_implementation_split=False)
        else :
            csv_path_validation = DATA_PATH + "validation/all_format/all_format_{}_{}_{}_than_cache.csv".format(system, implementation, cache)
            prediction_dataset = dataReader.SparseMatrixDataset(csv_file=csv_path_validation, using_implementation_split=True)
    else:
        if implementation == "None":
            csv_path_validation = DATA_PATH + "validation/all_format/all_format_{}.csv".format(system)
            prediction_dataset = dataReader.SparseMatrixDataset(csv_file=csv_path_validation, using_implementation_split=False)
        else:
            csv_path_validation = DATA_PATH + "validation/all_format/all_format_{}_{}.csv".format(system, implementation)
            prediction_dataset = dataReader.SparseMatrixDataset(csv_file=csv_path_validation, using_implementation_split=True)


    #Fitting the neural network
    (tbl_train_counter, tbl_train_losses, tbl_test_losses) = MLP_model.fit([],
                                                                           [],
                                                                           [],
                                                                           mlp_model,
                                                                           train_loader,
                                                                           test_loader,
                                                                           prediction_dataset,
                                                                           optimizer,
                                                                           MLP_globals.loss_fn,
                                                                           system,
                                                                           implementation,
                                                                           cache)
    
    
    #Saving the last model
    if implementation == "None":
        if cache != "None":
            saved_model_path = MODEL_PATH + "{}/mlp/{}/mlp_{}epochs_{}_than_cache".format(system, n_iteration, n_iteration, cache)
        else:
            saved_model_path = MODEL_PATH + "{}/mlp/{}/mlp_{}epochs".format(system, n_iteration, n_iteration)
    else:
        if cache != "None":
            saved_model_path = MODEL_PATH + "{}/mlp/{}/{}/mlp_{}epochs_{}_{}_than_cache".format(system, n_iteration, implementation, n_iteration, implementation, cache)
        else: 
            saved_model_path = MODEL_PATH + "{}/mlp/{}/{}/mlp_{}epochs_{}".format(system, n_iteration, implementation, n_iteration, implementation)
    torch.save(mlp_model.state_dict(), saved_model_path)
   
    # Ploting prediction dispersion for 5% of the train set

    if implementation == "None":
        if not(os.path.exists(MODEL_PATH + "{}/mlp/{}".format(system, n_iteration))):
            os.makedirs(MODEL_PATH + "{}/mlp/{}".format(system, n_iteration))
        name = "mlp_{}epochs_validation".format(n_iteration)
        path = MODEL_PATH + "{}/mlp/{}".format(system, n_iteration)
    else:
        if not(os.path.exists(MODEL_PATH + "{}/mlp/{}/{}".format(system, n_iteration, implementation))):
            os.makedirs(MODEL_PATH + "{}/mlp/{}/{}".format(system, n_iteration, implementation))
        name = "mlp_{}epochs_validation".format(n_iteration)
        path = MODEL_PATH + "{}/mlp/{}/{}".format(system, n_iteration, implementation)
    

    plot_prediction_dispersion_mlp(mlp_model, validation_dataset, name, path, implementation, cache)
    
    #Ploting loss history
    idx_test_counter = (len(tbl_train_counter)-1)//n_iteration
    test_counter = [tbl_train_counter[idx_test_counter*i] for i in range(n_iteration + 1)]
    plt.plot(tbl_train_counter, tbl_train_losses, 'bo')
    plt.scatter(test_counter, tbl_test_losses, color ='red', zorder=2)
    plt.legend(['Train Loss', 'Test Loss'], loc = 'upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    if implementation == "None":
        saved_figure_path = MODEL_PATH + "/{}/mlp/{}/mlp_{}_{}epochs_loss_history.png".format(system, n_iteration, system, n_iteration)
    else:
        saved_figure_path = MODEL_PATH + "/{}/mlp/{}/{}/mlp_{}_{}_{}epochs_loss_history.png".format(system, n_iteration, implementation, system, implementation, n_iteration)
    plt.savefig(saved_figure_path)
    plt.clf()

    return mlp_model

def predict_mlp(model, input, scaler_gflops:preprocessing.MinMaxScaler, scaler_energy_efficiency:preprocessing.MinMaxScaler):
    # print(input, input.shape)
    Y_pred = model(input)
    gflops_predicted_unscaled = torch.tensor(scaler_gflops.inverse_transform(Y_pred[0].detach().view(1, -1)))
    energy_efficiency_predicted_unscaled = torch.tensor(scaler_energy_efficiency.inverse_transform(Y_pred[1].detach().view(1, -1)))
    prediction = torch.cat((gflops_predicted_unscaled, energy_efficiency_predicted_unscaled), 1)
    return prediction

def run_svr(kernel, C, epsilon, gamma, csv_path, system, out_feature, implementation, cache):
    
    if implementation == "None":  
        dataset = dataReader.SparseMatrixDataset(csv_file=csv_path, using_implementation_split=False)
    else :
        dataset = dataReader.SparseMatrixDataset(csv_file=csv_path, using_implementation_split=True)
    

    svr_model = SVR_model.SvrPredictor(kernel, C, epsilon, gamma)
    SVR_model.train_usualSVR(svr_model, dataset, out_feature)
    #Saving the last model
    saved_model_path = ""
    if implementation == "None":
        if cache != "None":
            if out_feature == 0:
                saved_model_path = MODEL_PATH + "{}/svr/svr_gflops_{}_than_cache".format(system, cache)
            elif out_feature == 1:
                saved_model_path = MODEL_PATH + "{}/svr/svr_energy_efficiency_{}_than_cache".format(system, cache)
        else:
            if out_feature == 0:
                saved_model_path = MODEL_PATH + "{}/svr/svr_gflops".format(system)
            elif out_feature == 1:
                saved_model_path = MODEL_PATH + "{}/svr/svr_energy_efficiency".format(system)
    else:
        if cache != "None":
            if out_feature == 0: 
                saved_model_path = MODEL_PATH + "{}/svr/{}/svr_gflops_{}_than_cache".format(system, implementation, cache)
            elif out_feature == 1:
                saved_model_path = MODEL_PATH + "{}/svr/{}/svr_energy_efficiency_{}_than_cache".format(system, implementation, cache)
        else:
            if out_feature == 0:
                saved_model_path = MODEL_PATH + "{}/svr/{}/svr_gflops".format(system, implementation)
            elif out_feature == 1:
                saved_model_path = MODEL_PATH + "{}/svr/{}/svr_energy_efficiency".format(system, implementation)
    name = ""
    path = ""
    if implementation == "None":
        if not(os.path.exists(MODEL_PATH + "{}/svr".format(system))):
            os.makedirs(MODEL_PATH + "{}/svr".format(system))
        
        if out_feature == 0:
            name = "svr_gflops_validation"
        elif out_feature == 1:
            name = "svr_energy_efficiency_validation"
        path = MODEL_PATH + "{}/svr".format(system)
    else:
        if not(os.path.exists(MODEL_PATH + "{}/svr/{}".format(system, implementation))):
            os.makedirs(MODEL_PATH + "{}/svr/{}".format(system, implementation))
        
        if out_feature == 0:
            name = "svr_gflops_validation"
        elif out_feature == 1:
            name = "svr_energy_efficiency_validation"
        path = MODEL_PATH + "{}/svr/{}".format(system, implementation)
    
    dump(svr_model.usualSVR, saved_model_path + ".joblib")

    # Ploting prediction dispersion for 5% of the train set
    dataset_indices = list(range(len(dataset)))
    split = int(np.floor(0.05 * len(dataset)))
    
    np.random.seed(42)
    np.random.shuffle(dataset_indices)
    _, validation_indices = dataset_indices[split:], dataset_indices[:split]
    validation_dataset : db.SparseMatrixDataset = db.SparseMatrixDataset(dataframe=dataset.dataframe.iloc[validation_indices], using_implementation_split=(implementation!="None"))
    name = ""
    path = ""
    if implementation == "None":
        if not(os.path.exists(MODEL_PATH + "{}/svr".format(system))):
            os.makedirs(MODEL_PATH + "{}/svr".format(system))
        
        if out_feature == 0:
            name = "svr_gflops_validation"
        elif out_feature == 1:
            name = "svr_energy_efficiency_validation"
        path = MODEL_PATH + "{}/svr".format(system)
    else:
        if not(os.path.exists(MODEL_PATH + "{}/svr/{}".format(system, implementation))):
            os.makedirs(MODEL_PATH + "{}/svr/{}".format(system, implementation))
        
        if out_feature == 0:
            name = "svr_gflops_validation"
        elif out_feature == 1:
            name = "svr_energy_efficiency_validation"
        path = MODEL_PATH + "{}/svr/{}".format(system, implementation)
    
    

    plot_prediction_dispersion_sklearn(svr_model, validation_dataset, name, path, out_feature, "svr", implementation, cache)
    return svr_model 

def run_tree(max_depth, csv_path, system, out_feature, implementation, cache):
    if implementation == "None":
        dataset = dataReader.SparseMatrixDataset(csv_file=csv_path, using_implementation_split=False)
    else:
        dataset = dataReader.SparseMatrixDataset(csv_file=csv_path, using_implementation_split=True)


    tree_model : torch.nn.Module = Tree_model.TreePredictor(max_depth)
    Tree_model.train_TreePredictor(tree_model, dataset)

    name = ""
    path = ""
    if implementation == "None":
        if not(os.path.exists(MODEL_PATH + "{}/tree".format(system))):
            os.makedirs(MODEL_PATH + "{}/tree".format(system))
        
        if out_feature == 0:
            name = "tree_gflops_validation"
        elif out_feature == 1:
            name = "tree_energy_efficiency_validation"
        path = MODEL_PATH + "{}/tree".format(system)
    else:
        if not(os.path.exists(MODEL_PATH + "{}/tree/{}".format(system, implementation))):
            os.makedirs(MODEL_PATH + "{}/tree/{}".format(system, implementation))
        
        if out_feature == 0:
            name = "tree_gflops_validation"
        elif out_feature == 1:
            name = "tree_energy_efficiency_validation"
        path = MODEL_PATH + "{}/tree/{}".format(system, implementation)

    saved_model_path = ""
    if implementation == "None":
        if cache != "None":
            if out_feature == 0:
                saved_model_path = MODEL_PATH + "{}/tree/tree_gflops_{}_than_cache".format(system, cache)
            elif out_feature == 1:
                saved_model_path = MODEL_PATH + "{}/tree/tree_energy_efficiency_{}_than_cache".format(system, cache)
        else:
            if out_feature == 0:
                saved_model_path = MODEL_PATH + "{}/tree/tree_gflops".format(system)
            elif out_feature == 1:
                saved_model_path = MODEL_PATH + "{}/tree/tree_energy_efficiency".format(system)
    else:
        if cache != "None":
            if out_feature == 0:
                saved_model_path = MODEL_PATH + "{}/tree/{}/tree_gflops_{}_than_cache".format(system, implementation, cache)
            elif out_feature == 1:
                saved_model_path = MODEL_PATH + "{}/tree/{}/tree_energy_efficiency_{}_than_cache".format(system, implementation, cache)
        else:
            if out_feature == 0:
                saved_model_path = MODEL_PATH + "{}/tree/{}/tree_gflops".format(system, implementation)
            elif out_feature == 1:
                saved_model_path = MODEL_PATH + "{}/tree/{}/tree_energy_efficiency".format(system, implementation)

    dump(tree_model.tree, saved_model_path + ".joblib")

    # Ploting prediction dispersion for 5% of the train set
    dataset_indices = list(range(len(dataset)))
    split = int(np.floor(0.05 * len(dataset)))
    
    np.random.seed(42)
    np.random.shuffle(dataset_indices)
    _, validation_indices = dataset_indices[split:], dataset_indices[:split]
    validation_dataset : db.SparseMatrixDataset = db.SparseMatrixDataset(dataframe=dataset.dataframe.iloc[validation_indices], using_implementation_split=(implementation!="None"))


    if not(os.path.exists(MODEL_PATH + "{}/tree".format(system))):
        os.makedirs(MODEL_PATH + "{}/tree".format(system))
    
    
    
    plot_prediction_dispersion_sklearn(tree_model, validation_dataset, name, path, out_feature, "tree", implementation, cache)

    return tree_model

def load_mlp_model(activation_fn, 
                 nb_hidden_layers,
                 in_dimension, 
                 out_dimension, 
                 hidden_size,
                 name,
                 system,
                 implementation):
    print("loading mlp model")
    model = MLP_model.MlpPredictor(activation_fn, 
                                   nb_hidden_layers,
                                   in_dimension,
                                   out_dimension,
                                   hidden_size)
    model_path = ""
    if implementation == "None":
        model_path = MODEL_PATH + "/{}/mlp/{}/{}".format(system, MLP_globals.nb_epochs, name)
        model.load_state_dict(torch.load(model_path))
        return model
    else:
        model_path = MODEL_PATH + "/{}/mlp/{}/{}/{}".format(system, MLP_globals.nb_epochs, implementation, name)
        model.load_state_dict(torch.load(model_path))
        return model

def load_svr_model(name, system, implementation) -> SVR_model.SvrPredictor:
    print("loading svr model")
    model_path = ""
    if implementation == "None":
        model_path = MODEL_PATH + "{}/svr/{}".format(system, name)
    else:
        model_path = MODEL_PATH + "{}/svr/{}/{}".format(system, implementation, name)
    
    model = load(model_path + ".joblib")
    svr_model = SVR_model.SvrPredictor(svr=model)
    return svr_model

def load_tree_model(name, system, implementation):
    print("loading tree model")
    model_path = ""
    if implementation == "None":
        model_path = MODEL_PATH + "/{}/tree/{}".format(system, name)
    else:
        model_path = MODEL_PATH + "/{}/tree/{}/{}".format(system, implementation, name)
    
    model = load(model_path + ".joblib")
    tree_model = Tree_model.TreePredictor(tree=model)
    return tree_model

def plot_prediction_dispersion_sklearn(model:torch.nn.Module,
                                   dataset:db.SparseMatrixDataset,
                                   name:str,
                                   path:str,
                                   out_feature:int,
                                   model_name:str,
                                   implementation,
                                   cache):
    print("Plotting prediction for {} model".format(model_name))
    assert model_name in models
    predictions = []
    expectations = []
    for idx in tqdm(range(len(dataset))):
        (X, Y) = dataset[idx]
        input = np.array([X.numpy()])
        y_pred = model(input)
        if out_feature == 0:
            y_pred_unscaled = torch.tensor(dataset.scaler_gflops.inverse_transform(y_pred.reshape(-1, 1)))
            expectation = torch.tensor(dataset.scaler_gflops.inverse_transform(Y[out_feature].view(-1, 1)))
        elif out_feature == 1:
            y_pred_unscaled = torch.tensor(dataset.scaler_energy_efficiency.inverse_transform(y_pred.reshape(-1, 1)))
            expectation = torch.tensor(dataset.scaler_energy_efficiency.inverse_transform(Y[out_feature].view(-1, 1)))
        
        predictions.append(y_pred_unscaled.numpy().tolist()[0][0])
        expectations.append(expectation.numpy().tolist()[0][0])
    
    plt.clf()
    if implementation != "None":
        sns.scatterplot(x=predictions, y=expectations)
    else:
        implementations = get_implementations_list( dataset)
        sns.scatterplot(x=predictions, y=expectations, hue=implementations)
    
    if out_feature == 0:
        identity = np.arange(min(expectations), max(expectations))
    elif out_feature == 1:
        identity = np.arange(min(expectations), max(expectations), 0.1)
    sns.regplot(x=predictions, y=expectations, scatter=False, fit_reg=True, color="Blue")
    
    
    plot = sns.lineplot(x=identity, y=identity, color="Red")
    plt.xlabel("Predictions")
    plt.ylabel("Expectations")

    plot_title = ""
    if out_feature == 0:
        plot_title = "gflops_scattering"
    elif out_feature == 1:
        plot_title = "energy_effiency_scattering"
    
    plt.title(plot_title)

    if implementation == "None":
        if cache != "None":
            plot.get_figure().savefig("{}/scattering_{}_{}_than_cache.png".format(path, name, cache))
        else:
            plot.get_figure().savefig("{}/scaterring_{}.png".format(path, name))
    else:
        if cache != "None":
            plot.get_figure().savefig("{}/scattering_{}_{}_{}_than_cache.png".format(path, name, implementation, cache))
        else:
            plot.get_figure().savefig("{}/scattering_{}_{}.png".format(path, name, implementation))
    
    
    plt.clf()

def plot_prediction_dispersion_mlp(model:torch.nn.Module, 
                                   dataset:dataReader.SparseMatrixDataset,
                                   name:str, 
                                   path:str,
                                   implementation : str,
                                   cache):
    
    
    predictions = []
    expectations = []
    for idx in range(len(dataset)):
        (X, Y) = dataset[idx]
        prediction = predict_mlp(model, X, dataset.scaler_gflops, dataset.scaler_energy_efficiency)
        gflops_unscaled = torch.tensor(dataset.scaler_gflops.inverse_transform(Y[0].view(1, -1)))
        energy_efficiency_unscaled = torch.tensor(dataset.scaler_energy_efficiency.inverse_transform(Y[1].view(1, -1)))
        expectation = torch.cat((gflops_unscaled, energy_efficiency_unscaled), 1) 
        predictions.append(prediction.numpy().flatten().tolist())
        expectations.append(expectation.numpy().flatten().tolist())

    
    gflops_predictions = [val[0] for val in predictions[:]]
    energy_efficiency_predictions = [val[1] for val in predictions[:]]

    gflops_expectations = [val[0] for val in expectations[:]]
    energy_efficiency_expectations = [val[1] for val in expectations[:]]

    identity_gflops = np.arange(min(gflops_expectations), max(gflops_expectations), 10)
    identity_energy_efficiency = np.arange(min(energy_efficiency_expectations), max(energy_efficiency_expectations), 0.01)

    # Ploting gflops scattering
    sns.regplot(x=gflops_predictions, y=gflops_expectations, scatter=False, fit_reg=True, color="blue")
    if implementation != "None":
        sns.scatterplot(x=gflops_predictions, y=gflops_expectations)
    else:
        implementations = get_implementations_list(dataset)
        sns.scatterplot(x=gflops_predictions, y=gflops_expectations, hue=implementations)
    plot = sns.lineplot(x=identity_gflops, y=identity_gflops, color="red")
    plt.xlabel("Predictions")
    plt.ylabel("Expectations")
    plt.title("gflops_scattering")
    
    if implementation == "None":
        if cache != "None":
            print(path, "/gflops_scattering_", name, "_", cache, "_than_cache.png")
            plot.get_figure().savefig("{}/gflops_scattering_{}_{}_than_cache.png".format(path, name, cache))
        else:
            plot.get_figure().savefig("{}/gflops_scattering_{}.png".format(path, name))
    else:
        if cache != "None":
            print(path, "/gflops_scattering_",name, "_", implementation, cache, "_than_cache.png")
            plot.get_figure().savefig("{}/gflops_scattering_{}_{}_{}_than_cache.png".format(path, name, implementation, cache))
        else:
            plot.get_figure().savefig("{}/gflops_scattering_{}_{}.png".format(path, name, implementation))
    plt.clf()

    # Ploting energy efficiency scattering
    sns.regplot(x=energy_efficiency_predictions, y=energy_efficiency_expectations, scatter=False, fit_reg=True, color="blue")
    if implementation != "None":
        sns.scatterplot(x=energy_efficiency_predictions, y=energy_efficiency_expectations)
    else:
        implementations = get_implementations_list(dataset)
        sns.scatterplot(x=energy_efficiency_predictions, y=energy_efficiency_expectations, hue=implementations)
    plot = sns.lineplot(x=identity_energy_efficiency, y=identity_energy_efficiency, color="red")
    plt.xlabel("Predictions")
    plt.ylabel("Expectations")
    plt.title("energy_efficieny_scattering")
    if implementation == "None":
        if cache != "None":
            print(path, "/energy_efficiency_scattering_",name, "_", cache, "_than_cache.png")
            plot.get_figure().savefig("{}/energy_efficiency_scattering_{}_{}_than_cache.png".format(path, name, cache))
        else:
            plot.get_figure().savefig("{}/energy_efficiency_scattering_{}.png".format(path, name))
    else:
        if cache != "None":
            print(path, "/energy_efficiency_scattering_",name, "_", implementation, cache, "_than_cache.png")
            plot.get_figure().savefig("{}/energy_efficiency_scattering_{}_{}_{}_than_cache.png".format(path, name, implementation, cache))
        else:
            plot.get_figure().savefig("{}/energy_efficiency_scattering_{}_{}.png".format(path, name, implementation))
    plt.clf()

def plot_performance(model_lst:list,
                     validation_dataset_lst:list[db.SparseMatrixDataset],
                     model_name_lst:list[str],
                     save_path:str,
                     graph_name:str,
                     show_fliers:bool,
                     using_violin_plot:bool):
    
    d : dict = {"model_name" : list(), "loss" : list()}
    loss_fnc = MeanAbsolutePercentageError()
    for i in range(len(model_lst)):
        for idx in range(len(validation_dataset_lst[i])):
            (X, Y) = validation_dataset_lst[i][idx]
            modelType = re.search("mlp", model_name_lst[i])
            if modelType != None:
                prediction = predict_mlp(model_lst[i], X, validation_dataset_lst[i].scaler_gflops, validation_dataset_lst[i].scaler_energy_efficiency)
                gflops_unscaled = torch.tensor(validation_dataset_lst[i].scaler_gflops.inverse_transform(Y[0].view(1, -1)))
                energy_efficiency_unscaled = torch.tensor(validation_dataset_lst[i].scaler_energy_efficiency.inverse_transform(Y[1].view(1, -1)))
                expectation = torch.cat((gflops_unscaled, energy_efficiency_unscaled), 1) 
            else:
                (X, Y) = validation_dataset_lst[i][idx]
                input = np.array([X.numpy()])
                y_pred = model_lst[i](input)
                prediction = torch.tensor(validation_dataset_lst[i].scaler_gflops.inverse_transform(y_pred.reshape(-1, 1)))
                expectation = torch.tensor(validation_dataset_lst[i].scaler_gflops.inverse_transform(Y[0].view(-1, 1)))
            
            
            loss = loss_fnc(prediction, expectation)
            d["model_name"].append(model_name_lst[i])
            d['loss'].append(loss.tolist())
    df = pd.DataFrame(data=d, columns=('model_name', 'loss'))
    plt.figure(figsize=(15,7))
    outliers_dict : dict = dict()
    if using_violin_plot:
        for i in range(len(model_name_lst)):
                sns.violinplot(data=df, x="loss", y="model_name")
                outliers_dict[model_name_lst[i]] = len([y for stat in boxplot_stats(df['loss']) for y in stat['fliers']])
        graph_name += "_violinPlot"
    else:
        for i in range(len(model_name_lst)):
            sns.boxplot(data=df, x="loss", y="model_name", showfliers=show_fliers)
            outliers_dict[model_name_lst[i]] = len([y for stat in boxplot_stats(df['loss']) for y in stat['fliers']])

    print("saving boxplot")
    # print(outliers_dict)
    if show_fliers:
        graph_name += "_with_fliers"
    plt.title(graph_name)
    plt.savefig(save_path + graph_name + ".png")
    

def average_loss_mlp(model:torch.nn.Module, validation_dataset:db.SparseMatrixDataset, out_feature:int):
    print("Average loss mlp")
    avg_loss_lst : list = []
    loss_fnc = MeanAbsolutePercentageError()
    for idx in range(len(validation_dataset)):
        (X, Y) = validation_dataset[idx]
        
        prediction = predict_mlp(model, X, validation_dataset.scaler_gflops, validation_dataset.scaler_energy_efficiency)
        gflops_unscaled = torch.tensor(validation_dataset.scaler_gflops.inverse_transform(Y[0].view(1, -1)))
        energy_efficiency_unscaled = torch.tensor(validation_dataset.scaler_energy_efficiency.inverse_transform(Y[1].view(1, -1)))
        expectation = torch.cat((gflops_unscaled, energy_efficiency_unscaled), 1) 
        loss = loss_fnc(prediction, expectation)
        avg_loss_lst.append(loss)
    return sum(avg_loss_lst)/len(avg_loss_lst)  

def average_loss_sklearn(model:torch.nn.Module, validation_dataset:db.SparseMatrixDataset, out_feature:int):
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