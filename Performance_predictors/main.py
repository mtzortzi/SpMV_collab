import model_runners as runners
import MLP.globals as MLP_globals
import SVR.globals as SVR_globals
import Tree.globals as Tree_globals
import dataReader
import globals as g
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from Tree.model import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', metavar='MODEL', required=True, help='Model name to run')
    parser.add_argument('-s', '--system', metavar='SYSTEM', required=True, help='CPU/GPU name')
    parser.add_argument('-i', '--implementation', metavar='IMPLEMENTATION', required=True, help='Implementation of the matrix')
    parser.add_argument('-l', '--load', action='store_true', help='Load the model described from it\'s hyperparameters in it\'s corresponfing global.py file and the -m parameter described above')


    args = parser.parse_args()
    args_data = vars(args)

    model_used = ""
    system_used = ""
    implementation = ""
    load_model = False
    
    for arg, value in args_data.items():
        if arg == "model":
            model_used = value
        if arg == "system":
            system_used = value
        if arg == "load" and value:
            load_model = True
        if arg == "implementation":
            implementation = value
    

    assert model_used in g.models
    assert system_used in g.hardware
    if implementation != "None":
        if model_used == "AMD-EPYC-24":
            assert implementation in g.IMPLEMENTATIONS_AMD_EPYC_24
        elif model_used == "Tesla-A100":
            assert implementation in g.IMPLEMENTATIONS_TESLA_A100

    if load_model :
        if model_used == "mlp":
            csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
            model_name = "{}_{}epochs".format(model_used, MLP_globals.nb_epochs)
            model = runners.load_mlp_model(MLP_globals.activation_fn,
                                           MLP_globals.nb_hidden_layers,
                                           MLP_globals.in_dimension,
                                           MLP_globals.out_dimension,
                                           MLP_globals.hidden_size,
                                           model_name,
                                           system_used)
            
            validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation)
            validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
            name = "mlp_{}epochs_load".format(MLP_globals.nb_epochs)
            path = g.MODEL_PATH + "{}/mlp".format(system_used)
            runners.plot_prediction_dispersion_mlp(model, validation_dataset, validation_loader, name, path)

            avg_loss_gflops = runners.average_loss_mlp(model, validation_loader, validation_dataset, 0)
            print("Avg loss of model mlp on gflops : {}%".format(avg_loss_gflops.detach().tolist()*100))

            avg_loss_energy_efficiency = runners.average_loss_mlp(model, validation_loader, validation_dataset, 1)
            print("Avg loss of model mlp on energy efficiency : {}%".format(avg_loss_energy_efficiency.detach().tolist()*100))
        
        elif model_used == "svr":
            csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
            models_name_gflops = "svr_gflops"
            model_gflops = runners.load_svr_model(models_name_gflops, system_used)
            validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation)
            validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

            graph_name_gflops = "svr_load"
            path = g.MODEL_PATH + "{}/svr".format(system_used)
            runners.plot_prediction_dispersion_mlp(model_gflops, validation_dataset, validation_loader, graph_name_gflops, path)
            

    elif model_used == "mlp":
        if implementation == "None":
            csv_path = g.DATA_PATH + "/all_format/all_format_{}.csv".format(system_used)
            csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
            path = g.MODEL_PATH + "{}/mlp/{}".format(system_used, MLP_globals.nb_epochs)
        else :
            csv_path = g.DATA_PATH + "/all_format/all_format_{}_{}.csv".format(system_used, implementation)
            csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}_{}.csv".format(system_used, implementation)
            path = g.MODEL_PATH + "{}/mlp/{}/{}".format(system_used, MLP_globals.nb_epochs, implementation)
        
        validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation)    
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)


        # Running model
        mlp_model = runners.run_mlp(MLP_globals.activation_fn,
                        MLP_globals.nb_hidden_layers,
                        MLP_globals.in_dimension,
                        MLP_globals.out_dimension,
                        MLP_globals.hidden_size,
                        csv_path,
                        system_used,
                        implementation)
        
        # Plotting predictions
        name = "mlp_{}epochs_real_data".format(MLP_globals.nb_epochs)
        runners.plot_prediction_dispersion_mlp(mlp_model, validation_dataset, validation_loader, name, path, implementation)

        # Computing average loss on validation dataset
        avg_loss_gflops = runners.average_loss_mlp(mlp_model, validation_loader, validation_dataset, 0)
        print("Avg loss of model mlp on gflops : {}%".format(avg_loss_gflops.detach().tolist()*100))

        avg_loss_energy_efficiency = runners.average_loss_mlp(mlp_model, validation_loader, validation_dataset, 1)
        print("Avg loss of model mlp on energy efficiency : {}%".format(avg_loss_energy_efficiency.detach().tolist()*100))
        
    elif model_used == "svr":
        print("running Support Vector Regression model")
        csv_path = g.DATA_PATH + "/all_format/all_format_{}.csv".format(system_used)
        csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
        validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
        path = g.MODEL_PATH + "{}/svr".format(system_used)
        
        print("Running svr on gflops predictions")
        # Running model
        model_gflops = runners.run_svr(SVR_globals.kernel,
                        SVR_globals.C,
                        SVR_globals.epsilon,
                        SVR_globals.gamma,
                        csv_path,
                        system_used,
                        0)
        # Plotting predictions
        name_gflops = "svr_gflops"
        runners.plot_prediction_dispersion_sklearn(model_gflops, validation_dataset, validation_loader, name_gflops, path, 0, model_used)

        # Computing average loss on validation dataset
        avg_loss : torch.Tensor = runners.average_loss_sklearn(model_gflops, validation_dataset, 0)
        print("Avg loss of svr model on gflops : {}%".format(avg_loss.tolist()*100))

        print("Running svr on energy efficiency predictions")
        # Running model
        model_energy_efficiency = runners.run_svr(SVR_globals.kernel,
                        SVR_globals.C,
                        SVR_globals.epsilon,
                        SVR_globals.gamma,
                        csv_path,
                        system_used,
                        1)
        
        # Plotting predictions
        name_energy_efficiency = "svr_energy_efficiency"
        runners.plot_prediction_dispersion_sklearn(model_energy_efficiency, validation_dataset, validation_loader, name_energy_efficiency, path, 1, model_used)

        # Computing average loss on validation dataset
        avg_loss : torch.Tensor = runners.average_loss_sklearn(model_energy_efficiency, validation_dataset, 1)
        print("Avg loss of svr model on energy_efficiency : {}%".format(avg_loss.tolist()*100))

    elif model_used == "tree":
        print("running decision trees")
        csv_path = g.DATA_PATH + "/all_format/all_format_{}.csv".format(system_used)
        csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
        validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
        path = g.MODEL_PATH + "{}/tree".format(system_used)
        
        # Running model
        print("Running tree on gflops predictions")
        tree_gflops = runners.run_tree(Tree_globals.max_depth, csv_path, system_used, 0)
        name_gflops = "tree_gflops"

        # Plotting predictions
        runners.plot_prediction_dispersion_sklearn(tree_gflops, validation_dataset, validation_loader, name_gflops, path, 0, model_used)

        # Computing average loss on validation dataset
        avg_loss : torch.Tensor = runners.average_loss_sklearn(tree_gflops, validation_dataset, 0)
        print("Avg loss of tree model on gflops : {}%".format(avg_loss.tolist()*100))


        # Running model
        print("Running tree on energy efficiency predictions")
        tree_energy_efficiency = runners.run_tree(Tree_globals.max_depth, csv_path, system_used, 1)
        name_energy_efficiency = "tree_energy_efficiency"

        # Plotting predictions
        runners.plot_prediction_dispersion_sklearn(tree_energy_efficiency, validation_dataset, validation_loader, name_energy_efficiency, path, 1, model_used)

        # Computing average loss on validation dataset
        avg_loss : torch.Tensor = runners.average_loss_sklearn(tree_energy_efficiency, validation_dataset, 1)
        print("Avg loss of tree model on energy efficiency : {}%".format(avg_loss.tolist()*100))