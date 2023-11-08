import model_runners as runners
import MLP.globals as MLP_globals
import SVR.globals as SVR_globals
import Tree.globals as Tree_globals
import dataReader
import globals as g
import argparse
import numpy as np
import torch
from Tree.model import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', metavar='MODEL', required=True, help='Model name to run')
    parser.add_argument('-s', '--system', metavar='SYSTEM', required=True, help='CPU/GPU name')
    parser.add_argument('-l', '--load', action='store_true', help='Load the model described from it\'s hyperparameters in it\'s corresponfing global.py file and the -m parameter described above')

    args = parser.parse_args()
    args_data = vars(args)

    model_used = ""
    system_used = ""
    load_model = False
    
    for arg, value in args_data.items():
        if (arg == "model"):
            model_used = value
        if (arg == "system"):
            system_used = value
        if (arg == "load" and value):
            load_model = True
    

    assert model_used in g.models
    assert system_used in g.hardware

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
            name = "mlp_{}epochs_load".format(MLP_globals.nb_epochs)
            path = g.MODEL_PATH + "{}/mlp".format(system_used)
            runners.plot_prediction_dispersion_mlp(model, validation_dataset, name, path)
            avg_loss = runners.average_loss_mlp(model, validation_dataset)
            print("Avg loss of model mlp :", avg_loss)
        
        elif model_used == "tree":
            csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
            validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation)

            model_name_gflops = "tree_gflops"
            path = g.MODEL_PATH + "{}/tree".format(system_used)
            model_gflops = runners.load_tree_model(Tree_globals.max_depth,
                                                   model_name_gflops,
                                                   system_used)
            name_gflops = "tree_gflops_load"
            runners.plot_prediction_dispersion_sklearn(model_gflops, validation_dataset, name_gflops, path, 0, model_used)

            model_name_energy_efficiency = "tree_energy_effiency"
            path = g.MODEL_PATH + "{}/tree".format(system_used)
            model_energy_efficiency = runners.load_tree_model(Tree_globals.max_depth,
                                                              model_name_energy_efficiency,
                                                              system_used)
            name_energy_efficiency = "tree_energy_efficiency_load"
            runners.plot_prediction_dispersion_sklearn(model_energy_efficiency, validation_dataset, name_gflops, path, 1, model_used)
            

    elif model_used == "mlp":
        csv_path = g.DATA_PATH + "/all_format/all_format_{}.csv".format(system_used)
        csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
        validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation)
        path = g.MODEL_PATH + "{}".format(system_used)

        # Running model
        mlp_model = runners.run_mlp(MLP_globals.activation_fn,
                        MLP_globals.nb_hidden_layers,
                        MLP_globals.in_dimension,
                        MLP_globals.out_dimension,
                        MLP_globals.hidden_size,
                        csv_path,
                        system_used)
        
        # Plotting predictions
        name = "mlp_{}epochs".format(MLP_globals.nb_epochs)
        runners.plot_prediction_dispersion_mlp(mlp_model, validation_dataset, name, path)

        # Computing average loss on validation dataset
        avg_loss = runners.average_loss_mlp(mlp_model, validation_dataset)
        print("Avg loss of model mlp :", avg_loss)
        
    elif model_used == "svr":
        print("running Support Vector Regression model")
        csv_path = g.DATA_PATH + "/all_format/all_format_{}.csv".format(system_used)
        csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
        validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation)
        path = g.MODEL_PATH + "{}".format(system_used)
        
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
        runners.plot_prediction_dispersion_sklearn(model_gflops, validation_dataset, name_gflops, path, 0, model_used)

        # Computing average loss on validation dataset
        avg_loss : torch.Tensor = runners.average_loss_sklearn(model_gflops, validation_dataset, 0)
        print("Avg loss of svr model on gflops :", avg_loss.tolist())

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
        runners.plot_prediction_dispersion_sklearn(model_energy_efficiency, validation_dataset, name_energy_efficiency, path, 1, model_used)

        # Computing average loss on validation dataset
        avg_loss : torch.Tensor = runners.average_loss_sklearn(model_energy_efficiency, validation_dataset, 1)
        print("Avg loss of svr model on energy_efficiency:", avg_loss.tolist())

    elif model_used == "tree":
        print("running decision trees")
        csv_path = g.DATA_PATH + "/all_format/all_format_{}.csv".format(system_used)
        csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
        validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation)
        path = g.MODEL_PATH + "{}".format(system_used)
        
        # Running model
        print("Running tree on gflops predictions")
        tree_gflops = runners.run_tree(Tree_globals.max_depth, csv_path, system_used, 0)
        name_gflops = "tree_gflops"

        # Plotting predictions
        runners.plot_prediction_dispersion_sklearn(tree_gflops, validation_dataset, name_gflops, path, 0, model_used)

        # Computing average loss on validation dataset
        avg_loss : torch.Tensor = runners.average_loss_sklearn(tree_gflops, validation_dataset, 0)
        print("Avg loss of tree model on gflops :", avg_loss.tolist())


        # Running model
        print("Running tree on energy efficiency predictions")
        tree_energy_efficiency = runners.run_tree(Tree_globals.max_depth, csv_path, system_used, 1)
        name_energy_efficiency = "tree_energy_efficiency"

        # Plotting predictions
        runners.plot_prediction_dispersion_sklearn(tree_energy_efficiency, validation_dataset, name_energy_efficiency, path, 1, model_used)

        # Computing average loss on validation dataset
        avg_loss : torch.Tensor = runners.average_loss_sklearn(tree_energy_efficiency, validation_dataset, 1)
        print("Avg loss of tree model on energy efficiency:", avg_loss.tolist())