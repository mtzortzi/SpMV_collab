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
    parser.add_argument('-i', '--implementation', metavar='IMPLEMENTATION', required=True, help='Implementation of the matrix, None if you want to use all implementations')
    parser.add_argument('-c', '--cache-split', action='store_true', help='Tell if we want to use dataset seperated based on cache size')
    parser.add_argument('-l', '--load', action='store_true', help='Load the model described from it\'s hyperparameters in it\'s corresponfing global.py file and the -m parameter described above')


    args = parser.parse_args()
    args_data = vars(args)

    model_used = ""
    system_used = ""
    implementation = ""
    load_model = False
    cache_split = False
    
    for arg, value in args_data.items():
        if arg == "model":
            model_used = value
        elif arg == "system":
            system_used = value
        elif arg == "load" and value:
            load_model = True
        elif arg == "implementation":
            implementation = value
        elif arg == "cache_split" and value:
            cache_split = True
    

    assert model_used in g.models
    assert system_used in g.hardware
    if implementation != "None":
        if model_used == "AMD-EPYC-24":
            assert implementation in g.IMPLEMENTATIONS_AMD_EPYC_24
        elif model_used == "Tesla-A100":
            assert implementation in g.IMPLEMENTATIONS_TESLA_A100
    
    if cache_split:
        assert system_used == "AMD-EPYC-24"

    if load_model :
        if model_used == "mlp":
            if cache_split:
                if implementation != "None":
                    print("Loading mlp model with cache split and {} implementation".format(implementation))
                else:
                    print("Loading mlp model with cache split and without implementation")
            else:
                if implementation != "None":
                    print("Loading mlp model without cache splut and {} implementation".format(implementation))
                else:
                    print("Loading mlp model without cache split and without implementation")
                    csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
                    model_name = "{}_{}epochs".format(model_used, MLP_globals.nb_epochs)
                    model = runners.load_mlp_model(MLP_globals.activation_fn,
                                                MLP_globals.nb_hidden_layers,
                                                MLP_globals.in_dimension,
                                                MLP_globals.out_dimension,
                                                MLP_globals.hidden_size,
                                                model_name,
                                                system_used)
                    
                    validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation, False)
                    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
                    name = "mlp_{}epochs_load".format(MLP_globals.nb_epochs)
                    path = g.MODEL_PATH + "{}/mlp".format(system_used)
                    runners.plot_prediction_dispersion_mlp(model, validation_dataset, validation_loader, name, path, implementation, "None")

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
            runners.plot_prediction_dispersion_sklearn(model_gflops, validation_dataset, validation_loader, graph_name_gflops, path, 0, model_used, implementation, "None")
        
        elif model_used == "tree":
            csv_path_validation = g.DATA_PATH + "/validation/all_foramt/all_format_{}.csv".format(system_used)
            model_name_gflops = "tree_gflops"
            model_gflops = runners.load_tree_model(model_name_gflops, system_used)
            validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation)
            validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
            
    elif cache_split:
        if model_used == "mlp":
            if implementation != "None":
                # TODO : take into account implementation can be None
                print("running mlp with {} system with {} implementation with cache split".format(system_used, implementation))

                path = g.MODEL_PATH + "{}/mlp/{}/{}".format(system_used, MLP_globals.nb_epochs, implementation)
                # Running mlp for larger than cache
                csv_path_larger_than_cache = g.DATA_PATH + "all_format/all_format_{}_{}_larger_than_cache.csv".format(system_used, implementation)
                csv_path_validation_larger_than_cache = g.DATA_PATH + "validation/all_format/all_format_{}_{}_larger_than_cache.csv".format(system_used, implementation)
                validation_dataset_larger_than_cache = dataReader.SparseMatrixDataset(csv_path_validation_larger_than_cache, True)
                validation_loader_larger_than_cache = DataLoader(validation_dataset_larger_than_cache, batch_size=1, shuffle=True)

                mlp_model_larger_than_cache = runners.run_mlp(MLP_globals.activation_fn,
                                                            MLP_globals.nb_hidden_layers,
                                                            MLP_globals.in_dimension - 1,
                                                            MLP_globals.out_dimension,
                                                            MLP_globals.hidden_size,
                                                            csv_path_larger_than_cache,
                                                            system_used,
                                                            implementation,
                                                            "larger")
                name_larger_than_cache = "mlp_{}epochs_real_data".format(MLP_globals.nb_epochs)
                runners.plot_prediction_dispersion_mlp(mlp_model_larger_than_cache, 
                                                    validation_dataset_larger_than_cache, 
                                                    validation_loader_larger_than_cache,
                                                    name_larger_than_cache,
                                                    path,
                                                    implementation,
                                                    "larger")
                
                
                # Running mlp for smaller than cache
                csv_path_smaller_than_cache = g.DATA_PATH + "all_format/all_format_{}_{}_smaller_than_cache.csv".format(system_used, implementation)
                csv_path_valiation_smaller_than_cache = g.DATA_PATH + "validation/all_format/all_format_{}_{}_smaller_than_cache.csv".format(system_used, implementation)
                validation_dataset_smaller_than_cache = dataReader.SparseMatrixDataset(csv_path_valiation_smaller_than_cache, True)
                validation_loader_smaller_than_cache = DataLoader(validation_dataset_smaller_than_cache, batch_size=1, shuffle=True)

                mlp_model_smaller_than_cache = runners.run_mlp(MLP_globals.activation_fn,
                                                            MLP_globals.nb_hidden_layers,
                                                            MLP_globals.in_dimension - 1,
                                                            MLP_globals.out_dimension,
                                                            MLP_globals.hidden_size,
                                                            csv_path_smaller_than_cache,
                                                            system_used,
                                                            implementation,
                                                            "smaller")
                name_smaller_than_cache = "mlp_{}epochs_real_data".format(MLP_globals.nb_epochs)
                runners.plot_prediction_dispersion_mlp(mlp_model_smaller_than_cache, 
                                                    validation_dataset_smaller_than_cache, 
                                                    validation_loader_smaller_than_cache,
                                                    name_smaller_than_cache,
                                                    path,
                                                    implementation,
                                                    "smaller")
            elif implementation == "None":
                print("running mlp with {} system without implementation with cache split".format(system_used))
                path = g.MODEL_PATH + "{}/mlp/{}/".format(system_used, MLP_globals.nb_epochs)
                # Running mlp for larger than cache
                csv_path_larger_than_cache = g.DATA_PATH + "all_format/all_format_{}_larger_than_cache.csv".format(system_used)
                csv_path_validation_larger_than_cache = g.DATA_PATH + "validation/all_format/all_format_{}_larger_than_cache.csv".format(system_used)
                validation_dataset_larger_than_cache = dataReader.SparseMatrixDataset(csv_path_validation_larger_than_cache, False)
                validation_loader_larger_than_cache = DataLoader(validation_dataset_larger_than_cache, batch_size=1, shuffle=True)

                mlp_model_larger_than_cache = runners.run_mlp(MLP_globals.activation_fn,
                                                            MLP_globals.nb_hidden_layers,
                                                            MLP_globals.in_dimension,
                                                            MLP_globals.out_dimension,
                                                            MLP_globals.hidden_size,
                                                            csv_path_larger_than_cache,
                                                            system_used,
                                                            implementation,
                                                            "larger")
                name_larger_than_cache = "mlp_{}epochs_real_data".format(MLP_globals.nb_epochs)
                runners.plot_prediction_dispersion_mlp(mlp_model_larger_than_cache, 
                                                    validation_dataset_larger_than_cache, 
                                                    validation_loader_larger_than_cache,
                                                    name_larger_than_cache,
                                                    path,
                                                    implementation,
                                                    "larger")
                
                
                # Running mlp for smaller than cache
                csv_path_smaller_than_cache = g.DATA_PATH + "all_format/all_format_{}_smaller_than_cache.csv".format(system_used)
                csv_path_valiation_smaller_than_cache = g.DATA_PATH + "validation/all_format/all_format_{}_smaller_than_cache.csv".format(system_used)
                validation_dataset_smaller_than_cache = dataReader.SparseMatrixDataset(csv_path_valiation_smaller_than_cache, False)
                validation_loader_smaller_than_cache = DataLoader(validation_dataset_smaller_than_cache, batch_size=1, shuffle=True)

                mlp_model_smaller_than_cache = runners.run_mlp(MLP_globals.activation_fn,
                                                            MLP_globals.nb_hidden_layers,
                                                            MLP_globals.in_dimension,
                                                            MLP_globals.out_dimension,
                                                            MLP_globals.hidden_size,
                                                            csv_path_smaller_than_cache,
                                                            system_used,
                                                            implementation,
                                                            "smaller")
                name_smaller_than_cache = "mlp_{}epochs_real_data".format(MLP_globals.nb_epochs)
                runners.plot_prediction_dispersion_mlp(mlp_model_smaller_than_cache, 
                                                    validation_dataset_smaller_than_cache, 
                                                    validation_loader_smaller_than_cache,
                                                    name_smaller_than_cache,
                                                    path,
                                                    implementation,
                                                    "smaller")
        elif model_used == "svr":
            if implementation != "None":
                print("running svr with {} system with {} implementation with cache split".format(system_used, implementation))
                path = g.MODEL_PATH + "{}/svr/{}".format(system_used, implementation)

                # Running svr for larger than cache
                csv_path_larger_than_cache = g.DATA_PATH + "all_format/all_format_{}_{}_larger_than_cache.csv".format(system_used, implementation)
                csv_path_validation_larger_than_cache = g.DATA_PATH + "validation/all_format/all_format_{}_{}_larger_than_cache.csv".format(system_used, implementation)
                validation_dataset_larger_than_cache = dataReader.SparseMatrixDataset(csv_path_validation_larger_than_cache, True)
                validation_loader_larger_than_cache = DataLoader(validation_dataset_larger_than_cache, batch_size=1, shuffle=True)

                print("Larger than cache gflops")
                svr_model_larger_than_cache_gflops = runners.run_svr(SVR_globals.kernel,
                                                              SVR_globals.C,
                                                              SVR_globals.epsilon,
                                                              SVR_globals.gamma,
                                                              csv_path_larger_than_cache,
                                                              system_used,
                                                              0,
                                                              implementation,
                                                              "larger")
                name_larger_than_cache = "svr_real_data_larger_than_cache"
                runners.plot_prediction_dispersion_sklearn(svr_model_larger_than_cache_gflops,
                                                           validation_dataset_larger_than_cache,
                                                           validation_loader_larger_than_cache,
                                                           name_larger_than_cache,
                                                           path,
                                                           0,
                                                           model_used,
                                                           implementation,
                                                           "larger")
                
                print("Larger than cache energy efficiency")
                svr_model_larger_than_cache_energy = runners.run_svr(SVR_globals.kernel,
                                                              SVR_globals.C,
                                                              SVR_globals.epsilon,
                                                              SVR_globals.gamma,
                                                              csv_path_larger_than_cache,
                                                              system_used,
                                                              1,
                                                              implementation,
                                                              "larger")
                name_larger_than_cache = "svr_real_data_larger_than_cache"
                runners.plot_prediction_dispersion_sklearn(svr_model_larger_than_cache_energy,
                                                           validation_dataset_larger_than_cache,
                                                           validation_loader_larger_than_cache,
                                                           name_larger_than_cache,
                                                           path,
                                                           1,
                                                           model_used,
                                                           implementation,
                                                           "larger")
                # Running svr for smaller than cache
                csv_path_smaller_than_cache = g.DATA_PATH + "all_format/all_format_{}_{}_smaller_than_cache.csv".format(system_used, implementation)
                csv_path_validation_smaller_than_cache = g.DATA_PATH + "validation/all_format/all_format_{}_{}_smaller_than_cache.csv".format(system_used, implementation)
                validation_dataset_smaller_than_cache = dataReader.SparseMatrixDataset(csv_path_validation_smaller_than_cache, True)
                validation_loader_smaller_than_cache = DataLoader(validation_dataset_smaller_than_cache, batch_size=1, shuffle=True)

                print("Smaller than cache gflops")
                svr_model_smaller_than_cache_gflops = runners.run_svr(SVR_globals.kernel,
                                                              SVR_globals.C,
                                                              SVR_globals.epsilon,
                                                              SVR_globals.gamma,
                                                              csv_path_smaller_than_cache,
                                                              system_used,
                                                              0,
                                                              implementation,
                                                              "smaller")
                name_smaller_than_cache = "svr_real_data_smaller_than_cache"
                runners.plot_prediction_dispersion_sklearn(svr_model_smaller_than_cache_gflops,
                                                           validation_dataset_smaller_than_cache,
                                                           validation_loader_smaller_than_cache,
                                                           name_smaller_than_cache,
                                                           path,
                                                           0,
                                                           model_used,
                                                           implementation,
                                                           "smaller")
                
                print("Smaller than cache energy efficiency")
                svr_model_smaller_than_cache_energy = runners.run_svr(SVR_globals.kernel,
                                                              SVR_globals.C,
                                                              SVR_globals.epsilon,
                                                              SVR_globals.gamma,
                                                              csv_path_smaller_than_cache,
                                                              system_used,
                                                              1,
                                                              implementation,
                                                              "smaller")
                name_smaller_than_cache = "svr_real_data_smaller_than_cache"
                runners.plot_prediction_dispersion_sklearn(svr_model_smaller_than_cache_energy,
                                                           validation_dataset_smaller_than_cache,
                                                           validation_loader_smaller_than_cache,
                                                           name_smaller_than_cache,
                                                           path,
                                                           1,
                                                           model_used,
                                                           implementation,
                                                           "smaller")
            else:
                print("Running svr with {} system without implementation with cache split".format(system_used))
                path = g.MODEL_PATH + "{}/svr/".format(system_used)
                # Running mlp for larger than cache
                csv_path_larger_than_cache = g.DATA_PATH + "all_format/all_format_{}_larger_than_cache.csv".format(system_used)
                csv_path_validation_larger_than_cache = g.DATA_PATH + "validation/all_format/all_format_{}_larger_than_cache.csv".format(system_used)
                validation_dataset_larger_than_cache = dataReader.SparseMatrixDataset(csv_path_validation_larger_than_cache, False)
                validation_loader_larger_than_cache = DataLoader(validation_dataset_larger_than_cache, batch_size=1, shuffle=True)

                print("Larger than cache gflops")
                svr_model_larger_than_cache_gflops = runners.run_svr(SVR_globals.kernel,
                                                              SVR_globals.C,
                                                              SVR_globals.epsilon,
                                                              SVR_globals.gamma,
                                                              csv_path_larger_than_cache,
                                                              system_used,
                                                              0,
                                                              implementation,
                                                              "larger")
                
                print("plotting on real data")
                name_larger_than_cache = "svr_real_data_larger_than_cache"
                runners.plot_prediction_dispersion_sklearn(svr_model_larger_than_cache_gflops,
                                                           validation_dataset_larger_than_cache,
                                                           validation_loader_larger_than_cache,
                                                           name_larger_than_cache,
                                                           path,
                                                           0,
                                                           model_used,
                                                           implementation,
                                                           "larger")
                
                print("Larger than cache energy efficiency")
                svr_model_larger_than_cache_energy = runners.run_svr(SVR_globals.kernel,
                                                              SVR_globals.C,
                                                              SVR_globals.epsilon,
                                                              SVR_globals.gamma,
                                                              csv_path_larger_than_cache,
                                                              system_used,
                                                              1,
                                                              implementation,
                                                              "larger")
                name_larger_than_cache = "svr_real_data_larger_than_cache"
                runners.plot_prediction_dispersion_sklearn(svr_model_larger_than_cache_energy,
                                                           validation_dataset_larger_than_cache,
                                                           validation_loader_larger_than_cache,
                                                           name_larger_than_cache,
                                                           path,
                                                           1,
                                                           model_used,
                                                           implementation,
                                                           "larger")
                # Running svr for smaller than cache
                csv_path_smaller_than_cache = g.DATA_PATH + "all_format/all_format_{}_smaller_than_cache.csv".format(system_used)
                csv_path_validation_smaller_than_cache = g.DATA_PATH + "validation/all_format/all_format_{}_smaller_than_cache.csv".format(system_used)
                validation_dataset_smaller_than_cache = dataReader.SparseMatrixDataset(csv_path_validation_smaller_than_cache, False)
                validation_loader_smaller_than_cache = DataLoader(validation_dataset_smaller_than_cache, batch_size=1, shuffle=True)

                print("Smaller than cache gflops")
                svr_model_smaller_than_cache_gflops = runners.run_svr(SVR_globals.kernel,
                                                              SVR_globals.C,
                                                              SVR_globals.epsilon,
                                                              SVR_globals.gamma,
                                                              csv_path_smaller_than_cache,
                                                              system_used,
                                                              0,
                                                              implementation,
                                                              "smaller")
                name_smaller_than_cache = "svr_real_data_smaller_than_cache"
                runners.plot_prediction_dispersion_sklearn(svr_model_smaller_than_cache_gflops,
                                                           validation_dataset_smaller_than_cache,
                                                           validation_loader_smaller_than_cache,
                                                           name_smaller_than_cache,
                                                           path,
                                                           0,
                                                           model_used,
                                                           implementation,
                                                           "smaller")
                
                print("Smaller than cache energy efficiency")
                svr_model_smaller_than_cache_energy = runners.run_svr(SVR_globals.kernel,
                                                              SVR_globals.C,
                                                              SVR_globals.epsilon,
                                                              SVR_globals.gamma,
                                                              csv_path_smaller_than_cache,
                                                              system_used,
                                                              1,
                                                              implementation,
                                                              "smaller")
                name_smaller_than_cache = "svr_real_data_smaller_than_cache"
                runners.plot_prediction_dispersion_sklearn(svr_model_smaller_than_cache_energy,
                                                           validation_dataset_smaller_than_cache,
                                                           validation_loader_smaller_than_cache,
                                                           name_smaller_than_cache,
                                                           path,
                                                           1,
                                                           model_used,
                                                           implementation,
                                                           "smaller")
        elif model_used == "tree":
            if implementation != "None":
                print("running tree with {} system without implementation with cache split".format(system_used, implementation))
                path = g.MODEL_PATH + "{}/tree/{}".format(system_used, implementation)

                # Running tree for larger than cache
                csv_path_larger_than_cache = g.DATA_PATH + "all_format/all_format_{}_{}_larger_than_cache.csv".format(system_used, implementation)
                csv_path_validation_larger_than_cache = g.DATA_PATH + "validation/all_format/all_format_{}_{}_larger_than_cache.csv".format(system_used, implementation)
                validation_dataset_larger_than_cache = dataReader.SparseMatrixDataset(csv_path_validation_larger_than_cache, True)
                validation_loader_larger_than_cache = DataLoader(validation_dataset_larger_than_cache, batch_size=1, shuffle=True)

                print("Larger than cache gflops")
                tree_model_larger_than_cache_gflops = runners.run_tree(Tree_globals.max_depth,
                                                                       csv_path_larger_than_cache,
                                                                       system_used,
                                                                       0,
                                                                       implementation,
                                                                       "larger",)
                name_larger_than_cache = "tree_real_data_larger_than_cache"
                runners.plot_prediction_dispersion_sklearn(tree_model_larger_than_cache_gflops,
                                                           validation_dataset_larger_than_cache,
                                                           validation_loader_larger_than_cache,
                                                           name_larger_than_cache,
                                                           path,
                                                           0,
                                                           model_used,
                                                           implementation,
                                                           "larger")
                
                print("Larger than cache energy efficiency")
                tree_model_larger_than_cache_energy = runners.run_tree(Tree_globals.max_depth,
                                                                       csv_path_larger_than_cache,
                                                                       system_used,
                                                                       1,
                                                                       implementation,
                                                                       "larger")
                name_larger_than_cache = "tree_real_data_larger_than_cache"
                runners.plot_prediction_dispersion_sklearn(tree_model_larger_than_cache_energy,
                                                           validation_dataset_larger_than_cache,
                                                           validation_loader_larger_than_cache,
                                                           name_larger_than_cache,
                                                           path,
                                                           1,
                                                           model_used,
                                                           implementation,
                                                           "larger")
                # Running tree for smaller than cache
                csv_path_smaller_than_cache = g.DATA_PATH + "all_format/all_format_{}_{}_smaller_than_cache.csv".format(system_used, implementation)
                csv_path_validation_smaller_than_cache = g.DATA_PATH + "validation/all_format/all_format_{}_{}_smaller_than_cache.csv".format(system_used, implementation)
                validation_dataset_smaller_than_cache = dataReader.SparseMatrixDataset(csv_path_validation_smaller_than_cache, True)
                validation_loader_smaller_than_cache = DataLoader(validation_dataset_smaller_than_cache, batch_size=1, shuffle=True)

                print("Smaller than cache gflops")
                tree_model_smaller_than_cache_gflops = runners.run_tree(Tree_globals.max_depth,
                                                                        csv_path_larger_than_cache,
                                                                        system_used,
                                                                        0,
                                                                        implementation,
                                                                        "smaller")
                name_smaller_than_cache = "tree_real_data_smaller_than_cache"
                runners.plot_prediction_dispersion_sklearn(tree_model_smaller_than_cache_gflops,
                                                           validation_dataset_smaller_than_cache,
                                                           validation_loader_smaller_than_cache,
                                                           name_smaller_than_cache,
                                                           path,
                                                           0,
                                                           model_used,
                                                           implementation,
                                                           "smaller")
                
                print("Smaller than cache energy efficiency")
                tree_model_smaller_than_cache_energy = runners.run_tree(Tree_globals.max_depth,
                                                                        csv_path_larger_than_cache,
                                                                        system_used,
                                                                        1,
                                                                        implementation,
                                                                        "smaller")
                name_smaller_than_cache = "tree_real_data_smaller_than_cache"
                runners.plot_prediction_dispersion_sklearn(tree_model_smaller_than_cache_energy,
                                                           validation_dataset_smaller_than_cache,
                                                           validation_loader_smaller_than_cache,
                                                           name_smaller_than_cache,
                                                           path,
                                                           1,
                                                           model_used,
                                                           implementation,
                                                           "smaller")
            else:
                print("Running tree with {} system without implementation with cache split".format(system_used))
                path = g.MODEL_PATH + "{}/tree/".format(system_used)
                # Running mlp for larger than cache
                csv_path_larger_than_cache = g.DATA_PATH + "all_format/all_format_{}_larger_than_cache.csv".format(system_used)
                csv_path_validation_larger_than_cache = g.DATA_PATH + "validation/all_format/all_format_{}_larger_than_cache.csv".format(system_used)
                validation_dataset_larger_than_cache = dataReader.SparseMatrixDataset(csv_path_validation_larger_than_cache, False)
                validation_loader_larger_than_cache = DataLoader(validation_dataset_larger_than_cache, batch_size=1, shuffle=True)

                print("Larger than cache gflops")
                tree_model_larger_than_cache_gflops = runners.run_tree(Tree_globals.max_depth,
                                                                       csv_path_larger_than_cache,
                                                                       system_used,
                                                                       0,
                                                                       implementation,
                                                                       "larger")
                
                print("plotting on real data")
                name_larger_than_cache = "tree_real_data_larger_than_cache"
                runners.plot_prediction_dispersion_sklearn(tree_model_larger_than_cache_gflops,
                                                           validation_dataset_larger_than_cache,
                                                           validation_loader_larger_than_cache,
                                                           name_larger_than_cache,
                                                           path,
                                                           0,
                                                           model_used,
                                                           implementation,
                                                           "larger")
                
                print("Larger than cache energy efficiency")
                tree_model_larger_than_cache_energy = runners.run_tree(Tree_globals.max_depth,
                                                                       csv_path_larger_than_cache,
                                                                       system_used,
                                                                       1,
                                                                       implementation,
                                                                       "larger")
                name_larger_than_cache = "tree_real_data_larger_than_cache"
                runners.plot_prediction_dispersion_sklearn(tree_model_larger_than_cache_energy,
                                                           validation_dataset_larger_than_cache,
                                                           validation_loader_larger_than_cache,
                                                           name_larger_than_cache,
                                                           path,
                                                           1,
                                                           model_used,
                                                           implementation,
                                                           "larger")
                # Running tree for smaller than cache
                csv_path_smaller_than_cache = g.DATA_PATH + "all_format/all_format_{}_smaller_than_cache.csv".format(system_used)
                csv_path_validation_smaller_than_cache = g.DATA_PATH + "validation/all_format/all_format_{}_smaller_than_cache.csv".format(system_used)
                validation_dataset_smaller_than_cache = dataReader.SparseMatrixDataset(csv_path_validation_smaller_than_cache, False)
                validation_loader_smaller_than_cache = DataLoader(validation_dataset_smaller_than_cache, batch_size=1, shuffle=True)

                print("Smaller than cache gflops")
                tree_model_smaller_than_cache_gflops = runners.run_tree(Tree_globals.max_depth,
                                                                        csv_path_larger_than_cache,
                                                                        system_used,
                                                                        0,
                                                                        implementation,
                                                                        "smaller")
                name_smaller_than_cache = "tree_real_data_smaller_than_cache"
                runners.plot_prediction_dispersion_sklearn(tree_model_smaller_than_cache_gflops,
                                                           validation_dataset_smaller_than_cache,
                                                           validation_loader_smaller_than_cache,
                                                           name_smaller_than_cache,
                                                           path,
                                                           0,
                                                           model_used,
                                                           implementation,
                                                           "smaller")
                
                print("Smaller than cache energy efficiency")
                tree_model_smaller_than_cache_energy = runners.run_tree(Tree_globals.max_depth,
                                                                        csv_path_larger_than_cache,
                                                                        system_used,
                                                                        1,
                                                                        implementation,
                                                                        "smaller")
                name_smaller_than_cache = "svr_real_data_smaller_than_cache"
                runners.plot_prediction_dispersion_sklearn(tree_model_smaller_than_cache_energy,
                                                           validation_dataset_smaller_than_cache,
                                                           validation_loader_smaller_than_cache,
                                                           name_smaller_than_cache,
                                                           path,
                                                           1,
                                                           model_used,
                                                           implementation,
                                                           "smaller")

    elif model_used == "mlp":
        if implementation == "None": 
            csv_path = g.DATA_PATH + "all_format/all_format_{}.csv".format(system_used)
            csv_path_validation = g.DATA_PATH + "validation/all_format/all_format_{}.csv".format(system_used)
            path = g.MODEL_PATH + "{}/mlp/{}".format(system_used, MLP_globals.nb_epochs)
            validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation, False)    
        else :
            print("running mlp with {} system with {} implementation".format(system_used, implementation))
            csv_path = g.DATA_PATH + "all_format/all_format_{}_{}.csv".format(system_used, implementation)
            csv_path_validation = g.DATA_PATH + "validation/all_format/all_format_{}_{}.csv".format(system_used, implementation)
            path = g.MODEL_PATH + "{}/mlp/{}/{}".format(system_used, MLP_globals.nb_epochs, implementation)
            validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation, True)

        
            
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)


        # Running model
        in_dimension_fix = 0
        if implementation != "None":
            in_dimension_fix = 1
        
        mlp_model = runners.run_mlp(MLP_globals.activation_fn,
                        MLP_globals.nb_hidden_layers,
                        MLP_globals.in_dimension - in_dimension_fix,
                        MLP_globals.out_dimension,
                        MLP_globals.hidden_size,
                        csv_path,
                        system_used,
                        implementation,
                        "None")
        
        # Plotting predictions
        name = "mlp_{}epochs_real_data".format(MLP_globals.nb_epochs)
        runners.plot_prediction_dispersion_mlp(mlp_model, validation_dataset, validation_loader, name, path, implementation)

        # Computing average loss on validation dataset
        avg_loss_gflops = runners.average_loss_mlp(mlp_model, validation_loader, validation_dataset, 0)
        print("Avg loss of model mlp on gflops : {}%".format(avg_loss_gflops.detach().tolist()*100))

        avg_loss_energy_efficiency = runners.average_loss_mlp(mlp_model, validation_loader, validation_dataset, 1)
        print("Avg loss of model mlp on energy efficiency : {}%".format(avg_loss_energy_efficiency.detach().tolist()*100))
        
    elif model_used == "svr":
        if implementation == "None":
            print("Running SVR with {} system without implementation without cache split".format(system_used))
            csv_path = g.DATA_PATH + "/all_format/all_format_{}.csv".format(system_used)
            csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
            validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation, False)
            path = g.MODEL_PATH + "{}/svr".format(system_used)
        else:
            print("Running SVR with {} system with {} implementation without cache split".format(system_used, implementation))
            csv_path = g.DATA_PATH + "/all_format/all_format_{}_{}.csv".format(system_used, implementation)
            csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}_{}.csv".format(system_used, implementation)
            validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation, True)
            path = g.MODEL_PATH + "{}/svr/{}".format(system_used, implementation)
            
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
        
        print("Running svr on gflops predictions")
        # Running model
        model_gflops = runners.run_svr(SVR_globals.kernel,
                        SVR_globals.C,
                        SVR_globals.epsilon,
                        SVR_globals.gamma,
                        csv_path,
                        system_used,
                        0,
                        implementation,
                        "None")
        # Plotting predictions
        name_gflops = "svr_gflops"
        runners.plot_prediction_dispersion_sklearn(model_gflops, validation_dataset, validation_loader, name_gflops, path, 0, model_used, implementation, "None")

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
                        1,
                        implementation,
                        "None")
        
        # Plotting predictions
        name_energy_efficiency = "svr_energy_efficiency"
        runners.plot_prediction_dispersion_sklearn(model_energy_efficiency, 
                                                   validation_dataset, 
                                                   validation_loader, 
                                                   name_energy_efficiency, 
                                                   path, 
                                                   1,
                                                   model_used, 
                                                   implementation,
                                                   "None")

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
        runners.plot_prediction_dispersion_sklearn(tree_gflops, validation_dataset, validation_loader, name_gflops, path, 0, model_used, "None")

        # Computing average loss on validation dataset
        avg_loss : torch.Tensor = runners.average_loss_sklearn(tree_gflops, validation_dataset, 0)
        print("Avg loss of tree model on gflops : {}%".format(avg_loss.tolist()*100))


        # Running model
        print("Running tree on energy efficiency predictions")
        tree_energy_efficiency = runners.run_tree(Tree_globals.max_depth, csv_path, system_used, 1)
        name_energy_efficiency = "tree_energy_efficiency"

        # Plotting predictions
        runners.plot_prediction_dispersion_sklearn(tree_energy_efficiency, validation_dataset, validation_loader, name_energy_efficiency, path, 1, model_used, "None")

        # Computing average loss on validation dataset
        avg_loss : torch.Tensor = runners.average_loss_sklearn(tree_energy_efficiency, validation_dataset, 1)
        print("Avg loss of tree model on energy efficiency : {}%".format(avg_loss.tolist()*100))