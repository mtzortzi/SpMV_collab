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
    parser.add_argument('-m', '--model', metavar='MODEL', help='Model name to run')
    parser.add_argument('-s', '--system', metavar='SYSTEM', help='CPU/GPU name')
    parser.add_argument('-i', '--implementation', metavar='IMPLEMENTATION', help='Implementation of the matrix, None if you want to use all implementations')
    parser.add_argument('-c', '--cache-split', action='store_true', help='Tell if we want to use dataset seperated based on cache size')
    parser.add_argument('-l', '--load', action='store_true', help='Load the model described from it\'s hyperparameters in it\'s corresponfing global.py file and the -m parameter described above')
    parser.add_argument('-p', '--performance', action='store_true', help='Plots performances of all models')

    args = parser.parse_args()
    args_data = vars(args)

    model_used = ""
    system_used = ""
    implementation = ""
    load_model = False
    cache_split = False
    performance = False
    
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
        elif arg == "performance" and value:
            performance = True

    
    

    if performance:
        assert model_used == None and implementation == None and load_model == False and system_used != None
        if system_used == "Tesla-A100":
            assert not(cache_split)
        
    else:
        assert model_used in g.models
        assert system_used in g.hardware
        if implementation != "None":
            if model_used == "AMD-EPYC-24":
                assert implementation in g.IMPLEMENTATIONS_AMD_EPYC_24
            elif model_used == "Tesla-A100":
                assert implementation in g.IMPLEMENTATIONS_TESLA_A100
        
        if cache_split:
            assert system_used == "AMD-EPYC-24"
    

    # plot_performance(model_lst:list,
    #                  validation_dataset_lst:list[db.SparseMatrixDataset],
    #                  model_name_lst:list[str]):
    if performance:
        if cache_split:
            print("Plotting performances for {} with cache split".format(system_used))
            model_name_lst : list = list()
            model_lst : list = list()
            validation_dataset_lst : list[dataReader.SparseMatrixDataset] = list()
            for model in g.models:
                model_name_lst.append("{}_{}_LC".format(model, system_used))
                model_name_lst.append("{}_{}_SC".format(model, system_used))
                csv_path_validation_larger = g.DATA_PATH + "/validation/all_format/all_format_{}_{}_than_cache.csv".format(system_used, "larger")
                csv_path_validation_smaller = g.DATA_PATH + "/validation/all_format/all_format_{}_{}_than_cache.csv".format(system_used, "smaller")
                validation_dataset_larger = dataReader.SparseMatrixDataset(csv_path_validation_larger, False)
                validation_dataset_smaller = dataReader.SparseMatrixDataset(csv_path_validation_smaller, False)
                if model == "mlp":
                    model_name_larger = "{}_{}epochs_larger_than_cache".format(model, MLP_globals.nb_epochs)
                    model_name_smaller = "{}_{}epochs_smaller_than_cache".format(model, MLP_globals.nb_epochs)
                    

                    tempModelLarger = runners.load_mlp_model(MLP_globals.activation_fn,
                                                        MLP_globals.nb_hidden_layers,
                                                        MLP_globals.in_dimension,
                                                        MLP_globals.out_dimension,
                                                        MLP_globals.hidden_size,
                                                        model_name_larger,
                                                        system_used,
                                                        "None")
                    tempModelSmaller = model_smaller = runners.load_mlp_model(MLP_globals.activation_fn,
                                                        MLP_globals.nb_hidden_layers,
                                                        MLP_globals.in_dimension,
                                                        MLP_globals.out_dimension,
                                                        MLP_globals.hidden_size,
                                                        model_name_smaller,
                                                        system_used,
                                                        "None")

                    model_lst.append(tempModelLarger)
                    model_lst.append(tempModelSmaller)
                    validation_dataset_lst.append(validation_dataset_larger)
                    validation_dataset_lst.append(validation_dataset_smaller)
                elif model == "svr":
                    models_name_gflops_larger = "svr_gflops_larger_than_cache"
                    models_name_gflops_smaller = "svr_gflops_smaller_than_cache"
                    
                    tempModelLarger = runners.load_svr_model(models_name_gflops_larger, system_used, "None")
                    tempModelSmaller = runners.load_svr_model(models_name_gflops_smaller, system_used, "None")
                    model_lst.append(tempModelLarger)
                    model_lst.append(tempModelSmaller)
                    validation_dataset_lst.append(validation_dataset_larger)
                    validation_dataset_lst.append(validation_dataset_smaller)          
                elif model == "tree":
                    models_name_gflops_larger = "tree_gflops_larger_than_cache"
                    models_name_gflops_smaller = "tree_gflops_smaller_than_cache"
                    
                    tempModelLarger = runners.load_tree_model(models_name_gflops_larger, system_used, "None")
                    tempModelSmaller = runners.load_tree_model(models_name_gflops_smaller, system_used, "None")
                    model_lst.append(tempModelLarger)
                    model_lst.append(tempModelSmaller)
                    validation_dataset_lst.append(validation_dataset_larger)
                    validation_dataset_lst.append(validation_dataset_smaller)
            save_path = "./Performance_Summary/"
            graph_name = "BP_cache_{}".format(system_used)
            runners.plot_performance(model_lst, validation_dataset_lst, model_name_lst, save_path, graph_name)
        else:
            print("Plotting performance for {} without cache split".format(system_used))
            model_name_lst : list = list()
            model_lst : list = list()
            validation_dataset_lst : list[dataReader.SparseMatrixDataset] = list()
            for model in g.models:
                model_name_lst.append("{}_{}".format(model, system_used))
                csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
                validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation, False)
                if model == "mlp":
                    model_name_mlp = "{}_{}epochs".format(model, MLP_globals.nb_epochs)
                    tempModelMlp = runners.load_mlp_model(MLP_globals.activation_fn,
                                                          MLP_globals.nb_hidden_layers,
                                                          MLP_globals.in_dimension,
                                                          MLP_globals.out_dimension,
                                                          MLP_globals.hidden_size,
                                                          model_name_mlp,
                                                          system_used,
                                                          "None")
                    model_lst.append(tempModelMlp)
                    validation_dataset_lst.append(validation_dataset)
                elif model == "svr":
                    model_name_svr = "svr_gflops"
                    tempModelSvr = runners.load_svr_model(model_name_svr, system_used, "None")
                    model_lst.append(tempModelSvr)
                    validation_dataset_lst.append(validation_dataset)
                elif model == "tree":
                    model_name_tree = "tree_gflops"
                    tempModelTree = runners.load_tree_model(model_name_tree, system_used, "None")
                    model_lst.append(tempModelTree)
                    validation_dataset_lst.append(validation_dataset)
            print(model_lst)
            print(model_name_lst)
            save_path = "./Performance_Summary/"
            graph_name = "BP_{}".format(system_used)
            runners.plot_performance(model_lst, validation_dataset_lst, model_name_lst, save_path, graph_name)


    if load_model :
        if model_used == "mlp":
            if cache_split:
                model_name_larger = ""
                model_name_smaller = ""
                csv_path_validation_larger = ""
                csv_path_validation_smaller = ""
                path = ""

                if implementation != "None":
                    model_name_larger = "{}_{}epochs_{}_larger_than_cache".format(model_used, MLP_globals.nb_epochs, implementation)
                    model_name_smaller = "{}_{}epochs_{}_smaller_than_cache".format(model_used, MLP_globals.nb_epochs, implementation)
                    csv_path_validation_larger = g.DATA_PATH + "/validation/all_format/all_format_{}_{}_{}_than_cache.csv".format(system_used, implementation, "larger")
                    csv_path_validation_smaller = g.DATA_PATH + "/validation/all_format/all_format_{}_{}_{}_than_cache.csv".format(system_used, implementation, "smaller")
                    path = g.MODEL_PATH + "{}/{}/{}/{}/".format(system_used, model_used, MLP_globals.nb_epochs, implementation)
                    validation_dataset_larger = dataReader.SparseMatrixDataset(csv_path_validation_larger, True)
                    validation_dataset_smaller = dataReader.SparseMatrixDataset(csv_path_validation_smaller, True)
                    model_larger = runners.load_mlp_model(MLP_globals.activation_fn,
                                                    MLP_globals.nb_hidden_layers,
                                                    MLP_globals.in_dimension-1,
                                                    MLP_globals.out_dimension,
                                                    MLP_globals.hidden_size,
                                                    model_name_larger,
                                                    system_used,
                                                    implementation)
                    model_smaller = runners.load_mlp_model(MLP_globals.activation_fn,
                                                    MLP_globals.nb_hidden_layers,
                                                    MLP_globals.in_dimension-1,
                                                    MLP_globals.out_dimension,
                                                    MLP_globals.hidden_size,
                                                    model_name_smaller,
                                                    system_used,
                                                    implementation)
                    graph_name = "mlp_{}_{}epochs_load".format(implementation, MLP_globals.nb_epochs)
                elif implementation == "None":
                    model_name_larger = "{}_{}epochs_larger_than_cache".format(model_used, MLP_globals.nb_epochs)
                    model_name_smaller = "{}_{}epochs_smaller_than_cache".format(model_used, MLP_globals.nb_epochs)
                    csv_path_validation_larger = g.DATA_PATH + "/validation/all_format/all_format_{}_{}_than_cache.csv".format(system_used,  "larger")
                    csv_path_validation_smaller = g.DATA_PATH + "/validation/all_format/all_format_{}_{}_than_cache.csv".format(system_used, "smaller")
                    path = g.MODEL_PATH + "{}/{}/{}/".format(system_used, model_used, MLP_globals.nb_epochs)
                    validation_dataset_larger = dataReader.SparseMatrixDataset(csv_path_validation_larger, False)
                    validation_dataset_smaller = dataReader.SparseMatrixDataset(csv_path_validation_smaller, False)
                    model_larger = runners.load_mlp_model(MLP_globals.activation_fn,
                                                    MLP_globals.nb_hidden_layers,
                                                    MLP_globals.in_dimension,
                                                    MLP_globals.out_dimension,
                                                    MLP_globals.hidden_size,
                                                    model_name_larger,
                                                    system_used,
                                                    implementation)
                    model_smaller = runners.load_mlp_model(MLP_globals.activation_fn,
                                                    MLP_globals.nb_hidden_layers,
                                                    MLP_globals.in_dimension,
                                                    MLP_globals.out_dimension,
                                                    MLP_globals.hidden_size,
                                                    model_name_smaller,
                                                    system_used,
                                                    implementation)
                    graph_name = "mlp_{}epochs_load".format(MLP_globals.nb_epochs)
                    
                print("Loading mlp model with cache split and {} implementation".format(implementation))
                print("Larger than cache")  
                validation_loader = DataLoader(validation_dataset_larger, batch_size=1, shuffle=True)
                runners.plot_prediction_dispersion_mlp(model_larger, validation_dataset_larger, graph_name, path, implementation, "larger")
                

                print("Smaller than cache")
                validation_loader = DataLoader(validation_dataset_smaller, batch_size=1, shuffle=True)
                runners.plot_prediction_dispersion_mlp(model_smaller, validation_dataset_smaller, graph_name, path, implementation, "smaller")

                model_lst = [model_larger, model_smaller]
                validation_dataset_lst = [validation_dataset_larger, validation_dataset_smaller]
                model_name_lst = [model_name_larger, model_name_smaller]
                runners.plot_performance(model_lst, validation_dataset_lst, model_name_lst)
            else:
                if implementation != "None":
                    print("Loading mlp model without cache split and {} implementation".format(implementation))
                    csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}_{}.csv".format(system_used, implementation)
                    model_name = "{}_{}_{}epochs".format(model_used, implementation, MLP_globals.nb_epochs)
                    model = runners.load_mlp_model(MLP_globals.activation_fn,
                                                   MLP_globals.nb_hidden_layers,
                                                   MLP_globals.in_dimension-1,
                                                   MLP_globals.out_dimension,
                                                   MLP_globals.hidden_size,
                                                   model_name,
                                                   system_used,
                                                   implementation)
                    validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation, True)
                    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
                    graph_name = "mlp_{}_{}epochs".format(implementation, MLP_globals.nb_epochs)
                    path = g.MODEL_PATH + "{}/{}/{}/{}/".format(system_used, model_used, MLP_globals.nb_epochs, implementation)
                    runners.plot_prediction_dispersion_mlp(model, validation_dataset, model_name, path, implementation, "None")
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
                                                system_used,
                                                implementation)
                    
                    validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation, False)
                    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
                    name = "mlp_{}epochs_load".format(MLP_globals.nb_epochs)
                    path = g.MODEL_PATH + "{}/{}/{}".format(system_used, model_used, MLP_globals.nb_epochs)
                    runners.plot_prediction_dispersion_mlp(model, validation_dataset, name, path, implementation, "None")

                    avg_loss_gflops = runners.average_loss_mlp(model, validation_dataset, 0)
                    print("Avg loss of model mlp on gflops : {}%".format(avg_loss_gflops.detach().tolist()*100))

                    avg_loss_energy_efficiency = runners.average_loss_mlp(model, validation_dataset, 1)
                    print("Avg loss of model mlp on energy efficiency : {}%".format(avg_loss_energy_efficiency.detach().tolist()*100))
        
        elif model_used == "svr":
            if cache_split:
                
                model_name_larger = "svr_gflops_larger_than_cache"
                model_name_smaller = "svr_gflops_smaller_than_cache"

                if implementation != "None":
                    csv_path_validation_larger = g.DATA_PATH + "/validation/all_format/all_format_{}_{}_larger_than_cache.csv".format(system_used, implementation)
                    csv_path_validation_smaller = g.DATA_PATH + "/validation/all_format/all_format_{}_{}_smaller_than_cache.csv".format(system_used, implementation)
                    validation_dataset_larger = dataReader.SparseMatrixDataset(csv_path_validation_larger, True)
                    validation_dataset_smaller = dataReader.SparseMatrixDataset(csv_path_validation_smaller, True)
                    path = g.MODEL_PATH + "{}/svr/{}".format(system_used, implementation)
                if implementation == "None":
                    csv_path_validation_larger = g.DATA_PATH + "/validation/all_format/all_format_{}_larger_than_cache.csv".format(system_used)
                    csv_path_validation_smaller = g.DATA_PATH + "/validation/all_format/all_format_{}_smaller_than_cache.csv".format(system_used)
                    validation_dataset_larger = dataReader.SparseMatrixDataset(csv_path_validation_larger, False)
                    validation_dataset_smaller = dataReader.SparseMatrixDataset(csv_path_validation_smaller, False)
                    path = g.MODEL_PATH + "{}/svr".format(system_used)
                
                print("Larger than cache gflops")
                models_name_gflops = "svr_gflops_larger_than_cache"
                model_gflops_larger = runners.load_svr_model(models_name_gflops, system_used, implementation)
                validation_loader = DataLoader(validation_dataset_larger, batch_size=1, shuffle=True)
                graph_name_gflops = "svr_load_gflops"
                runners.plot_prediction_dispersion_sklearn(model_gflops_larger, validation_dataset_larger, graph_name_gflops, path, 0, model_used, implementation, "larger")

                print("Smaller than cache gflops")
                models_name_gflops = "svr_gflops_smaller_than_cache"
                model_gflops_smaller = runners.load_svr_model(models_name_gflops, system_used, implementation)
                validation_loader = DataLoader(validation_dataset_smaller, batch_size=1, shuffle=True)
                graph_name_gflops = "svr_load_gflops"
                runners.plot_prediction_dispersion_sklearn(model_gflops_smaller, validation_dataset_smaller, graph_name_gflops, path, 0, model_used, implementation, "smaller")
            
                model_lst = [model_gflops_larger, model_gflops_smaller]
                validation_dataset_lst = [validation_dataset_larger, validation_dataset_smaller]
                model_name_lst = [model_name_larger, model_name_smaller]
                runners.plot_performance(model_lst, validation_dataset_lst, model_name_lst)
            else:
                if implementation != "None":
                    csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}_{}.csv".format(system_used, implementation)
                    validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation, True)
                    path = g.MODEL_PATH + "{}/svr/{}".format(system_used, implementation)        
                if implementation == "None":
                    csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
                    validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation, False)
                    path = g.MODEL_PATH + "{}/svr".format(system_used)
                
                
                models_name_gflops = "svr_gflops"
                model_gflops = runners.load_svr_model(models_name_gflops, system_used, implementation)
                validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
                graph_name_gflops = "svr_load"
                runners.plot_prediction_dispersion_sklearn(model_gflops, validation_dataset, graph_name_gflops, path, 0, model_used, implementation, "None")
        
        elif model_used == "tree":
            if cache_split:
                model_name_larger = "tree_gflops_larger_than_cache"
                model_name_smaller = "tree_gflops_smaller_than_cache"
                
                if implementation != "None":
                    csv_path_validation_larger = g.DATA_PATH + "/validation/all_format/all_format_{}_{}_larger_than_cache.csv".format(system_used, implementation)
                    csv_path_validation_smaller = g.DATA_PATH + "/validation/all_format/all_format_{}_{}_smaller_than_cache.csv".format(system_used, implementation)
                    validation_dataset_larger = dataReader.SparseMatrixDataset(csv_path_validation_larger, True)
                    validation_dataset_smaller = dataReader.SparseMatrixDataset(csv_path_validation_smaller, True)
                    path = g.MODEL_PATH + "{}/tree/{}".format(system_used, implementation)
                if implementation == "None":
                    csv_path_validation_larger = g.DATA_PATH + "/validation/all_format/all_format_{}_larger_than_cache.csv".format(system_used)
                    csv_path_validation_smaller = g.DATA_PATH + "/validation/all_format/all_format_{}_smaller_than_cache.csv".format(system_used)
                    validation_dataset_larger = dataReader.SparseMatrixDataset(csv_path_validation_larger, False)
                    validation_dataset_smaller = dataReader.SparseMatrixDataset(csv_path_validation_smaller, False)
                    path = g.MODEL_PATH + "{}/tree".format(system_used)
                
                print("Larger than cache gflops")
                models_name_gflops = "tree_gflops_larger_than_cache"
                model_gflops = runners.load_tree_model(models_name_gflops, system_used, implementation)
                validation_loader = DataLoader(validation_dataset_larger, batch_size=1, shuffle=True)
                graph_name_gflops = "tree_load_gflops"
                runners.plot_prediction_dispersion_sklearn(model_gflops, validation_dataset_larger, graph_name_gflops, path, 0, model_used, implementation, "larger")

                print("Smaller than cache gflops")
                models_name_gflops = "tree_gflops_smaller_than_cache"
                model_gflops = runners.load_tree_model(models_name_gflops, system_used, implementation)
                validation_loader = DataLoader(validation_dataset_smaller, batch_size=1, shuffle=True)
                graph_name_gflops = "tree_load_gflops"
                runners.plot_prediction_dispersion_sklearn(model_gflops, validation_dataset_smaller, graph_name_gflops, path, 0, model_used, implementation, "smaller")
            else:
                if implementation != "None":
                    csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}_{}.csv".format(system_used, implementation)
                    validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation, True)
                    path = g.MODEL_PATH + "{}/tree/{}".format(system_used, implementation)        
                if implementation == "None":
                    csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
                    validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation, False)
                    path = g.MODEL_PATH + "{}/tree".format(system_used)
                
                
                models_name_gflops = "tree_gflops"
                model_gflops = runners.load_tree_model(models_name_gflops, system_used, implementation)
                validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
                graph_name_gflops = "tree_load_gflops"
                runners.plot_prediction_dispersion_sklearn(model_gflops, validation_dataset, graph_name_gflops, path, 0, model_used, implementation, "None")
          
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
        runners.plot_prediction_dispersion_mlp(mlp_model, validation_dataset, name, path, implementation, "None")

        # Computing average loss on validation dataset
        avg_loss_gflops = runners.average_loss_mlp(mlp_model, validation_dataset, 0)
        print("Avg loss of model mlp on gflops : {}%".format(avg_loss_gflops.detach().tolist()*100))

        avg_loss_energy_efficiency = runners.average_loss_mlp(mlp_model, validation_dataset, 1)
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
        runners.plot_prediction_dispersion_sklearn(model_gflops, validation_dataset, name_gflops, path, 0, model_used, implementation, "None")

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
        if implementation == "None":
            print("Running Tree with {} system without implementation without cache split".format(system_used))
            csv_path = g.DATA_PATH + "/all_format/all_format_{}.csv".format(system_used)
            csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}.csv".format(system_used)
            validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation, False)
            path = g.MODEL_PATH + "{}/tree".format(system_used)
        else:
            print("Running Tree with {} system with {} implementation without cache split".format(system_used, implementation))
            csv_path = g.DATA_PATH + "/all_format/all_format_{}_{}.csv".format(system_used, implementation)
            csv_path_validation = g.DATA_PATH + "/validation/all_format/all_format_{}_{}.csv".format(system_used, implementation)
            validation_dataset = dataReader.SparseMatrixDataset(csv_path_validation, True)
            path = g.MODEL_PATH + "{}/tree/{}".format(system_used, implementation)
        validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
        
        # Running model
        print("Running tree on gflops predictions")
        tree_gflops = runners.run_tree(Tree_globals.max_depth, csv_path, system_used, 0, implementation, "None")
        name_gflops = "tree_gflops"

        # Plotting predictions
        runners.plot_prediction_dispersion_sklearn(tree_gflops, validation_dataset, name_gflops, path, 0, model_used, implementation, "None")

        # Computing average loss on validation dataset
        avg_loss : torch.Tensor = runners.average_loss_sklearn(tree_gflops, validation_dataset, 0)
        print("Avg loss of tree model on gflops : {}%".format(avg_loss.tolist()*100))


        # Running model
        print("Running tree on energy efficiency predictions")
        tree_energy_efficiency = runners.run_tree(Tree_globals.max_depth, csv_path, system_used, 1, implementation, "None")
        name_energy_efficiency = "tree_energy_efficiency"

        # Plotting predictions
        runners.plot_prediction_dispersion_sklearn(tree_energy_efficiency, validation_dataset, name_energy_efficiency, path, 1, model_used, implementation, "None")

        # Computing average loss on validation dataset
        avg_loss : torch.Tensor = runners.average_loss_sklearn(tree_energy_efficiency, validation_dataset, 1)
        print("Avg loss of tree model on energy efficiency : {}%".format(avg_loss.tolist()*100))