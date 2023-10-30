import model_runners as runners
import MLP.globals as MLP_globals
import SVR.globals as SVR_globals
import dataReader
import globals as g
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', metavar='MODEL', required=True, help='Model to run')
    parser.add_argument('-s', '--system', metavar='SYSTEM', required=True, help='CPU/GPU name')
    parser.add_argument('-l', '--load', action='store_true')

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
            csv_path = g.DATA_PATH + "/all_format/all_format_{}.csv".format(system_used)
            model_name = "{}_{}epochs".format(model_used, MLP_globals.nb_epochs)
            model = runners.load_mlp_model(MLP_globals.activation_fn,
                                           MLP_globals.nb_hidden_layers,
                                           MLP_globals.in_dimension,
                                           MLP_globals.out_dimension,
                                           MLP_globals.hidden_size,
                                           model_name,
                                           system_used)
            dataset = dataReader.SparseMatrixDataset(csv_path)
            runners.predict(model, dataset)
            

    elif model_used == "mlp":
        csv_path = g.DATA_PATH + "/all_format/all_format_{}.csv".format(system_used)
        runners.run_mlp(MLP_globals.activation_fn,
                        MLP_globals.nb_hidden_layers,
                        MLP_globals.in_dimension,
                        MLP_globals.out_dimension,
                        MLP_globals.hidden_size,
                        csv_path,
                        system_used)
        
        
    elif model_used == "svr":
        csv_path = g.DATA_PATH + "/all_format/all_format_{}.csv".format(system_used)
        runners.run_svr(SVR_globals.kernel,
                        SVR_globals.C,
                        SVR_globals.epsilon,
                        SVR_globals.gamma,
                        csv_path)
        print("running Support Vector Regression model")