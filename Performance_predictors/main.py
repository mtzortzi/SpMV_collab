import getopt, sys
import model_runners as runners
import MLP.globals as MLP_globals
import dataReader
import globals as g

if __name__ == "__main__":
    argumentList = sys.argv[1:]
    options = "hm:s:"
    lst_options = ["help", "model="]
    selectect_system = ""
    try:
        arguments, values = getopt.getopt(argumentList, options, lst_options)
        for currentArg, currentVal in arguments:
            if currentArg in ("-h", "--help"):
                print("Help")
            elif currentArg in ("-s", "--system"):
                if currentVal in g.hardware:
                    selectect_system = currentVal
                    print("Selected system : {}".format(currentVal))
                else:
                    print("No valid system found for {}".format(currentVal))
                    print("Available systems \n{}".format(g.hardware))
            elif currentArg in ("-m", "--model"):
                print("Enabling model {}".format(currentVal))
                if currentVal  == "mlp":
                    print("running mlp model")
                    csv_path = g.DATA_PATH + "/all_format/all_format_{}.csv".format(selectect_system)
                    runners.run_mlp(MLP_globals.activation_fn,
                                    MLP_globals.nb_hidden_layers,
                                    MLP_globals.in_dimension,
                                    MLP_globals.out_dimension,
                                    MLP_globals.hidden_size,
                                    csv_path)
                elif currentVal == "svr":
                    dataset = dataReader.SparseMatrixDataset("./Dataset/data/data_sample.csv")
                    print("running Support Vector Regression model")
    except getopt.error as err:
        print(str(err))