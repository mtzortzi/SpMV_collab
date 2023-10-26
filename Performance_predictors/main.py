import getopt, sys
import model_runners as runners
import MLP.globals as MLP_globals
import dataReader
import globals as g

if __name__ == "__main__":
    argumentList = sys.argv[1:]
    options = "hm:"
    lst_options = ["help", "model="]
    try:
        arguments, values = getopt.getopt(argumentList, options, lst_options)
        for currentArg, currentVal in arguments:
            if currentArg in ("-h", "--help"):
                print("Help")
            elif currentArg in ("-m", "--model"):
                print("Enabling model {}".format(currentVal))
                if currentVal  == "mlp":
                    print("running mlp model")
                    runners.run_mlp(MLP_globals.activation_fn,
                                    MLP_globals.nb_hidden_layers,
                                    MLP_globals.in_dimension,
                                    MLP_globals.out_dimension,
                                    MLP_globals.hidden_size,
                                    g.DATA_PATH)
                elif currentVal == "svr":
                    dataset = dataReader.SparseMatrixDataset("./Dataset/data/data_sample.csv")
                    print("running Support Vector Regression model")
    except getopt.error as err:
        print(str(err))