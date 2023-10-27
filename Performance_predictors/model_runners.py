import MLP.globals as MLP_globals
import MLP.model as MLP_model
import dataReader
import torch
import matplotlib.pyplot as plt
import numpy as np

def run_mlp(activation_function,
            nb_hidden_layers,
            in_dimension,
            out_dimension,
            hidden_size,
            csv_path):
    print("running MLP")

    mlp_model = MLP_model.MlpPredictor(activation_function,
                                       nb_hidden_layers,
                                       in_dimension,
                                       out_dimension,
                                       hidden_size,)
    
    dataset = dataReader.SparseMatrixDataset(csv_path)
    optimizer = torch.optim.SGD(mlp_model.parameters(), lr=MLP_globals.lr)

    costTbl, countTbl = MLP_model.train(mlp_model, 
                                            MLP_globals.nb_epochs, 
                                            dataset, 
                                            MLP_globals.loss_fn,
                                            optimizer)
    plt.plot(countTbl, costTbl)
    plt.savefig("temp.png")
    plt.show()
