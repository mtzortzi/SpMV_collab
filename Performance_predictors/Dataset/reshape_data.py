import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import preprocessing

features = ["A_mem_footprint", 
            "avg_nz_row", 
            "skew_coeff", 
            "avg_num_neighbours",
            "cross_row_similarity", 
            "avg_bandwidth_scaled"]

dataframe = pd.read_csv("./data/data_sample_2.csv") # Getting all data

# Scaling first row
first_row = dataframe[features[0]].to_numpy()
scaler = preprocessing.MinMaxScaler()
first_row_scaled = scaler.fit_transform(first_row.reshape(-1, 1))
X = torch.tensor(first_row_scaled, dtype=torch.float32)

# Scaling all rows except for the "avg_bandwidth_scaled" that is already scaled
for feature in features[1:-1]:
    row = dataframe[feature].to_numpy()
    
    scaler = preprocessing.MinMaxScaler()
    row_scaled = scaler.fit_transform(row.reshape(-1, 1))
    X = torch.cat((X, torch.tensor(row_scaled, dtype=torch.float32)), 1)

# Adding the bandwidth feature to the X entry
avg_bandwidth_scaled = torch.tensor(dataframe[features[-1]].to_numpy().reshape(-1, 1), dtype=torch.float32)
X = torch.cat((X, avg_bandwidth_scaled), 1)

# Adding the implementation feature to the X entry
implementation_labels = dataframe["implementation"].unique().tolist() # Getting all unique labels of implementation
mappings = {k: i for i, k in enumerate(implementation_labels)} # Mapping one-hot-encoded labels

encoded_implementation = dataframe["implementation"].apply(lambda x: mappings[x]).to_numpy().reshape((-1, 1)) # Apply one hot encoded featre
t_encoded_implementation = torch.as_tensor(encoded_implementation, dtype=torch.float32) # Transfroming it into a tensor
X = torch.cat((X, t_encoded_implementation), 1)

print(X)

# Scaling gflops output
gflops = dataframe[["gflops"]].values
scaler = preprocessing.MinMaxScaler().fit(gflops)
gflops_scaled = torch.tensor(scaler.transform(gflops), dtype=torch.float32)

energy_efficiency = torch.tensor(dataframe[["energy_efficiency"]].values, dtype=torch.float32)

# Adding the scaled glops to the energy_efficiency output
Y = torch.cat((gflops_scaled, energy_efficiency), 1)
print(Y)