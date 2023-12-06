import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing



class SparseMatrixDataset(Dataset):
    def __init__(self, csv_file, using_implementation_split : bool):
        self.features = ["A_mem_footprint", 
            "avg_nz_row", 
            "skew_coeff", 
            "avg_num_neighbours",
            "cross_row_similarity",
            "avg_bandwidth_scaled"]
        
        self.scaled_features = ["avg_bandwidth_scaled", "avg_scattering"]

        self.dataframe = pd.read_csv(csv_file)

        # Scaling first row
        self.scalers = dict()
        first_row = self.dataframe[self.features[0]].to_numpy()
        first_row_scaled, self.scalers[self.features[0]] = self.scale_row(first_row)
        self.x = torch.tensor(first_row_scaled, dtype=torch.float32)

        # Scaling all rows except for the "avg_bandwidth_scaled" that is already scaled
        for feature in self.features[1:-1]:
            row = self.dataframe[feature].to_numpy()
            row_scaled, self.scalers[feature] = self.scale_row(row)
            self.x = torch.cat((self.x, torch.tensor(row_scaled, dtype=torch.float32)), 1)

        # Adding the bandwidth feature to the X entry
        for feature in self.scaled_features:
            scaled_feature = torch.tensor(self.dataframe[feature].to_numpy().reshape(-1, 1), dtype=torch.float32)
            self.x = torch.cat((self.x, scaled_feature), 1)

        avg_bandwidth_scaled = torch.tensor(self.dataframe[self.features[-2]].to_numpy().reshape(-1, 1), dtype=torch.float32)
        self.x = torch.cat((self.x, avg_bandwidth_scaled), 1)
        
        if not(using_implementation_split):
            # Adding the implementation feature to the X entry
            implementation_labels = self.dataframe["implementation"].unique().tolist() # Getting all unique labels of implementation
            self.mappings = {k: i for i, k in enumerate(implementation_labels)} # Mapping one-hot-encoded labels

            encoded_implementation = self.dataframe["implementation"].apply(lambda x: self.mappings[x]).to_numpy().reshape((-1, 1)) # Apply one hot encoded featre
            t_encoded_implementation = torch.as_tensor(encoded_implementation, dtype=torch.float32) # Transfroming it into a tensor
            self.x = torch.cat((self.x, t_encoded_implementation), 1)

        # Scaling gflops output
        gflops = self.dataframe[["gflops"]].values
        self.scaler_gflops = preprocessing.MinMaxScaler().fit(gflops)
        gflops_scaled = torch.tensor(self.scaler_gflops.transform(gflops), dtype=torch.float32)

        # Scaling energy_efficiency output
        energy_efficiency = torch.tensor(self.dataframe[["energy_efficiency"]].values, dtype=torch.float32)
        self.scaler_energy_efficiency = preprocessing.MinMaxScaler().fit(energy_efficiency)
        energy_efficiency_scaled = torch.tensor(self.scaler_energy_efficiency.transform(energy_efficiency), dtype=torch.float32)

        # Adding the scaled glops and energy_efficiency to output
        self.y = torch.cat((gflops_scaled, energy_efficiency_scaled), 1)

    

    def scale_row(self, row):
        scaler = preprocessing.MinMaxScaler()
        row_scaled = scaler.fit_transform(row.reshape(-1, 1))
        return row_scaled, scaler
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
