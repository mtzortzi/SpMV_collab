import pandas as pd
import torch
import os
from sklearn import preprocessing

def scale_row(row):
        scaler = preprocessing.MinMaxScaler()
        row_scaled = scaler.fit_transform(row.reshape(-1, 1))
        return row_scaled

features = ["A_mem_footprint", 
            "avg_nz_row", 
            "skew_coeff", 
            "avg_num_neighbours",
            "cross_row_similarity",]

header = ["mtx_name","distribution","placement","seed","m","n","nz","density","A_mem_footprint","mem_range","avg_nz_row","std_nz_row","avg_bandwidth","std_bandwidth","avg_bandwidth_scaled","std_bandwidth_scaled","avg_scattering","std_scattering","avg_scattering_scaled","std_scattering_scaled","skew_coeff","avg_num_neighbours","cross_row_similarity","implementation","time","gflops","W_avg","J_estimated","System","Arch","friends","impl_arch","energy_efficiency","GFLOPs^2-per-W","crs_categ","ann_categ","regularity","anr_categ","skew_categ",]

scaled_features = ["avg_bandwidth_scaled",
                   "avg_scattering_scaled"]



dataframe = pd.read_csv('./Dataset/data/all_format/all_format_AMD-EPYC-24.csv',
                        sep=",",
                        
                        encoding='utf-8')

print(dataframe.head())


# chunk_size = 10  # Adjust the chunk size as needed
# chunks = pd.read_csv('./Dataset/data/validation/all_format/all_format_AMD-EPYC-24.csv', header=0, sep=",", chunksize=chunk_size)
# for chunk in chunks:
# 	print(set(chunk.columns))

# dataframe = pd.read_csv('./Dataset/data/all_format_runs_March_2023.csv',
#                         sep=",",
#                         encoding='utf-8')
print(dataframe.columns)


print(dataframe.info())
print("\n================")


first_row = dataframe[features[0]].to_numpy()
first_row_scaled = scale_row(first_row)
x = torch.tensor(first_row_scaled, dtype=torch.float32)

for feature in features[1:]:
    print(feature)
    row = dataframe[feature].to_numpy()
    row_scaled = scale_row(row)
    x = torch.cat((x, torch.tensor(row_scaled, dtype=torch.float32)), 1)

for feature in scaled_features:
    print(feature)
    scaled_feature = torch.tensor(dataframe[feature].to_numpy().reshape(-1, 1), dtype=torch.float32)
    x = torch.cat((x, scaled_feature), 1)
