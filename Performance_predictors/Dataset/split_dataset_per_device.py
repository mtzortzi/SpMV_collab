import pandas as pd

data = pd.read_csv("./data/all_format_runs_March_2023.csv")
architectures = data["System"].unique().tolist()
print(architectures)
