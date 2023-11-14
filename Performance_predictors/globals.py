DATA_PATH = "./Dataset/data/"

hardware = ["Tesla-P100", 
            "Tesla-V100", 
            "AMD-EPYC-24", 
            "ARM-NEON", 
            "INTEL-XEON", 
            "IBM-POWER9", 
            "Tesla-A100", 
            "Alveo-U280", 
            "AMD-EPYC-64"]

MODEL_PATH = "./saved_models/"

models = ["mlp",
          "svr",
          "tree"]

IMPLEMENTATIONS_AMD_EPYC_24 = ['AOCL', 'CSR5', 'Vec-CSR', 'Merge-CSR', 'MKL-IE', 'Naive-CSR', 'SELL-C-s', 'SparseX']
IMPLEMENTATIONS_TESLA_A100 = ['Merge', 'cu-COO', 'cu-CSR']