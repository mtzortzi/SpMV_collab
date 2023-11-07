# SpMV_collab

The aim of this project is to build machine learning models to predict the performance of a given architecture (CPU, GPU, FPGA) of the SpMV kernel given a set of features that describes sparce matrices.

## Dataset Info
### Experimental Setup - 9 Devices
| Device  |  Arch |  Formats |
| ------------- | ------------- | ------------- |
| Tesla-P100    | GPU           | cu-COO, cu-CSR, cu-HYB, CSR5 |
| Tesla-V100    | GPU           | cu-COO, cu-CSR, cu-HYB, CSR5 |
| Tesla-A100    | GPU           |  cu-COO, cu-CSR, Merge       |
| AMD-EPYC-64   | CPU           | Naive-CSR, CSR5, MKL-IE      |
| AMD-EPYC-24   | CPU           | Naive-CSR, Vec-CSR, AOCL, MKL-IE, SELL-C-s,Merge-CSR, CSR5, SparseX |
| INTEL-XEON    | CPU           | Naive-CSR, Vec-CSR, MKL-IE, SELL-C-s, Merge-CSR, CSR5, SparseX      |
| ARM-NEON      | CPU           | Naive-CSR, ARM-lib, Merge-CSR, SparseX, SELL-C-s                    |
| IBM-POWER9    | CPU           | Naive-CSR, Bal-CSR, Merge-CSR, SparseX                              |
| Alveo-U280    | FPGA          | Xilinx-lib                                                          |


**Matrix dataset**
  -  ~15260, Artificial matrices

**Two results datasets**
  - All matrices - all format runs (size 568158 x 35)
      > Filename : [all_format_runs_March_2023.csv](https://github.com/mtzortzi/SpMV_collab/blob/main/Performance_predictors/Dataset/data/all_format_runs_March_2023.csv)
  - All matrices - best performing (per Device) format run (size 122941 x 35)
     > Filename: [best_format_runs_March_2023.csv](https://github.com/mtzortzi/SpMV_collab/blob/main/Performance_predictors/Dataset/data/best_format_runs_March_2023.csv)

**35 columns of the two result datasets**
| Feature       |  Description | 
| ------------- | -------------------|
| mtx_name      |        -           |
| distribution  |        -           |
| placement     |        -           |
| seed          |        -           |
| m             | Rows of matrix     |
| n             | Columns of matrix  |
| nz            | Nonzeros of matrix |
| density       | Density of matrix, nz / (m*n) (percentage) |
| A_mem_footprint | Memory footprint of matrix in MBs (for CSR representation of matrix) |
| mem_range    | It will be one of the following, it was used to group many matrices in the same memory range [4-8], [8-16], [16-32], [32-64], [64-128], [128-256], [256-512], [512-1024], [1024-2048] |
| avg_nz_row   | Average number of nonzeros per row |
| std_nz_row   | Standard deviation of nonzeros per row |
| avg_bandwidth | Average bandwidth of matrix, bandwidth is the distance between the first and the last element of a row |
| std_bandwidth | Standard deviation of bandwidth |
| avg_bandwidth_scaled | This is scaled, divided by number of columns of matrix, in order to be in range [0,1] for all matrices |
| std_bandwidth_scaled  | Scaled standard deviation |
| skew_coeff           | Skew coefficient of row size, calculates how unbalanced a matrix is Calculated as (max-avg)/avg (of row size) |
| avg_num_neighbours | Average of neighbors per row. We define as “neighbors” of a nonzero element all the other same-row elements residing in a predetermined maximum column distance of 1, left or right of the element. (range [0-2]) (Captures spatial locality on vector x) |
| cross_row_similarity | Average of cross-row similarity, which expresses a measure of similarity between adjacent rows. It calculates how many nonzeros reside in a column distance of one, which means either the same column or the adjacent left and right ones. (range [0-1]) (Captures temporal locality on vector x) |
| implementation | SpMV format |
| time | Time in microseconds (I think?) - ignore it - gflops is more important |
| gflops | Performance of SpMV, measured in GFLOPs For SpMV it is calculated as 2*nz / time |
| W_avg | Watt of device during execution of SpMV |
| J_estimated | Joules consumed during execution of SpMV |
| System | Device (listed above) |
| Arch | GPU or CPU or FPGA |
| friends | - |
| impl_arch | Combination of implementation + device columns E.g. “( Naive CSR )  AMD-EPYC-24” |
| energy_efficiency | GFLOPs per Watt, an energy efficiency metric |
| GFLOPs^2-per-W | Another energy efficiency metric, focusing more on performance |
| crs_categ | Category of cross row similarity feature, in Small, Medium or Large ( in ranges [0-0.3], [0.3-0.7], [0.7-1] respectively ) |
| ann_categ | Category of average neighbors feature, in Small, Medium or Large ( in ranges [0-0.6], [0.6-1.4], [1.4-2] respectively ) |
| regularity | Combination of previous two (crs_categ, ann_categ) |
| anr_categ | Range of average nonzeros per row, each matrix will be in one of these [0-15], [15-40], [40-75], [75-150], [150-510] |
| skew_categ | Range of skew coefficient, each matrix will be in one of these [0-1.5], [1.5-50], [50-250], [250-3000], [3000-10000] |



 




















## Running models
### MLP
Multi-Layer Perceptron (MLP) is widely used in data science in order to make predictions given a set of features. Generally speaking, MLP neural networks are characterized by several features :
* Input dimension
* Output dimension
* Number of hidden layers
* Dimensions of those hidden layers

To have a better understanding of those hyperparameters you can take a look at the following schema : 

![image](./img/mlp_architecturel.jpg)

In our model we've chosen to have an input dimension of 7 for all of these features :
* A_mem_footprint
* avg_nz_row
* skew_coeff
* avg_num_neighbours
* cross_row_similarity
* avg_bandwidth_scaled
* implementation

And an output dimension of 2 for these features :
* GFLOPs
* Energry efficiency

For further explanation of the data preprocessing see [Dataset_section](#dataset).

### SVR
Support Vector Regression (SVR) is based on Support Vector Machines (SVM). Without going into details SVM modifies dimension of our working space in order to make non-linear separable population of data separable. Then with the "Kernel trick" we project back our data in our original space. Here is a visual example :
![image](./img/kernel_trick.png)

In our project we use SVR in order to find a model by using the kernel trick to make prediction of the GFLOPs and the energy efficiency or our system given sparse matrix features

### Decision trees
As explained in the documentation of the scikit-learn library those non-parametric supervised learning models represent some advantages and drawbacks primarly listed below :\
\
Advantages :
* Simple to understand and to interpret. Trees can be visualized.
* Requires little data preparation. Other techniques often require data normalization, dummy variables need to be created and blank values to be removed. Some tree and algorithm combinations support missing values.
* The cost of using the tree (i.e., predicting data) is logarithmic in the number of data points used to train the tree.
* Uses a white box model. If a given situation is observable in a model, the explanation for the condition is easily explained by boolean logic. By contrast, in a black box model (e.g., in an artificial neural network), results may be more difficult to interpret.
* Possible to validate a model using statistical tests. That makes it possible to account for the reliability of the model.

\
Drawbacks :
* Decision-tree learners can create over-complex trees that do not generalize the data well. This is called overfitting. Mechanisms such as pruning, setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are necessary to avoid this problem.
* Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an ensemble.
* Predictions of decision trees are neither smooth nor continuous, but piecewise constant approximations. Therefore, they are not good at extrapolation.
* Decision tree learners create biased trees if some classes dominate. It is therefore recommended to balance the dataset prior to fitting with the decision tree.


## Folders and file architectures
### Dataset
In the Dataset folder you will find all the data that is needed for us to train our models. In addition to that you can find some python scripts that aims to reshape data and split dataset. The dataset is splited in a way that each row that corresponds to a given system will in their own csv file. Also some data samples have been added in order to have a better view of the real data.

### Saved models
The saved_model folder is where we store all the results from our trained model. You will find given on the system that our model was trained the binary file of the corresponding model as well as a plot that sums up the training history.

### Models
For each model that is implemented in this projet, a specific folder is created where the following is stored :
* ``model.py`` : this is the file where the model class as well as the train function and other functianlities relative to the model is written.
* ``globals.py`` : this is the file where all the parameters fo the model are stored.


### Main program
The main program consist of several bricks to work. First it retrives parameters from the command line in order to work. Then given of those parameters parsed the corresponding runner is called from the ``model_runner.py`` file. Also from this file before runnning the program data is preprocessed from the ``dataReader.py`` file. In this file we extract the corresponding features from our dataset :
* A_mem_footprint
* avg_nz_row
* skew_coeff
* avg_num_neighbours
* cross_row_similarity
* avg_bandwidth_scaled
* system

All of these features needs to be scaled from 0 to 1 (except the avg_bandwidth_scaled that is already scaled and system that is a string). We scale our data in order to increase stability in the learning process. For the system feature, given the fact that it's a string we used a one hot encoding system, where we retrived each unique class and associate it a unique number in [0, nb_class - 1].

Finally in the ``globals.py`` file you will find the global path for the dataset and the saved models, the different hardware system that are implemented in the dataset and the implemented models.


## How to make it work on your computer
### Adding a new model
If you want to add a new model you must to the following :
1. Create a new folder associated to your model with the ``model.py`` and ``globals.py`` associated in it. Also it is mandatory that your model class inherits the torch.nn.Module class.
2. Add the corresponding function to run your model in the runner
3. Add the name of your implemented model in the `models` array in the globals.py file.
4. Finally in the main program add a hookup to the parameter parser in order to run or load ur model when executing the program.

### Running it on your computer
```
usage: main.py [-h] -m MODEL -s SYSTEM [-l]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model name to run
  -s SYSTEM, --system SYSTEM
                        CPU/GPU name
  -l, --load            Load the model described from it's hyperparameters in it's 
  corresponfing global.py file and the -m parameter described above
````
