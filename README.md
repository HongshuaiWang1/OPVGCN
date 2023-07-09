# OPVGCN Fast and accurate screening framework for organic solar cells based on molecular structure and deep learning

## Introduction

A Self-Learning-Input Graph Neural Network introduces a dynamic embedding layer to accept the feedback of backpropagation during the training process and introduce the Infomax mechanism to maximize the correlation between the local features and the global features.

## Dependencies

The project is built using the Python language and the following third-party frameworks:

```pyton
SLI-GNN
rdkit
lightgbm 
```

# Installation

Before downloading, please ensure that other dependencies have been installed.

First, create a new conda environment

```shell
conda create --name version python=3.7
```
conda install -c conda-forge rdkit,lightgbm

SLI-GNN

```shell
git clone https://github.com/Austin6035/SLI-GNN.git
```

# Model 1

## Dataset
Two datasets were used during the training of model 1
one of which is from the CEP database, which includes hundreds of thousands of molecular structures and properties: https://www.matter.toronto.edu/basic-content-page/data-download
Another data set is constructed by ourselves, including 440 published opv molecular structures and PCE in /data/train.db & test.db with sqlite3 format.


### Running

Training sample data and other parameter descriptions can be viewed using the following command `python trainer.py -h`. Combined with `ray-tune`, automatic parameter tune can be realized. Depending on the task type, the results will be saved in the `results/regression/` or `results/classification/` directory, and the loss during training will be saved in the `results/` directory, and the log information during training will be saved in the `log/` directory.

```
python trainer.py sample-dataset sample-targets
```

### Testing

After the training is complete, the best model will be saved to the `weight/` directory, and you can use `test.py` for testing. When testing, there can only be a material_id column in the target property file.

```shell
python test.py model_best.pth.tar sample-dataset sample-targets
```

# Model 2

 Model 2 utilizes the output of Model 1 as input, which can be easily implemented using lightgbm
### Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

 Set up the LightGBM parameters
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9
}

### Train the LightGBM model
num_rounds = 100
model = lgb.train(params, train_data, num_rounds)

### Make predictions
y_pred = model.predict(X_test)



