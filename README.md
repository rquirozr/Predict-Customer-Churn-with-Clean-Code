# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The purpose of this project is to show and apply clean code principles to a model trained in an experimental environment, jupyter notebook.
Some principles applied to this project:
- Refactoring
- Modular
- PEP8 styling
- Documentation
- Testing


## Files and data description
This is the structure of the project (only relevant files):

    |-- data                            # data folder containing bank_data.csv
    |-- images                          
        |-- eda                         # descriptive statistics images
        |-- results                     # report of model performance and explainability
    |-- logs                            # log results
    |-- models                          # pickled models, logistics and random forest
    churn_library.py                    # main library that performs data processing and model training
    churn_notebook.ipynb                # initial code in jupyter notebook
    churn_script_logging_and_tests.py   # test library, with logging results
    conftest.py                         # utility for passing variables in test file
    constants.py                        # constants (paths, lists)
    README.md                           # documentaion
    requirements.txt                    # requirements with python dependencies
    .pylintrc                           # utility for allowing typical naming convention in ml code


## Running Files
The project is built using ```python3.7``` and the dependencies listed in ```requirements.txt```.

The steps for running the project is as follows:
1. Install dependencies
    ```bash
    python -m pip install -r requirements.txt
    ```
2. Train Model
    ```bash
    python churn_library.py
    ```
This code will perform the following steps:
- import_data: Load data from path to pandas dataframe
- perform_eda: Generates descriptive statistics and save it into the image directory ```./images/eda/```
- encoder_helper: Feature engineering (mean encoding)
- perform_feature_engineering: More feature engineering (subset and split)
- train_models: Trains 2 models (logistic regression and random forest), save it into pickles files in ```./models/``` and generates 2 reports (performance on classification and feature importance) into the image directory ```./images/results/```

3. Test and log script
    ```bash
    python churn_script_logging_and_tests.py
    ```
This code will test the churn_library module and saves the log messages into ```./logs/churn_library.log```

4. Both scripts have followed the PEP8 coding style, we can validate it using pylint:
    ```bash
    pylint churn_library.py
    ...
    -----------------------------------
    Your code has been rated at 9.72/10    
    ```
    
    ```bash
    pylint churn_script_logging_and_tests.py
    ...
    -----------------------------------
    Your code has been rated at 10.00/10    
    ```
