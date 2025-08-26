# TabPFN - Replicated & Evaluated

The [TabPFN](https://github.com/automl/TabPFN/) is a neural network that learned to do tabular data prediction.

We replicate the TabPFN model and evaluate it on more than 200 datasets from OpenML.

## Table of Contents
- [TabPFN - Replicated \& Evaluated](#tabpfn---replicated--evaluated)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
      - [Running the demo code](#running-the-demo-code)
      - [Running the trained model](#running-the-trained-model)
      - [Generating datasets](#generating-datasets)
      - [Running evals](#running-evals)
          - [Evaluate basline without hyperparameter tuning:](#evaluate-basline-without-hyperparameter-tuning)
          - [Evaluate basline with hyperparameter tuning:](#evaluate-basline-with-hyperparameter-tuning)
          - [Evaluate TabPFN:](#evaluate-tabpfn)
  - [Evaluation Results](#evaluation-results)
    - [TabPFN (original)](#tabpfn-original)
    - [TabPFN (retrained/modified) - trained for just 5 hours.](#tabpfn-retrainedmodified---trained-for-just-5-hours)
  - [About TabPFN Usage](#about-tabpfn-usage)

## Installation

This project requires Python 3.9 or below. You can set up the environment using the following commands:

```bash
conda create --name tabpfn python=3.9
conda activate tabpfn
```

After setting up the environment, install all the necessary libra

```bash
pip install tabpfn[full]
pip install -r requirements.txt
```

## Getting Started

The project structure is as follows:

```shell
├── README.md
├── requirements.txt    
├── src
│ ├── __init__.py                                   # initializing package
│ ├── data
│ │ ├── openml_baseline_scores.csv                  # evaluation results for baseline models with hyperparameter tuning
│ │ ├── openml_baseline_scores_wo_hyperopt.csv      # evaluation results for baseline models without hyperparameter tuning
│ │ ├── openml_baseline_tabpfn.csv                  # evaluation results for baseline TabPFN
│ │ ├── openml_list.csv
│ │ ├── openml_modified_tabpfn.csv                   # evaluation results for modified TabPFN
│ │ ├── tabpfn_new_params.json                      # training parameters of the modified TabPFN model
│ │ └── tabpfn_orig_params.json                     # training parameters of the original TabPFN model
│ ├── evals
│ │ ├── datasets.py                                 # selecting the datasets from OpenML
│ │ ├── eval_baseline_models.py                     # evaluating baseline models with hyperparameter tuning
│ │ ├── eval_baseline_models_wo_hyperopt.py         # evaluating baseline models without hyperparameter tuning
│ │ ├── eval_tabpfn.py                              # evaluate original and modified TabPFN models
│ │ └── summarize_results.py                        # summarize evaluation results
│ ├── plots
│ │ ├── classifier_decision_boundary.py              # plot classifier decision boundary
│ │ ├── transformer_model_viz.py                    # plot transformer model architecture
│ │ └── visualize_priors.py                         # plot priors
│ ├── tabpfn
│ │ ├── model_configs.py                             # model configurations
│ │ ├── models_diff                                 # model checkpoints
│ │ ├── tabpfn_orig_params.py                       # extract original training params
│ │ ├── train.py                                    # train the modified TabPFN
│ │ └── train_config.py                              # training configurations
│ ├── tabpfn_demo.py                                # TabPFN demo using scikit-learn interface
│ ├── tabpfn_vs_xgb_demo.py                         # TabPFN demo comparison against XGBoost
│ └── utils.py                                      # utility functions
└── training_log.txt
```

#### Running the demo code

See `src/tabpfn_demo.py` for scikit-learn interface and a basic classification task.

```python
python -m src.tabpfn_vs_xgb_demo
# Accuracy (TabPFN) = 97.872%
# Accuracy (XGBoost) = 96.809%
```

#### Running the trained model

Download the retrained/modified model - `prior_diff_real_checkpointkc_n_0_epoch_100.cpkt` and save it to `src/tabpfn/models_diff`
```
cd ./src/tabpfn/models_diff
wget https://github.com/carteakey/tabpfn-eval/raw/main/src/tabpfn/models_diff/prior_diff_real_checkpointkc_n_0_epoch_100.cpkt
```

- Please change BASE_DIR in `src/__init__.py`!
- The execution directory should be the root of the project.

#### Generating datasets

```shell
python3 -m src.evals.datasets
```

#### Running evals

###### Evaluate basline without hyperparameter tuning:
```shell
python -m src.evals.eval_baseline_models_wo_hyperopt
```

###### Evaluate basline with hyperparameter tuning:
```shell
python -m src.evals.eval_baseline_models
```

###### Evaluate TabPFN:
```shell
python -m src.evals.eval_tabpfn
# set to True to use the modified version of TabPFN
# use_mod = False
```

## Evaluation Results

### TabPFN (original)

- No of datasets: 247
- Mean ROC: 0.846
- Mean Cross Entropy: nan
- Mean Accuracy: 0.803
- Mean Prediction Time: 0.233s

### TabPFN (retrained/modified) - trained for just 5 hours.

- No of datasets: 247
- Mean ROC: 0.84
- Mean Cross Entropy: 0.45
- Mean Accuracy: 0.793
- Mean Prediction Time: 0.217s

See `training_log.txt`

## About TabPFN Usage

TabPFN is different from other methods you might know for tabular classification. Here are some key points to note:

- Do not preprocess inputs to TabPFN. TabPFN pre-processes inputs internally.
- TabPFN expects scalar values only (you need to encode categoricals as integers e.g. with OrdinalEncoder).
- TabPFN ensembles multiple input encodings per default.
- TabPFN does not use any statistics from the test set.
- TabPFN is differentiable in principle, only the pre-processing is not and relies on numpy.