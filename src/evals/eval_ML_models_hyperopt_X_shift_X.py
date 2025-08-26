import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
import func_timeout
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tqdm import tqdm
import torch
from tabpfn import TabPFNClassifier

from .. import BASE_DIR, OPENML_LIST
from ..utils import get_openml_classification, preprocess_impute, inv_freq_sampling


openml_list = OPENML_LIST

timeout = 30000

classifier_dict = {
    "lr":
    LogisticRegression(),
    "rf":
    RandomForestClassifier(),
    "svm":
    SVC(),
    "mlp":
    MLPClassifier(),
    "knn":
    KNeighborsClassifier(),
    "lgb":
    lgb.LGBMClassifier(),
    "cat":
    CatBoostClassifier(verbose=0),
}


# Set random seed
for key in classifier_dict:
    classifier_dict[key].random_state = 42

scores = {}

for did in tqdm(openml_list.index):
    entry = openml_list.loc[did]
    print(entry)
    try:
        X, y, categorical_feats, attribute_names = get_openml_classification(
            int(entry.id), max_samples=4000, multiclass=True, shuffled=True)
    except:
        continue

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        train_size=0.5,
                                                        test_size=0.5)

    # preprocess_impute
    X_train, y_train, X_test, y_test = preprocess_impute(
        X_train,
        y_train,
        X_test,
        y_test,
        impute=True,
        one_hot=True,
        standardize=True,
        cat_features=categorical_feats,
    )

    for key in classifier_dict:
    
        try:
            start = time.time()

            try:
                # predict on test data
                classifier_dict[key].fit(X_train, y_train)
                train_time = time.time() - start
                y_pred = classifier_dict[key].predict(X_test)
                y_prob = classifier_dict[key].predict_proba(X_test)

                pred_time = time.time() - start

                if y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]

                roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
                cross_entropy = log_loss(y_test, y_prob)
                accuracy = accuracy_score(y_test, y_pred)

            except Exception as e:
                print(e)
                continue

        except ValueError as ve:
            print(ve)
            print("ve", did)
            continue
        except TypeError as te:
            print(te)
            print("te", did)
            continue
        except Exception as e:
            print(e)
            print("e", did)
            continue

        print(
            f"Dataset: {entry['Name']}, Classifier: {key}, ROC: {roc_auc}, Cross-Entropy: {cross_entropy}, Accuracy: {accuracy}, Prediction Time: {pred_time}, Train Time: {train_time}"
        )

        dict_key = f"{entry['Name']}_{key}"
        scores[dict_key] = {
            "Classifier": key,
            "Dataset": entry["Name"],
            "roc": roc_auc,
            "cross_entropy": cross_entropy,
            "accuracy": accuracy,
            "pred_time": pred_time,
            "train_time": train_time,
        }

#Save json just in case
pd.DataFrame(scores).to_json(
    os.path.join(BASE_DIR, 'data/openml_baseline_scores_wo_hyperopt.json'))

scores_df = pd.read_json(
    os.path.join(BASE_DIR, 'data/openml_baseline_scores_wo_hyperopt.json'))
# Join scores and openml_list on name
scores_df = scores_df.T
scores_df = scores_df.reset_index()
# scores_df['Classifier'] = scores_df['Classifier'].str.split('_').str[-1]
scores_df.columns = [
    "index", "classifier", "Name", "roc", "cross_entropy", "accuracy",
    "pred_time", "train_time"
]

openml_list = openml_list.reset_index()
result = pd.merge(openml_list, scores_df, on="Name")
result.to_csv(os.path.join(BASE_DIR,
                           "data/openml_baseline_scores_wo_hyperopt.csv"),
              index=False)

# Calculate mean scores for each classifier
print(
    scores_df.groupby("classifier")[[
        "roc", "cross_entropy", "accuracy", "pred_time", "train_time"
    ]].mean())

# print to latex
# Combine pred_time and train_time
scores_df["pred_time"] = scores_df["pred_time"] + scores_df["train_time"]
print(
    scores_df.groupby("classifier")[[
        "roc", "cross_entropy", "accuracy", "pred_time"
    ]].mean().round(3).T.to_latex(escape=True))
