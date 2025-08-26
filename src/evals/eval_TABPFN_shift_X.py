import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from .. import BASE_DIR, OPENML_LIST
from ..utils import get_openml_classification, preprocess_impute, inv_freq_sampling

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

openml_list = OPENML_LIST

DistPFN = False
DistPFN_T = True

def cross_entropy(p, q, eps=1e-12):
    """
    p: (N, C) shape numpy array - target distributions (e.g., one-hot or soft labels)
    q: (C,) shape numpy array - predicted distribution (shared for all samples)
    eps: small value to prevent log(0)
    
    Returns:
        (N,) shape numpy array - cross-entropy for each sample
    """
    p = np.asarray(p)                # shape (N, C)
    q = np.clip(q, eps, 1.0)         # shape (C,)
    log_q = np.log(q)                # shape (C,)
    ce = -np.sum(p * log_q, axis=1)  # shape (N,)
    return ce

def softmax_temperature(p, T=1.0):
    p = p / T[:, np.newaxis]
    p = p - np.max(p, axis=1, keepdims=True)  # 안정성 확보
    exp_p = np.exp(p)
    return exp_p / np.sum(exp_p, axis=1, keepdims=True)

# Hyperparameters
ensemble_configurations = [4, 8, 16, 32]

scores = {}
for did in tqdm(openml_list.index):
    entry = openml_list.loc[did]
    print(entry)
    try:
        X, y, categorical_feats, attribute_names = get_openml_classification(
            int(entry.id), max_samples=4000, multiclass=True, shuffled=True)
    except:
        continue

    with torch.no_grad():
        X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.5,test_size=0.5, random_state=42)
        
        X_train, y_train, X_test, y_test = preprocess_impute(
            X_train,
            y_train,
            X_test,
            y_test,
            impute=True,
            one_hot=True,
            standardize=False,
            cat_features=categorical_feats)

        values = []

        for N_ensemble_configurations in ensemble_configurations:

            try:
                classifier = TabPFNClassifier(
                    device=device,
                    n_estimators=N_ensemble_configurations,
                    )

                start = time.time()
                classifier.fit(X_train, y_train)
                y_prob = classifier.predict_proba(X_test)

                if DistPFN:
                    P_test_avg = y_prob.mean(axis=0)  
                    y_prior_train = np.bincount(y_train) / len(y_train)
                    adjusted = (y_prob * P_test_avg) / (y_prior_train + 1e-8)
                    y_prob = adjusted / adjusted.sum(axis=1, keepdims=True)  
                
                if DistPFN_T:
                    P_test_avg = y_prob.mean(axis=0)
                    y_prior_train = np.bincount(y_train) / len(y_train)
                    tau = cross_entropy(P_test_avg, y_prior_train)
                    P_test_avg = softmax_temperature(P_test_avg, T=tau)
                    adjusted = (y_prob * P_test_avg) / (y_prior_train + 1e-8)
                    y_prob = adjusted / adjusted.sum(axis=1, keepdims=True)
                
                y_pred_cls = y_prob.argmax(axis=1)
                pred_time = time.time() - start

                if y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
                roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
                cross_entropy = log_loss(y_test, y_prob)
                accuracy = accuracy_score(y_test, y_pred_cls)
                values.append((roc_auc, cross_entropy, accuracy, pred_time))

            except ValueError as ve:
                print(ve)
                print("ve", did)
                continue
            except TypeError as te:
                print(te)
                print("te", did)
                continue

        if not values:
            continue

        roc_auc, cross_entropy, accuracy, pred_time = max(values, key=lambda x: x[0])

        scores[entry['Name']] = {
            "roc": roc_auc,
            "pred_time": pred_time,
            "cross_entropy": cross_entropy,
            "accuracy": accuracy
        }
        print(entry['Name'], scores[entry['Name']])

for n, score in scores.items():
    print(n, score)

# Join scores and openml_list on name
scores_df = pd.DataFrame(scores).T
scores_df = scores_df.reset_index()
scores_df.columns = ['Name', 'roc', 'pred_time', 'cross_entropy', 'accuracy']
openml_list = openml_list.reset_index()
result = pd.merge(openml_list, scores_df, on='Name')

if DistPFN_T:
    fn = 'openml_tabpfn_distpfn_t'
elif DistPFN:
    fn = 'openml_tabpfn_distpfn'
else:
    fn = 'openml_tabpfn'
    
result.to_csv(os.path.join(BASE_DIR, f'data/{fn}.csv'), index=False)

roc = sum(s["roc"] for _, s in scores.items()) / len(scores)
# only calculate cross entropy for binary classification
cross_entropy_list = [
    s["cross_entropy"] for _, s in scores.items()
    if s["cross_entropy"] is not None
]
cross_entropy = sum(cross_entropy_list) / len(cross_entropy_list)
accuracy = sum(s["accuracy"] for _, s in scores.items()) / len(scores)
pred_time = sum(s["pred_time"] for _, s in scores.items()) / len(scores)

print(f"No of datasets: {len(scores)}")
print(f"Mean ROC: {round(roc,3)}")
print(f"Mean Cross Entropy: {round(cross_entropy,3)}")
print(f"Mean Accuracy: {round(accuracy,3)}")
print(f"Mean Prediction Time: {round(pred_time,3)}s")
