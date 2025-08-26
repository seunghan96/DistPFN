# Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment

- Backbone: **TabPFN-v2** ([[Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6)]) 
- Dataset: 253 OpenML datasets.

<br>

## DistPFN & DistPFN-T 
**Plug-in** methods for any **tabular foundation models based on in-context learning**
```python
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
```


<br>


## Step 1) Installation
```bash
conda create --name tabpfn python=3.9
conda activate tabpfn
pip install tabpfn[full]
pip install -r requirements.txt
```

- Please change BASE_DIR in `src/__init__.py`!

<br>

## Step 2) Generating datasets

```shell
python3 -m src.evals.datasets
```
<br>

## Step 3) Evaluation

```shell
python -m src.evals.eval_TABPFN_shift_O # w/ label shift
python -m src.evals.eval_TABPFN_shift_X # w/o label shift
```
