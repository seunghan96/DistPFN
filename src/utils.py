import pandas as pd
import torch
import numpy as np
import openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


def make_serializable(config_sample):
    if isinstance(config_sample, dict):
        config_sample = {
            k: make_serializable(config_sample[k])
            for k in config_sample
        }
    if isinstance(config_sample, list):
        config_sample = [make_serializable(v) for v in config_sample]
    if callable(config_sample):
        config_sample = str(config_sample)
    return config_sample


def get_openml_classification(did,
                              max_samples,
                              multiclass=True,
                              shuffled=True):
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute)
    #print(X)
    #print(y)
    #asdfads
    if not multiclass:
        X = X[y < 2]
        y = y[y < 2]

    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        print("Not a NP Array, skipping")
        #X = np.asarray(X)
        #y = np.asarray(y)
        #print(X)
        #print(y)
        return None, None, None, None

    if not shuffled:
        sort = np.argsort(y) if y.mean() < 0.5 else np.argsort(-y)
        pos = int(y.sum()) if y.mean() < 0.5 else int((1 - y).sum())
        X, y = X[sort][-pos * 2:], y[sort][-pos * 2:]
        y = torch.tensor(y).reshape(2, -1).transpose(0, 1).reshape(-1).flip(
            [0]).float()
        X = (torch.tensor(X).reshape(2, -1,
                                     X.shape[1]).transpose(0, 1).reshape(
                                         -1, X.shape[1]).flip([0]).float())
    else:
        order = np.arange(y.shape[0])
        np.random.seed(13)
        np.random.shuffle(order)
        X, y = torch.tensor(X[order]), torch.tensor(y[order])
    if max_samples:
        X, y = X[:max_samples], y[:max_samples]

    return X, y, list(np.where(categorical_indicator)[0]), attribute_names


def preprocess_impute(x,
                      y,
                      test_x,
                      test_y,
                      impute,
                      one_hot,
                      standardize,
                      cat_features=[]):

    x, y, test_x, test_y = (
        x.cpu().numpy(),
        y.cpu().long().numpy(),
        test_x.cpu().numpy(),
        test_y.cpu().long().numpy(),
    )

    if impute:
        imp_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
        imp_mean.fit(x)
        x, test_x = imp_mean.transform(x), imp_mean.transform(test_x)

    if one_hot:

        def make_pd_from_np(x):
            data = pd.DataFrame(x)
            for c in cat_features:
                data.iloc[:, c] = data.iloc[:, c].astype("int")
            return data

        x, test_x = make_pd_from_np(x), make_pd_from_np(test_x)
        transformer = ColumnTransformer(
            transformers=[(
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_features,
            )],
            remainder="passthrough",
        )
        transformer.fit(x)
        x, test_x = transformer.transform(x), transformer.transform(test_x)

    if standardize:
        scaler = MinMaxScaler()
        scaler.fit(x)
        x, test_x = scaler.transform(x), scaler.transform(test_x)

    return x, y, test_x, test_y

def inv_freq_sampling(X, y, test_size=0.5, shift_degree=1.0, random_state=42):
    """
    Step 1: Split data into train/test using scikit-learn's train_test_split with 50:50 ratio.
    Step 2: Induce class imbalance in the training set by oversampling inversely to class frequency^shift_degree,
            while ensuring each class has at least 10 samples in both train and test sets.

    Parameters:
    - X, y: input features and labels
    - test_size: proportion for the test set (default 0.5)
    - shift_degree: imbalance strength (0 = uniform, higher = stronger imbalance)
    - random_state: random seed for reproducibility

    Returns:
    - X_train_imbalanced, X_test_final, y_train_imbalanced, y_test_final
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Step 1: 50-50 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Step 2: Compute inverse-frequency^shift_degree weights from full data
    class_labels, counts = np.unique(y, return_counts=True)
    freqs = counts / counts.sum()
    imbalance_weights = (1.0 / freqs) ** shift_degree
    imbalance_weights = imbalance_weights / imbalance_weights.sum()

    # Determine number of samples to draw from each class in train set
    total_train_samples = len(y_train)
    samples_per_class = {
        cls: max(10, int(imbalance_weights[i] * total_train_samples))
        for i, cls in enumerate(class_labels)
    }

    # Oversample train set
    X_train_imbalanced, y_train_imbalanced = [], []
    rng = np.random.default_rng(random_state)
    for cls in class_labels:
        cls_indices = np.where(y_train == cls)[0]
        n_samples = samples_per_class[cls]

        if len(cls_indices) == 0:
            continue

        resampled_idx = rng.choice(cls_indices, size=n_samples, replace=True)
        X_train_imbalanced.append(X_train[resampled_idx])
        y_train_imbalanced.append(y_train[resampled_idx])

    # Step 3: Filter test set to ensure at least 10 samples per class
    X_test_filtered, y_test_filtered = [], []
    for cls in class_labels:
        cls_indices = np.where(y_test == cls)[0]
        if len(cls_indices) >= 10:
            X_test_filtered.append(X_test[cls_indices])
            y_test_filtered.append(y_test[cls_indices])
        else:
            # Fill with duplicated samples if fewer than 10
            resampled_idx = rng.choice(cls_indices, size=10, replace=True)
            X_test_filtered.append(X_test[resampled_idx])
            y_test_filtered.append(y_test[resampled_idx])

    return (
        np.vstack(X_train_imbalanced),
        np.vstack(X_test_filtered),
        np.hstack(y_train_imbalanced),
        np.hstack(y_test_filtered)
    )