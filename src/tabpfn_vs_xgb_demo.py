from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tabpfn import TabPFNClassifier


def tabpfn_classifier(X_train, y_train, X_test):
    # N_ensemble_configurations controls the number of model predictions that are ensembled with feature and class rotations (See our work for details).
    # # When N_ensemble_configurations > #features * #classes, no further averaging is applied.
    tab_pfn_model = TabPFNClassifier(device='cpu',
                                     N_ensemble_configurations=32)
    tab_pfn_model.fit(X_train, y_train)
    y_pred, p_pred = tab_pfn_model.predict(X_test,
                                           return_winning_probability=True)
    return y_pred, p_pred


def xgboost_classifier(X_train, y_train, X_test):
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    return y_pred


# Load data
X, y = load_breast_cancer(return_X_y=True)

# Train classifier
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.33,
                                                    random_state=42)

# TabPFN model
y_pred, p_pred = tabpfn_classifier(X_train, y_train, X_test)
tabpfn_accuracy = 100 * accuracy_score(y_test, y_pred)

# XGBoost model
y_pred = xgboost_classifier(X_train, y_train, X_test)
xgb_accuracy = 100 * accuracy_score(y_test, y_pred)

# Results
print(f'Accuracy (TabPFN) = {tabpfn_accuracy:.3f}%')
print(f'Accuracy (XGBoost) = {xgb_accuracy:.3f}%')
