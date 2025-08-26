import pandas as pd
import warnings
import os

BASE_DIR = '/home/seunghan9613/tabpfn-eval/src'
OPENML_LIST = pd.read_csv(os.path.join(BASE_DIR, "data/openml_list.csv"))

# ignore warnings
def warn(*args, **kwargs):
    pass

warnings.warn = warn

