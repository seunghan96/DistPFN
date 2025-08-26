# https://github.com/automl/TabPFN/issues/26#issuecomment-1407630200

import json
import os
from tabpfn import TabPFNClassifier
from src import BASE_DIR

classifier = TabPFNClassifier(device='cpu', N_ensemble_configurations=32)

# Save dict as json
# Sort keys to make it easier to compare
json.dump(classifier.c, open(os.path.join(BASE_DIR,"tabpfn_orig_params.json"), "w"), indent=4, sort_keys=True)

