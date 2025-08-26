import time
import os
from tabpfn.scripts.model_builder import get_model, save_model
from src.tabpfn.train_config import config_sample
from src import BASE_DIR

device = "cuda"
base_path = os.path.join(BASE_DIR, "tabpfn/")

max_features = 100
maximum_runtime = 60 * 1  # in minutes

Uncomment the following lines to load a previously trained model
model_state, optimizer_state, config_sample = torch.load(
    os.path.join(BASE_DIR, "tabpfn/models_diff",
                 "prior_diff_real_checkpointkc_n_0_epoch_16.cpkt"))


def train_function(config_sample, i, add_name=""):
    start_time = time.time()
    N_epochs_to_save = 50

    def save_callback(model, epoch):
        if not hasattr(model, "last_saved_epoch"):
            model.last_saved_epoch = 0
        if ((time.time() - start_time) /
            (maximum_runtime * 60 /
             N_epochs_to_save)) > model.last_saved_epoch:
            print("Saving model..")
            config_sample["epoch_in_training"] = epoch
            save_model(
                model,
                base_path,
                f"models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{model.last_saved_epoch}.cpkt",
                config_sample,
            )
            model.last_saved_epoch = (model.last_saved_epoch + 1)

    get_model(
        config_sample,
        device,
        should_train=True,
        verbose=1,
        epoch_callback=save_callback,
    )

    return

train_function(config_sample, 0, add_name="kc")
