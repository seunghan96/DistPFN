import json
import os
from datetime import datetime

from src import BASE_DIR
from src.tabpfn.model_configs import *
from src.utils import make_serializable


def reload_config(config_type="causal", task_type="multiclass", longer=0):
    config = get_prior_config(config_type=config_type)

    config["prior_type"], config["differentiable"], config["flexible"] = (
        "prior_bag",
        True,
        True,
    )

    model_string = ""

    config["epochs"] = 12000
    config["recompute_attn"] = True

    config["max_num_classes"] = 10
    config["num_classes"] = uniform_int_sampler_f(2, config["max_num_classes"])
    config["balanced"] = False
    model_string = model_string + "_multiclass"

    model_string = model_string + "_" + datetime.now().strftime(
        "%m_%d_%Y_%H_%M_%S")

    return config, model_string


config, model_string = reload_config(longer=1)

config["bptt_extra_samples"] = None
# del config["differentiable_hyperparameters"]["output_multiclass_ordered_p"]
config["sampling"] = "mixed"
config["dynamic_batch_size"] = False
config["multiclass_loss_type"] = "nono"
config["normalize_to_ranking"] = False
config["categorical_feature_p"] = 0.2
config["nan_prob_no_reason"] = 0.0
config["nan_prob_unknown_reason"] = 0.0
config["set_value_to_nan"] = 0.1
config["normalize_with_sqrt"] = False
config["new_mlp_per_example"] = True
config["prior_mlp_scale_weights_sqrt"] = True
config["batch_size_per_gp_sample"] = None
config["normalize_ignore_label_too"] = True
config["differentiable_hps_as_style"] = False
config["max_eval_pos"] = 1000
config["random_feature_rotation"] = True
config["rotate_normalized_labels"] = True
config["mix_activations"] = True
config["emsize"] = 512
config["nhead"] = config["emsize"] // 128
config[
    "bptt"] = 2048  # Extended to 2048 to have effective sequence length of 2048
config["canonical_y_encoder"] = False
config["aggregate_k_gradients"] = 8
config["batch_size"] = 128 * config[
    "aggregate_k_gradients"]  # Extended to 128 to have effective batch size of 1024
config["num_steps"] = 1024 // config["aggregate_k_gradients"]
config["epochs"] = 400
config["total_available_time_in_s"] = None
config["train_mixed_precision"] = True
config["efficient_eval_masking"] = True

config_sample = evaluate_hypers(config)
config_sample_json = make_serializable(config_sample)

# Save dict to json serializing the objects
json.dump(config_sample_json,
          open(os.path.join(BASE_DIR, "data/tabpfn_new_params.json"), "w"),
          indent=4,
          sort_keys=True)
