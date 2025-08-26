import os

from torchview import draw_graph
from torchviz import make_dot

from src.tabpfn.train_config import config_sample
from tabpfn.scripts.model_builder import get_model, load_model_only_inference

from .. import BASE_DIR

# Load the model for inference
model_tuple, config = load_model_only_inference(
    os.path.join(BASE_DIR, "tabpfn/models_diff/"),
    "prior_diff_real_checkpointkc_n_0_epoch_100.cpkt", "cpu")

model = model_tuple[2]

print(model)

# Set batch size in the config
config_sample['batch_size'] = 4

# Get the model for evaluation
model_2 = get_model(config_sample, 'cpu', should_train=False, verbose=2)
input, targets, single_eval_pos = next(iter(model_2[3]))

# Perform forward pass
output = model(input, single_eval_pos=single_eval_pos)

# Visualize the model using torchviz
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render(os.path.join(BASE_DIR, 'plots/img/transformer_model_viz'))

# Visualize the model using torchview
model_graph = draw_graph(model,
                         input_data=(input, None, single_eval_pos),
                         hide_module_functions=False).visual_graph

model_graph.render(directory='img', cleanup=True)
