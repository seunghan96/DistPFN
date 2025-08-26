import os

import matplotlib.pyplot as plt
import numpy as np

from src.tabpfn.train_config import config_sample
from tabpfn.priors.utils import plot_features
from tabpfn.scripts.model_builder import get_model

from .. import BASE_DIR


def main():
    config_sample['batch_size'] = 4
    model = get_model(config_sample, 'cpu', should_train=False, verbose=2)

    (_, data, _), targets, _ = next(iter(model[3]))

    plot_prior_features(data, targets)
    plot_prior_features_corr(data, targets)


def plot_prior_features(data, targets):
    fig = plt.figure(figsize=(8, 8))
    N = 100
    plot_features(data[0:N, 0, 0:4], targets[0:N, 0], fig=fig)
    plt.savefig(os.path.join(BASE_DIR, 'plots/img/prior_features.png'))


def plot_prior_features_corr(data, targets):
    d = np.concatenate([data[:, 0, :].T, np.expand_dims(targets[:, 0], -1).T])
    d[np.isnan(d)] = 0
    c = np.corrcoef(d)
    plt.matshow(np.abs(c), vmin=0, vmax=1)
    plt.savefig(os.path.join(BASE_DIR, 'plots/img/prior_features_corr.png'))


if __name__ == "__main__":
    main()
