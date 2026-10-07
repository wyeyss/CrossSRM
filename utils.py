"""Shared helpers for city selection and traffic-window construction.

Portions of this file are adapted from TPB:
https://github.com/zhyliu00/TPB

The original TPB code is released under the MIT License.
See third_party/TPB_LICENSE for the original license notice.
"""

import random

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)


def get_data_list(data_list):
    dlist = []
    split_data_list = list(data_list.split('_'))
    if('chengdu' in split_data_list):
        dlist.append('chengdu_m')
    if('metr' in split_data_list):
        dlist.append('metr-la')
    if('pems' in split_data_list):
        dlist.append('pems-bay')
    if('shenzhen' in split_data_list):
        dlist.append('shenzhen')
    return dlist


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output, means, stds, inter_step):
    """Create windows from X [N, F, T] with a stride of inter_step.

    Returns inputs [B, N, num_timesteps_input, F] and speed targets
    [B, N, num_timesteps_output]. Inputs retain their existing normalization;
    speed targets are restored to their original units using means and stds.
    """

    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(0, X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1, inter_step)]

    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j]*stds[0]+means[0])

    x = torch.from_numpy(np.array(features)).float()
    y = torch.from_numpy(np.array(target)).float()

    return x,y
