import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import torch

logger = logging.getLogger(__name__)




def get_most_recent_file(file_path, file_pattern):
    ckpt_files = [file for file in os.listdir(file_path) if file.endswith(file_pattern)]
    if len(ckpt_files) > 0:
        ckpt_files_with_time = [
            (file, os.path.getctime(os.path.join(file_path, file)))
            for file in ckpt_files
        ]

        sorted_ckpt_files = sorted(
            ckpt_files_with_time, key=lambda x: x[1], reverse=True
        )

        most_recent_ckpt = sorted_ckpt_files[0][0]

        return os.path.join(file_path, most_recent_ckpt)
    else:
        return None


def load_checkpoint_model(log_path: str):

    ckpt_path = get_most_recent_file(log_path, ".ckpt")

    return ckpt_path


def plot_npz_files(filepaths):
    all_data = {}
    for filepath in filepaths:
        try:
            data = np.load(str(filepath), allow_pickle=True)
        except pickle.UnpicklingError:
            # print(f"Error loading data from {filepath}. Skipping this file.")
            continue

        keys = data.files
        keys = sorted(keys, key=lambda x: ("epoch" in x, x), reverse=True)
        unwanted_substrings = ["step", "gpu", "memory"]
        for idx, key in enumerate(keys):
            if not any(substring in key for substring in unwanted_substrings):
                data_values = data[key].tolist()
                if key not in all_data:
                    all_data[key] = data_values
                else:
                    all_data[key]["steps"].extend(data_values["steps"])
                    all_data[key]["values"].extend(data_values["values"])
    for k, v in all_data.items():
        data_frame = pd.DataFrame.from_dict(all_data[k])
        sns.lineplot(x="steps", y="values", data=data_frame, label=k)
        plt.show()


def blockwise_sum(
    A: torch.Tensor, timepoints: torch.Tensor, dim: int = 0, reduce: str = "sum"
):
    if not A.shape[dim] == len(timepoints):
        raise ValueError(
            f"Dimension {dim} of A ({A.shape[dim]}) must match length of timepoints"
            f" ({len(timepoints)})"
        )

    A = A.transpose(dim, 0)

    if len(timepoints) == 0:
        logger.warning("Empty timepoints in block_sum. Returning zero tensor.")
        return A
    # -1 is the filling value for padded/invalid timepoints
    min_t = timepoints[timepoints >= 0]
    if len(min_t) == 0:
        logger.warning("All timepoints are -1 in block_sum. Returning zero tensor.")
        return A

    min_t = min_t.min()
    # after that, valid timepoints start with 1 (padding timepoints will be mapped to 0)
    ts = torch.clamp(timepoints - min_t + 1, min=0)
    index = ts.unsqueeze(1).expand(-1, len(ts))
    blocks = ts.max().long() + 1
    out = torch.zeros((blocks, A.shape[1]), device=A.device, dtype=A.dtype)
    out = torch.scatter_reduce(out, 0, index, A, reduce=reduce)
    B = out[ts]
    B = B.transpose(0, dim)

    return B


def blockwise_causal_norm(
    A: torch.Tensor,
    timepoints: torch.Tensor,
    mode: str = "softmax",
    mask_invalid: torch.BoolTensor = None,
    eps: float = 1e-6,
):
    """Normalization over the causal dimension of A.

    For each block of constant timepoints, normalize the corresponding block of A
    such that the sum over the causal dimension is 1.

    Args:
        A (torch.Tensor): input tensor
        timepoints (torch.Tensor): timepoints for each element in the causal dimension
        mode: normalization mode.
            `linear`: Simple linear normalization.
            `softmax`: Apply exp to A before normalization.
            `quiet_softmax`: Apply exp to A before normalization, and add 1 to the denominator of each row/column.
        mask_invalid: Values that should not influence the normalization.
        eps (float, optional): epsilon for numerical stability.
    """
    assert A.ndim == 2 and A.shape[0] == A.shape[1]
    A = A.clone()

    if mode in ("softmax", "quiet_softmax"):
        # Subtract max for numerical stability
        # https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning
        # TODO test without this subtraction

        if mask_invalid is not None:
            assert mask_invalid.shape == A.shape
            A[mask_invalid] = -torch.inf
        # TODO set to min, then to 0 after exp

        # Blockwise max
        ma0 = blockwise_sum(A, timepoints, dim=0, reduce="amax")
        ma1 = blockwise_sum(A, timepoints, dim=1, reduce="amax")

        u0 = torch.exp(A - ma0)
        u1 = torch.exp(A - ma1)
    elif mode == "linear":
        A = torch.sigmoid(A)
        if mask_invalid is not None:
            assert mask_invalid.shape == A.shape
            A[mask_invalid] = 0

        u0, u1 = A, A
        ma0 = ma1 = 0
    else:
        raise NotImplementedError(f"Mode {mode} not implemented")

    # get block boundaries and normalize within blocks
    # bounds = _bounds_from_timepoints(timepoints)
    # u0_sum = _blockwise_sum_with_bounds(u0, bounds, dim=0) + eps
    # u1_sum = _blockwise_sum_with_bounds(u1, bounds, dim=1) + eps

    u0_sum = blockwise_sum(u0, timepoints, dim=0) + eps
    u1_sum = blockwise_sum(u1, timepoints, dim=1) + eps

    if mode == "quiet_softmax":
        # Add 1 to the denominator of the softmax. With this, the softmax outputs can be all 0, if the logits are all negative.
        # If the logits are positive, the softmax outputs will sum to 1.
        # Trick: With maximum subtraction, this is equivalent to adding 1 to the denominator
        u0_sum += torch.exp(-ma0)
        u1_sum += torch.exp(-ma1)

    mask0 = timepoints.unsqueeze(0) > timepoints.unsqueeze(1)
    # mask1 = timepoints.unsqueeze(0) < timepoints.unsqueeze(1)
    # Entries with t1 == t2 are always masked out in final loss
    mask1 = ~mask0

    # blockwise diagonal will be normalized along dim=0
    res = mask0 * u0 / u0_sum + mask1 * u1 / u1_sum
    res = torch.clamp(res, 0, 1)
    return res
