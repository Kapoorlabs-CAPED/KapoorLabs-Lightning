import os
from natsort import natsorted
import numpy as np
import pandas as pd
import seaborn as sns
import pickle 
import matplotlib.pyplot as plt

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
    present_files = os.listdir(log_path)
    hpc_ckpt_files = []
    other_ckpt_files = []
    ckpt_path = None
    for file in present_files:
        if file.endswith(".ckpt") and "hpc" in file:
            hpc_ckpt_files.append(file)
        if file.endswith(".ckpt") and "hpc" not in file:
            other_ckpt_files.append(file)
    if len(hpc_ckpt_files) > 0:
        hpc_ckpt_files = natsorted(hpc_ckpt_files)
        ckpt_path = os.path.join(log_path, hpc_ckpt_files[-1])
    if len(hpc_ckpt_files) == 0 and len(other_ckpt_files) > 0:
        other_ckpt_files = natsorted(other_ckpt_files)
        ckpt_path = os.path.join(log_path, other_ckpt_files[-1])

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
        unwanted_substrings = ["step","gpu", "memory"]
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