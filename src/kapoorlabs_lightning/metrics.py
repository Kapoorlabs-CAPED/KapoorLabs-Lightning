import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from natsort import natsorted
import seaborn as sns


def extract_metrics_from_model(lightning_model, after_final_validation=True):

    if after_final_validation:
        extra_validation_runs = 1
        val_losses = lightning_model.val_losses[:-extra_validation_runs]
        val_accuracies = lightning_model.val_accuracies[:-extra_validation_runs]
        extra_val_loss = lightning_model.val_losses[-1]  # extra run
        extra_val_accuracy = lightning_model.val_accuracies[-1]  # extra run
    else:
        extra_validation_runs = 0
        val_losses = lightning_model.val_losses
        val_accuracies = lightning_model.val_accuracies
        extra_val_loss = None
        extra_val_accuracy = []

    train_losses = lightning_model.train_losses
    epoch_times = lightning_model.epoch_times
    current_epoch = lightning_model.current_epoch

    # check whether scaling c's should be logged
    scaling_cs = []
    try:
        scaling_cs = lightning_model.scaling_cs
    except Exception:
        pass

    # check whether cc activation magnitudes should be logged
    cc_activation_magnitudes = []
    try:
        cc_activation_magnitudes = lightning_model.cc_activation_magnitudes
    except Exception:
        pass

    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "extra_val_loss": extra_val_loss,
        "extra_val_accuracy": extra_val_accuracy,
        "time_per_epoch": epoch_times,
        "current_epoch": current_epoch,
        "scaling_cs": scaling_cs,
        "cc_activation_magnitudes": cc_activation_magnitudes,
    }
    return metrics


def save_metrics_to_file(metrics, path, filename="metrics.json"):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), "w") as file:
        json.dump(metrics, file, indent=4)


def load_metrics_from_file(filename):
    with open(filename) as file:
        data = json.load(file)
    return data


def plot_npz_files(filepaths):
    all_data = {}
    for filepath in filepaths:
        try:
            data = np.load(str(filepath), allow_pickle=True)
        except pickle.UnpicklingError:
            print(f"Error loading data from {filepath}. Skipping this file.")
            continue

        keys = data.files
        keys = sorted(keys, key=lambda x: ("epoch" in x, x), reverse=True)

        for idx, key in enumerate(keys):
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


def best_npz_files(filepaths):
    all_data = {}
    max_values_filename = {}
    for filepath in filepaths:
        try:
            data = np.load(str(filepath), allow_pickle=True)
        except pickle.UnpicklingError:
            print(f"Error loading data from {filepath}. Skipping this file.")
            continue

        keys = data.files
        keys = sorted(keys, key=lambda x: ("epoch" in x, x), reverse=True)
        for idx, key in enumerate(keys):
            data_values = data[key].tolist()

            if "step" not in key:
                if key not in all_data:
                    all_data[key] = data_values
                else:
                    # Ensure steps and values are arrays
                    steps_array = np.array(all_data[key]["steps"])
                    values_array = np.array(all_data[key]["values"])

                    new_steps = np.array(data_values["steps"])
                    new_values = np.array(data_values["values"])

                    # Check if shapes match before calculating mean
                    if (
                        steps_array.shape != new_steps.shape
                        or values_array.shape != new_values.shape
                    ):
                        print(
                            f"Shapes do not match for key: {key}. Skipping this entry."
                        )
                        continue

                    # Update with new values

                    all_data[key]["steps"] = np.mean([steps_array, new_steps], axis=0)
                    if "loss" in key:
                        all_data[key]["values"] = np.min(
                            [values_array, new_values], axis=0
                        )
                    else:
                        all_data[key]["values"] = np.max(
                            [values_array, new_values], axis=0
                        )
                    if key not in max_values_filename or np.max(new_values) > np.max(
                        max_values_filename[key]["values"]
                    ):
                        max_values_filename[key] = {
                            "filename": os.path.basename(filepath),
                            "values": new_values.tolist(),
                        }

    for k, v in all_data.items():
        data_frame = pd.DataFrame.from_dict(all_data[k])

        data_frame = data_frame.drop_duplicates(subset="steps")
        sns.lineplot(x="steps", y="values", data=data_frame, label=k)
        if k in max_values_filename:
            print(f"Max Values in {k}: {max_values_filename[k]['filename']}")
        plt.show()


def average_npz_files(filepaths):
    all_data = {}
    max_values_filename = {}
    for filepath in filepaths:
        try:
            data = np.load(str(filepath), allow_pickle=True)
        except pickle.UnpicklingError:
            print(f"Error loading data from {filepath}. Skipping this file.")
            continue

        keys = data.files
        keys = sorted(keys, key=lambda x: ("epoch" in x, x), reverse=True)
        for idx, key in enumerate(keys):
            data_values = data[key].tolist()

            if "step" not in key:
                if key not in all_data:
                    all_data[key] = data_values
                else:
                    # Ensure steps and values are arrays
                    steps_array = np.array(all_data[key]["steps"])
                    values_array = np.array(all_data[key]["values"])

                    new_steps = np.array(data_values["steps"])
                    new_values = np.array(data_values["values"])

                    # Check if shapes match before calculating mean
                    if (
                        steps_array.shape != new_steps.shape
                        or values_array.shape != new_values.shape
                    ):
                        print(
                            f"Shapes do not match for key: {key}. Skipping this entry."
                        )
                        continue

                    # Update with new values
                    all_data[key]["steps"] = np.mean([steps_array, new_steps], axis=0)
                    all_data[key]["values"] = np.mean(
                        [values_array, new_values], axis=0
                    )
                    if key not in max_values_filename or np.max(new_values) > np.max(
                        max_values_filename[key]["values"]
                    ):
                        max_values_filename[key] = {
                            "filename": os.path.basename(filepath),
                            "values": new_values.tolist(),
                        }

    for k, v in all_data.items():
        data_frame = pd.DataFrame.from_dict(all_data[k])

        data_frame = data_frame.drop_duplicates(subset="steps")
        sns.lineplot(x="steps", y="values", data=data_frame, label=k)
        if k in max_values_filename:
            print(f"Max Values in {k}: {max_values_filename[k]['filename']}")
        plt.show()


def plot_loss_epochs(loss_plots: dict, step, loss, name=""):
    loss_plots[name] = [step, loss]


def plot_generic(generic_plots: dict, step, metric, name=""):
    generic_plots[name] = [step, metric]


def plot_accuracies(accuracy_plots: dict, step, accuracies, name=""):
    if len(accuracies.shape) > 1:
        timesteps = len(accuracies[0])
        for i in range(timesteps):
            accuracy = [acc_per_epoch[i] for acc_per_epoch in accuracies]
    else:
        accuracy = accuracies

    accuracy_plots[name] = [step, accuracy]


def render_dataframe(
    loss_plots,
    accuracy_plots,
    data_frame_name,
    plot_steps,
    plot_epoch,
    steps_per_epoch,
    generic_plots=None,
):
    if steps_per_epoch <= 0:
        steps_per_epoch = 1
    metric_name = data_frame_name.columns[1]
    step = data_frame_name.columns[0]
    y = data_frame_name[metric_name]
    step = data_frame_name[step] / steps_per_epoch
    if "loss" in metric_name:
        if plot_steps:
            if "step" in metric_name:
                plot_loss_epochs(loss_plots, step, y, name=metric_name)
        if plot_epoch:
            if "epoch" in metric_name:
                plot_loss_epochs(loss_plots, step, y, name=metric_name)

    if "accuracy" in metric_name:
        if plot_steps:
            if "step" in metric_name:
                plot_accuracies(accuracy_plots, step, y, metric_name)
        if plot_epoch:
            if "epoch" in metric_name:
                plot_accuracies(accuracy_plots, step, y, metric_name)

    elif generic_plots is not None:
        plot_generic(generic_plots, step, y, metric_name)


def plot_npz(npz_directory, plot_steps=True, plot_epoch=True, steps_per_epoch=1):
    npz_files = [file for file in os.listdir(npz_directory) if file.endswith(".npz")]
    npz_files_with_time = [
        (file, os.path.getctime(os.path.join(npz_directory, file)))
        for file in npz_files
    ]
    sorted_npz_files = sorted(npz_files_with_time, key=lambda x: x[1], reverse=True)
    most_recent_npz_file = sorted_npz_files[0][0]
    npz_columns = None
    loss_plots = {}
    accuracy_plots = {}
    generic_plots = {}
    npz_data = np.load(
        os.path.join(npz_directory, most_recent_npz_file), allow_pickle=True
    )
    npz_columns = npz_data.files
    for column in npz_columns:
        if len(npz_data[column] > 0):
            data_frame = pd.DataFrame(npz_data[column], columns=["step", column])

            render_dataframe(
                loss_plots,
                accuracy_plots,
                data_frame,
                plot_steps=plot_steps,
                plot_epoch=plot_epoch,
                steps_per_epoch=steps_per_epoch,
                generic_plots=generic_plots,
            )
    show_plots(loss_plots, ylabel="Loss")
    show_plots(accuracy_plots, ylabel="Accuracy")
    show_plots(generic_plots, ylabel="")


def stitch_requeue(npz_directory, plot_steps=True, plot_epoch=True, steps_per_epoch=1):
    npz_files = natsorted(os.listdir(npz_directory))
    npz_columns = None
    loss_plots = {}
    accuracy_plots = {}
    for count, npz_file in enumerate(npz_files):
        if npz_columns is not None:
            break
        data_file = os.path.join(npz_directory, npz_file)
        npz_size = os.stat(data_file).st_size
        if npz_size > 0 and ".npz" in npz_file:
            npz_data = np.load(data_file, allow_pickle=True)
            npz_columns = npz_data.files
    data_frame_list: List[pd.DataFrame] = []
    for column in npz_columns:
        data_frame = pd.DataFrame(columns=["step", column])

        data_frame_list.append(data_frame)

    for count, npz_file in enumerate(npz_files):
        data_file = os.path.join(npz_directory, npz_file)
        npz_size = os.stat(data_file).st_size
        if npz_size > 0 and ".npz" in npz_file:
            npz_data = np.load(data_file, allow_pickle=True)
            filled_data_frame_list = [
                pd.DataFrame(npz_data[column], columns=["step", column])
                for column in npz_data.files
            ]
            for j in range(len(data_frame_list)):
                data_frame_name = data_frame_list[j]
                data_frame_name = pd.concat(
                    [data_frame_name, filled_data_frame_list[j]]
                )
                data_frame_list[j] = data_frame_name

    for j in range(len(data_frame_list)):
        data_frame_name = data_frame_list[j]

        render_dataframe(
            loss_plots,
            accuracy_plots,
            data_frame_name,
            plot_steps=plot_steps,
            plot_epoch=plot_epoch,
            steps_per_epoch=steps_per_epoch,
        )
    show_plots(loss_plots, ylabel="Loss")
    show_plots(accuracy_plots, ylabel="Accuracy")


def show_plots(plots, ylabel=""):
    num_plots = len(plots.items())
    num_rows = 1
    num_cols = num_plots
    plt.figure(figsize=(4 * num_cols, 4 * num_rows))

    if num_plots > 0:
        plt.subplots(1, num_plots, figsize=(15, 4))
        for i, (k, v) in enumerate(plots.items()):
            name = k
            x_data, y_data = v
            plt.subplot(num_rows, num_cols, i + 1)
            sns.lineplot(x=x_data, y=y_data)
            plt.title(name)
            plt.ylabel(ylabel, fontsize=15)
            plt.xlabel("Epochs", fontsize=15)

        plt.tight_layout()
        plt.show()


__all__ = [
    "render_dataframe",
    "stitch_requeue",
    "plot_npz",
    "average_npz_files",
    "best_npz_files",
]
