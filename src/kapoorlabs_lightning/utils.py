import os
from natsort import natsorted


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
