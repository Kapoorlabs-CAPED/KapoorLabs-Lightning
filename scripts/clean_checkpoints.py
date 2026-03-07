import os
import re

# Configuration
directory = "/projects/extern/nhr/nhr_ni/nhr_ni_test/nhr_ni_test_27040/dir.project/oneat_mitosis_model_adam/"  
keep_n_first = 2
keep_n_middle = 2
keep_n_last = 2

def parse_checkpoint_name(filename):
    """Extract epoch and step numbers from checkpoint filename."""
    match = re.search(r'epoch=(\d+)-step=(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None

def clean_ckpt_files(directory, keep_n_first, keep_n_middle, keep_n_last):
    # Get all checkpoint files
    files = [f for f in os.listdir(directory) if f.endswith(".ckpt")]
    
    # Parse and sort by epoch, then step
    file_data = []
    for f in files:
        epoch, step = parse_checkpoint_name(f)
        if epoch is not None:
            file_data.append((f, epoch, step))
    
    # Sort by epoch first, then by step
    file_data.sort(key=lambda x: (x[1], x[2]))
    sorted_files = [os.path.join(directory, f[0]) for f in file_data]
    
    if len(sorted_files) <= (keep_n_first + keep_n_middle + keep_n_last):
        print(f"Nothing to delete. Only {len(sorted_files)} files found.")
        return
    
    # Select files to keep
    first_files = sorted_files[:keep_n_first]
    last_files = sorted_files[-keep_n_last:]
    
    # Middle files from the center of the list
    middle_start = (len(sorted_files) - keep_n_middle) // 2
    middle_files = sorted_files[middle_start:middle_start + keep_n_middle]
    
    files_to_keep = set(first_files + middle_files + last_files)
    files_to_delete = [f for f in sorted_files if f not in files_to_keep]
    
    # Delete files
    for file in files_to_delete:
        print(f"Deleting: {os.path.basename(file)}")
        os.remove(file)
    
    print(f"\nKept {len(files_to_keep)} files:")
    print(f"  First {keep_n_first}: {[os.path.basename(f) for f in first_files]}")
    print(f"  Middle {keep_n_middle}: {[os.path.basename(f) for f in middle_files]}")
    print(f"  Last {keep_n_last}: {[os.path.basename(f) for f in last_files]}")

# Run cleanup
clean_ckpt_files(directory, keep_n_first, keep_n_middle, keep_n_last)
