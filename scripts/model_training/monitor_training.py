#!/usr/bin/env python3
"""
Training Monitor Script
Monitors model folders for new checkpoints and generates plots.
Runs continuously, checking every 5 minutes.
"""

import time
import json
from pathlib import Path
from datetime import datetime
from kapoorlabs_lightning.utils import plot_npz_files_interactive

# Configuration
BASE_DIR = Path("/projects/extern/nhr/nhr_ni/nhr_ni_test/nhr_ni_test_27040/dir.project")
CHECK_INTERVAL = 300  # 5 minutes in seconds
STATE_FILE = Path(__file__).parent / ".monitor_state.json"

# Model folders to monitor
MODEL_FOLDERS = [
    "oneat_mitosis_model",
    "oneat_mitosis_model_light",
    "oneat_mitosis_model_heavy",
    "oneat_mitosis_model_adam",
    "oneat_mitosis_model_adam_light",
    "oneat_mitosis_model_adam_heavy",
]

# Output directory for plots (relative to script location)
PLOTS_OUTPUT_DIR = Path(__file__).parent / "training_plots"


def load_state():
    """Load the state of processed checkpoints."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    """Save the state of processed checkpoints."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_latest_checkpoint(model_dir):
    """Get the latest .ckpt file in a model directory."""
    ckpt_files = list(model_dir.glob("*.ckpt"))
    if not ckpt_files:
        return None, None

    latest = max(ckpt_files, key=lambda x: x.stat().st_mtime)
    mtime = latest.stat().st_mtime
    return str(latest), mtime


def get_latest_npz(model_dir):
    """Get the latest .npz file in a model directory."""
    npz_files = list(model_dir.glob("*.npz"))
    if not npz_files:
        return None
    return max(npz_files, key=lambda x: x.stat().st_mtime)


def generate_plots(model_dir, output_subdir):
    """Generate plots for a model directory."""
    npz_file = get_latest_npz(model_dir)
    if npz_file is None:
        print(f"  No .npz file found in {model_dir.name}")
        return False

    output_dir = PLOTS_OUTPUT_DIR / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Generating plots from {npz_file.name}")
    try:
        plot_npz_files_interactive(
            [npz_file],
            save_plots=True,
            show_plots=False,
            page_output_dir=str(output_dir)
        )
        return True
    except Exception as e:
        print(f"  Error generating plots: {e}")
        return False


def check_and_process():
    """Check all model folders and process new checkpoints."""
    state = load_state()
    updated = False

    print(f"\n{'='*60}")
    print(f"Checking models at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    for folder_name in MODEL_FOLDERS:
        model_dir = BASE_DIR / folder_name

        if not model_dir.exists():
            print(f"[SKIP] {folder_name} - directory not found")
            continue

        ckpt_path, ckpt_mtime = get_latest_checkpoint(model_dir)

        if ckpt_path is None:
            print(f"[SKIP] {folder_name} - no checkpoint yet")
            continue

        # Check if this checkpoint was already processed
        prev_mtime = state.get(folder_name, {}).get("mtime", 0)

        if ckpt_mtime > prev_mtime:
            print(f"[NEW]  {folder_name} - new checkpoint detected")

            if generate_plots(model_dir, folder_name):
                state[folder_name] = {
                    "mtime": ckpt_mtime,
                    "checkpoint": ckpt_path,
                    "processed_at": datetime.now().isoformat()
                }
                updated = True
                print(f"       Plots generated successfully")
            else:
                print(f"       Failed to generate plots")
        else:
            print(f"[OK]   {folder_name} - no new checkpoint")

    save_state(state)

    return updated


def main():
    """Main monitoring loop."""
    print("="*60)
    print("ONEAT Training Monitor")
    print("="*60)
    print(f"Monitoring {len(MODEL_FOLDERS)} model folders")
    print(f"Check interval: {CHECK_INTERVAL} seconds ({CHECK_INTERVAL//60} minutes)")
    print(f"Plots output: {PLOTS_OUTPUT_DIR}")
    print(f"State file: {STATE_FILE}")
    print("="*60)
    print("Press Ctrl+C to stop\n")

    # Create output directory
    PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            check_and_process()
            print(f"\nNext check in {CHECK_INTERVAL//60} minutes...")
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")


if __name__ == "__main__":
    main()
