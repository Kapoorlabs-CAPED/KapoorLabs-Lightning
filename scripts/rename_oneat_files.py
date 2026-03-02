#!/usr/bin/env python3
"""
Rename ONEAT CSV files from CamelCase to snake_case.
ONEATMitosis... → oneat_mitosis_...
ONEATNormal... → oneat_normal_...
ONEATApoptosis... → oneat_apoptosis_...
"""

import os
import re
import sys
from pathlib import Path


def rename_files(directory, dry_run=True):
    """Rename ONEAT files to snake_case."""
    directory = Path(directory)

    # Pattern to match ONEAT files
    pattern = re.compile(r'^ONEAT(Mitosis|Normal|Apoptosis)(.*)$')

    renamed = []

    for f in sorted(directory.glob('ONEAT*.csv')):
        old_name = f.name
        match = pattern.match(old_name)

        if match:
            event_type = match.group(1).lower()  # Mitosis → mitosis
            rest = match.group(2)  # everything after the event type
            new_name = f"oneat_{event_type}_{rest}"

            # Clean up any double underscores
            new_name = re.sub(r'_+', '_', new_name)

            new_path = f.parent / new_name

            print(f"{old_name}")
            print(f"  → {new_name}")

            if not dry_run:
                f.rename(new_path)
                renamed.append((old_name, new_name))

    return renamed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rename_oneat_files.py <directory> [--execute]")
        print("  Default is dry-run (shows what would be renamed)")
        print("  Add --execute to actually rename files")
        sys.exit(1)

    directory = sys.argv[1]
    dry_run = "--execute" not in sys.argv

    if dry_run:
        print("=== DRY RUN (no files will be renamed) ===\n")
    else:
        print("=== RENAMING FILES ===\n")

    renamed = rename_files(directory, dry_run=dry_run)

    if dry_run:
        print("\nRun with --execute to apply changes")
