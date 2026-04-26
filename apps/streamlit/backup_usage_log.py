#!/usr/bin/env python3
"""Mirror the live ONEAT usage log from the sshfs mount into a local backup.

Run on a cron / by hand. Skips work if:
  - the mount isn't there (sshfs gone) — leaves last good backup intact
  - the source hasn't changed since the last copy (size + mtime match)

The backup file (data/usage_log.jsonl) is gitignored — it contains ORCID iDs
and IP addresses, and KapoorLabs-Lightning is a public repo.

Usage:
    python backup_usage_log.py
"""

import shutil
import sys
from pathlib import Path

SOURCE = Path("/home/debian/jean-zay/demo/usage_log.jsonl")
DEST = Path(__file__).resolve().parent / "data" / "usage_log.jsonl"


def mount_alive(path):
    """True if the sshfs mount is responsive and the source file is readable."""
    try:
        return path.exists() and path.is_file()
    except OSError:
        return False


def needs_copy(src, dst):
    if not dst.exists():
        return True
    s = src.stat()
    d = dst.stat()
    return (s.st_size, int(s.st_mtime)) != (d.st_size, int(d.st_mtime))


def main():
    if not mount_alive(SOURCE):
        print(f"[skip] source not available: {SOURCE}", file=sys.stderr)
        return 0

    DEST.parent.mkdir(parents=True, exist_ok=True)

    if not needs_copy(SOURCE, DEST):
        print(f"[noop] backup up-to-date: {DEST}")
        return 0

    shutil.copy2(SOURCE, DEST)
    size = DEST.stat().st_size
    print(f"[ok]   copied {size} bytes to {DEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
