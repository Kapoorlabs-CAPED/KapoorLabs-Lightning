#!/usr/bin/env python3
"""Print a usage report from the ONEAT app log.

Reads the local backup (data/usage_log.jsonl) by default, falls back to the
live mount path if the backup is missing. Use --source to point elsewhere.

Outputs:
  - Totals by mode (free vs heavy)
  - Submissions in the last 7 days, broken down per ORCID
  - Top users all-time
  - Recent failures (rows with non-null error)

Usage:
    python usage_report.py
    python usage_report.py --source /home/debian/jean-zay/demo/usage_log.jsonl
    python usage_report.py --days 30
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_BACKUP = Path(__file__).resolve().parent / "data" / "usage_log.jsonl"
LIVE_MOUNT = Path("/home/debian/jean-zay/demo/usage_log.jsonl")


def load_events(source):
    events = []
    with source.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


def parse_ts(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def section(title):
    print()
    print(title)
    print("-" * len(title))


def main():
    parser = argparse.ArgumentParser(description="ONEAT usage report")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to usage_log.jsonl. Defaults to local backup, then live mount.",
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Window for the 'recent' breakdown (default: 7).",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="How many top ORCIDs to show all-time (default: 10).",
    )
    args = parser.parse_args()

    if args.source is not None:
        source = args.source
    elif DEFAULT_BACKUP.exists():
        source = DEFAULT_BACKUP
    elif LIVE_MOUNT.exists():
        source = LIVE_MOUNT
    else:
        print("No usage log found. Run backup_usage_log.py first or pass --source.",
              file=sys.stderr)
        return 1

    events = load_events(source)
    print(f"Source: {source}")
    print(f"Total events: {len(events)}")

    if not events:
        return 0

    section("Totals by mode")
    mode_counts = Counter(e.get("mode") for e in events)
    for mode, n in mode_counts.most_common():
        print(f"  {mode or '<none>':<10} {n}")

    section("Submissions vs failures")
    success = sum(1 for e in events if e.get("slurm_id"))
    failed = sum(1 for e in events if e.get("error"))
    print(f"  successful : {success}")
    print(f"  failed     : {failed}")

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    recent = []
    for e in events:
        ts = parse_ts(e.get("ts"))
        if ts and ts >= cutoff:
            recent.append((ts, e))

    section(f"Last {args.days} days — per ORCID")
    per_orcid = defaultdict(lambda: Counter())
    for _, e in recent:
        per_orcid[e.get("orcid") or "<anon>"][e.get("mode") or "<none>"] += 1
    if not per_orcid:
        print("  (no activity in window)")
    else:
        print(f"  {'ORCID':<22} {'free':>6} {'heavy':>6} {'total':>6}")
        rows = sorted(
            per_orcid.items(),
            key=lambda kv: -sum(kv[1].values()),
        )
        for orcid, c in rows:
            total = sum(c.values())
            print(f"  {orcid:<22} {c.get('free', 0):>6} {c.get('heavy', 0):>6} {total:>6}")

    section(f"Top {args.top} ORCIDs all-time (by submission count)")
    all_orcids = Counter(e.get("orcid") or "<anon>" for e in events if e.get("slurm_id"))
    for orcid, n in all_orcids.most_common(args.top):
        print(f"  {orcid:<22} {n}")

    section("Recent failures (last 10)")
    fails = [e for e in events if e.get("error")][-10:]
    if not fails:
        print("  (none)")
    else:
        for e in fails:
            print(f"  {e.get('ts')} {e.get('orcid') or '<anon>':<22} "
                  f"{e.get('mode'):<6} job={e.get('job_id')} "
                  f"error={e.get('error')[:80]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
