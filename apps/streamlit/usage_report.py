#!/usr/bin/env python3
"""Print + plot a usage report from the ONEAT app log.

Reads the live mount path by default, falls back to the local backup if the
mount is unreachable. Use --source to point elsewhere.

Outputs to stdout:
  - Totals by mode (free vs heavy)
  - Submissions in the last N days, broken down per ORCID
  - Top users all-time
  - Recent failures

Outputs PNG plots into apps/streamlit/data/usage_plots/:
  - daily_submissions.png       — stacked bar of free vs heavy per day
  - cumulative_submissions.png  — cumulative count over time
  - per_orcid_totals.png        — bar of total submissions by ORCID
  - hour_of_day.png             — submissions by hour of day (UTC)

Usage:
    python usage_report.py
    python usage_report.py --days 30
    python usage_report.py --source /home/debian/jean-zay/demo/usage_log.jsonl
    python usage_report.py --no-plots
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

LIVE_MOUNT = Path("/home/debian/jean-zay/demo/usage_log.jsonl")
LOCAL_BACKUP = Path(__file__).resolve().parent / "data" / "usage_log.jsonl"
PLOTS_DIR = Path(__file__).resolve().parent / "data" / "usage_plots"


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


def resolve_source(arg):
    """Pick the source file: explicit > live mount > local backup."""
    if arg is not None:
        return arg
    try:
        if LIVE_MOUNT.exists() and LIVE_MOUNT.is_file():
            return LIVE_MOUNT
    except OSError:
        pass
    if LOCAL_BACKUP.exists():
        return LOCAL_BACKUP
    return None


def make_plots(events):
    """Render PNG plots into PLOTS_DIR. Imports matplotlib lazily."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[plots] matplotlib not installed — skipping plots.", file=sys.stderr)
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Successful events with timestamps
    rows = []
    for e in events:
        ts = parse_ts(e.get("ts"))
        if ts is None:
            continue
        rows.append({
            "ts": ts,
            "date": ts.date(),
            "hour": ts.hour,
            "mode": e.get("mode") or "unknown",
            "orcid": e.get("orcid") or "<anon>",
            "ok": e.get("slurm_id") is not None,
        })
    if not rows:
        print("[plots] no usable events — skipping plots.", file=sys.stderr)
        return

    # 1. Daily stacked bar: free vs heavy
    days = sorted({r["date"] for r in rows})
    free_per_day = [sum(1 for r in rows if r["date"] == d and r["mode"] == "free")
                    for d in days]
    heavy_per_day = [sum(1 for r in rows if r["date"] == d and r["mode"] == "heavy")
                     for d in days]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(days, free_per_day, label="free", color="#60a5fa")
    ax.bar(days, heavy_per_day, bottom=free_per_day, label="heavy (A100)",
           color="#22c55e")
    ax.set_title("Daily ONEAT submissions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Submissions")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "daily_submissions.png", dpi=120)
    plt.close(fig)

    # 2. Cumulative submissions over time
    rows_sorted = sorted(rows, key=lambda r: r["ts"])
    times = [r["ts"] for r in rows_sorted]
    cum = list(range(1, len(times) + 1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, cum, color="#0f172a")
    ax.fill_between(times, cum, alpha=0.2, color="#0f172a")
    ax.set_title("Cumulative ONEAT submissions")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Total submissions")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cumulative_submissions.png", dpi=120)
    plt.close(fig)

    # 3. Per-ORCID totals (top 20)
    orcid_counts = Counter(r["orcid"] for r in rows)
    top = orcid_counts.most_common(20)
    if top:
        labels = [o for o, _ in top]
        counts = [c for _, c in top]
        fig, ax = plt.subplots(figsize=(10, max(3, 0.4 * len(top))))
        ax.barh(labels, counts, color="#a855f7")
        ax.invert_yaxis()
        ax.set_title("Submissions per ORCID iD (top 20)")
        ax.set_xlabel("Submissions")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "per_orcid_totals.png", dpi=120)
        plt.close(fig)

    # 4. Hour-of-day histogram (when do users actually run?)
    hours = [0] * 24
    for r in rows:
        hours[r["hour"]] += 1
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(range(24), hours, color="#f59e0b")
    ax.set_title("Submissions by hour of day (UTC)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Submissions")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "hour_of_day.png", dpi=120)
    plt.close(fig)

    print(f"[plots] wrote 4 PNGs to {PLOTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="ONEAT usage report")
    parser.add_argument(
        "--source", type=Path, default=None,
        help="Path to usage_log.jsonl. Default: live mount, then local backup.",
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Window for the 'recent' breakdown (default: 7).",
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="How many top ORCIDs to show all-time (default: 10).",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation.",
    )
    args = parser.parse_args()

    source = resolve_source(args.source)
    if source is None:
        print("No usage log found. Tried live mount and local backup.",
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
    recent = [(parse_ts(e.get("ts")), e) for e in events]
    recent = [(ts, e) for ts, e in recent if ts and ts >= cutoff]

    section(f"Last {args.days} days — per ORCID")
    per_orcid = defaultdict(lambda: Counter())
    for _, e in recent:
        per_orcid[e.get("orcid") or "<anon>"][e.get("mode") or "<none>"] += 1
    if not per_orcid:
        print("  (no activity in window)")
    else:
        print(f"  {'ORCID':<22} {'free':>6} {'heavy':>6} {'total':>6}")
        rows = sorted(per_orcid.items(), key=lambda kv: -sum(kv[1].values()))
        for orcid, c in rows:
            total = sum(c.values())
            print(f"  {orcid:<22} {c.get('free', 0):>6} {c.get('heavy', 0):>6} {total:>6}")

    section(f"Top {args.top} ORCIDs all-time (by successful submission count)")
    all_orcids = Counter(
        e.get("orcid") or "<anon>" for e in events if e.get("slurm_id")
    )
    for orcid, n in all_orcids.most_common(args.top):
        print(f"  {orcid:<22} {n}")

    section("Recent failures (last 10)")
    fails = [e for e in events if e.get("error")][-10:]
    if not fails:
        print("  (none)")
    else:
        for e in fails:
            err = (e.get("error") or "")[:80]
            print(f"  {e.get('ts')} {e.get('orcid') or '<anon>':<22} "
                  f"{e.get('mode'):<6} job={e.get('job_id')} error={err}")

    if not args.no_plots:
        make_plots(events)

    return 0


if __name__ == "__main__":
    sys.exit(main())
