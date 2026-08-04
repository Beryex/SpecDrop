"""Probe SuperNI Domains distribution AFTER hierarchical normalization.

Applies:
  1. Split multi-label Domains (avg 1.25/task) — take first label as primary
  2. Normalize hierarchy: "Social Media -> Twitter" -> "Social Media" (root)
  3. Report frequency-sorted distribution for top-K + miscellaneous cutoff decision

Also fetches FULL train split (756 tasks) not just 150-sample to give final mapping.
Runtime ~10 min (756 GET requests threaded).
"""
import json
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://raw.githubusercontent.com/allenai/natural-instructions/master"
TRAIN_URL = f"{BASE}/splits/default/train_tasks.txt"
TEST_URL = f"{BASE}/splits/default/test_tasks.txt"


def fetch(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "specdrop/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return f.read()


def fetch_task(task_id):
    try:
        raw = fetch(f"{BASE}/tasks/{task_id}.json")
        return task_id, json.loads(raw)
    except Exception:
        return task_id, None


def normalize_domain(d):
    """Take root of hierarchical path: 'A -> B -> C' -> 'A'."""
    if not d:
        return None
    head = d.split(" -> ")[0].strip()
    return head if head else None


def categorize(tasks, strategy):
    """Return Counter mapping normalized_domain -> count. Single-label per task."""
    counts = Counter()
    assigned = 0
    for task_id, d in tasks:
        if d is None:
            continue
        doms = d.get("Domains", []) or []
        # strategy='first': take first listed domain; 'all': each label counted (multi-label)
        if strategy == "first":
            for dm in doms[:1]:
                norm = normalize_domain(dm)
                if norm:
                    counts[norm] += 1
                    assigned += 1
                    break
        elif strategy == "all":
            for dm in doms:
                norm = normalize_domain(dm)
                if norm:
                    counts[norm] += 1
                    assigned += 1
        else:
            raise ValueError(strategy)
    return counts, assigned


def main():
    print("Fetching TRAIN + TEST task lists...")
    train = [ln.strip() for ln in fetch(TRAIN_URL).decode().splitlines() if ln.strip()]
    test = [ln.strip() for ln in fetch(TEST_URL).decode().splitlines() if ln.strip()]
    print(f"  TRAIN: {len(train)}  TEST: {len(test)}")

    all_tasks = train + test
    print(f"\nFetching all {len(all_tasks)} task JSONs (threaded, ~5 min)...")
    tasks = []
    with ThreadPoolExecutor(max_workers=30) as pool:
        futures = {pool.submit(fetch_task, t): t for t in all_tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            tasks.append(fut.result())
            if i % 100 == 0:
                print(f"  {i}/{len(all_tasks)} fetched")

    ok = sum(1 for _, d in tasks if d is not None)
    print(f"  OK: {ok}/{len(all_tasks)}")

    # Separate train/test after full fetch
    train_set = set(train)
    train_tasks = [(t, d) for t, d in tasks if t in train_set and d is not None]
    test_tasks = [(t, d) for t, d in tasks if t not in train_set and d is not None]
    print(f"  Train parsed: {len(train_tasks)}  Test parsed: {len(test_tasks)}")

    print("\n" + "=" * 70)
    print(" DOMAIN DISTRIBUTION (hierarchy-normalized, first-domain-as-primary)")
    print("=" * 70)
    counts_first, n_first = categorize(train_tasks, "first")
    print(f"\nTotal train tasks with Domain primary: {n_first}/{len(train_tasks)}")
    print(f"Unique normalized domains: {len(counts_first)}")
    print(f"\n{'rank':>4}  {'n':>4}  {'cum%':>6}  {'domain':<50}")
    cum = 0
    for i, (k, n) in enumerate(counts_first.most_common(), 1):
        cum += n
        pct = 100 * cum / n_first
        marker = " ★" if i == 10 else ""
        print(f"  {i:3d}.  {n:4d}  {pct:5.1f}%  {k}{marker}")

    print(f"\n--- Top-K + 'miscellaneous' cutoff analysis ---")
    for K in (7, 10, 12, 15, 20):
        cum = sum(n for _, n in counts_first.most_common(K - 1))
        pct = 100 * cum / n_first
        print(f"  K={K:2d}  top-{K-1} covers {pct:5.1f}%  ({n_first - cum} tasks in misc)")

    print("\n" + "=" * 70)
    print(" MULTI-LABEL VIEW (each listed Domain counted; gives 'all' distribution)")
    print("=" * 70)
    counts_all, n_all = categorize(train_tasks, "all")
    print(f"\nTotal label assignments: {n_all} across {len(train_tasks)} tasks "
          f"(avg {n_all/len(train_tasks):.2f}/task)")
    print(f"Unique normalized domains: {len(counts_all)}")
    print(f"\n{'rank':>4}  {'n':>4}  {'cum%':>6}  {'domain':<50}")
    cum = 0
    for i, (k, n) in enumerate(counts_all.most_common(25), 1):
        cum += n
        pct = 100 * cum / n_all
        print(f"  {i:3d}.  {n:4d}  {pct:5.1f}%  {k}")

    # Also check test distribution
    print("\n" + "=" * 70)
    print(" TEST SET DOMAIN DISTRIBUTION (for eval routing)")
    print("=" * 70)
    test_counts, test_n = categorize(test_tasks, "first")
    print(f"\nTest unique normalized domains: {len(test_counts)}")
    cum = 0
    for i, (k, n) in enumerate(test_counts.most_common(), 1):
        cum += n
        pct = 100 * cum / test_n
        print(f"  {i:3d}.  {n:4d}  {pct:5.1f}%  {k}")

    print("\nDone.")


if __name__ == "__main__":
    main()
