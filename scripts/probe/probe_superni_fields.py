"""Probe SuperNI v2 metadata fields to decide K-clustering strategy.

Fetches train_tasks.txt from allenai/natural-instructions (official), samples
~150 random tasks, aggregates Categories / Domains / Reasoning / URL source
distributions. No downloads are cached here — this is a one-shot read-only
probe to inform the LoRA track's routing-label decision.

Runs in ~3 min over public GitHub raw URLs (no auth needed).
"""
import json
import random
import urllib.request
import urllib.error
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://raw.githubusercontent.com/allenai/natural-instructions/master"
TRAIN_TASKS_URL = f"{BASE}/splits/default/train_tasks.txt"
TEST_TASKS_URL = f"{BASE}/splits/default/test_tasks.txt"
SAMPLE_N = 150
SEED = 42


def fetch_url(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "specdrop-probe/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return f.read()


def fetch_task_list(url):
    raw = fetch_url(url).decode()
    return [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.startswith("#")]


def fetch_task_json(task_id):
    try:
        raw = fetch_url(f"{BASE}/tasks/{task_id}.json")
        return task_id, json.loads(raw)
    except Exception as e:
        return task_id, None


def summarize(tasks):
    cats = Counter()
    doms = Counter()
    reasons = Counter()
    input_lang = Counter()
    output_lang = Counter()
    source = Counter()
    field_presence = Counter()

    for task_id, d in tasks:
        if d is None:
            continue
        for k in d:
            field_presence[k] += 1
        for c in d.get("Categories", []) or []:
            cats[c] += 1
        for dm in d.get("Domains", []) or []:
            doms[dm] += 1
        for r in d.get("Reasoning", []) or []:
            reasons[r] += 1
        for il in d.get("Input_language", []) or []:
            input_lang[il] += 1
        for ol in d.get("Output_language", []) or []:
            output_lang[ol] += 1
        for src in d.get("Source", []) or []:
            source[src] += 1

    return {
        "cats": cats, "doms": doms, "reasons": reasons,
        "input_lang": input_lang, "output_lang": output_lang, "source": source,
        "field_presence": field_presence,
    }


def print_dist(name, counter, n_sampled, top=30):
    total_assignments = sum(counter.values())
    unique = len(counter)
    print(f"\n=== {name} | unique={unique}  assignments={total_assignments} "
          f"(avg {total_assignments/max(n_sampled,1):.2f}/task)")
    cum = 0
    for i, (k, n) in enumerate(counter.most_common(top), 1):
        cum += n
        pct = 100 * cum / total_assignments if total_assignments else 0
        print(f"  {i:3d}. {n:4d}  {k}   (cum {pct:5.1f}%)")
    if unique > top:
        print(f"  ... {unique - top} more (tail)")


def main():
    print(f"Fetching train+test task lists from {BASE} ...")
    train_tasks = fetch_task_list(TRAIN_TASKS_URL)
    test_tasks = fetch_task_list(TEST_TASKS_URL)
    print(f"  TRAIN: {len(train_tasks)} tasks")
    print(f"  TEST:  {len(test_tasks)} tasks")
    print(f"  Total in splits/default/: {len(train_tasks)+len(test_tasks)}")

    random.seed(SEED)
    sample = random.sample(train_tasks, min(SAMPLE_N, len(train_tasks)))
    print(f"\nFetching {len(sample)} random train task JSONs (threaded)...")

    tasks = []
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(fetch_task_json, t) for t in sample]
        for i, fut in enumerate(as_completed(futures), 1):
            tasks.append(fut.result())
            if i % 20 == 0:
                print(f"  {i}/{len(sample)} fetched")

    ok = sum(1 for _, d in tasks if d is not None)
    print(f"  Parsed {ok}/{len(sample)} tasks successfully")

    s = summarize(tasks)

    print(f"\n=== Field presence (which keys appear at all) ===")
    for k, n in s["field_presence"].most_common():
        print(f"  {n:4d}/{ok}  {k}")

    print_dist("Categories (TASK FORMAT)", s["cats"], ok)
    print_dist("Domains (SEMANTIC CONTENT)", s["doms"], ok)
    print_dist("Reasoning (SKILL TYPE)", s["reasons"], ok)
    print_dist("Source (PROVENANCE)", s["source"], ok, top=20)
    print_dist("Input_language", s["input_lang"], ok, top=15)


if __name__ == "__main__":
    main()
