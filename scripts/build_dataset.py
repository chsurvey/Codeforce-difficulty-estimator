"""Parallel builder for Codeforces / CodeChef datasets.

Examples
--------
# 3000 Codeforces problems, 8 parallel workers, API auth
export CF_KEY=xxxx CF_SECRET=yyyy
python scripts/build_dataset.py --dataset codeforces --limit 3000 \
       --api_key $CF_KEY --api_secret $CF_SECRET --workers 8

# CodeChef from CSV directory (no API)
python scripts/build_dataset.py --dataset codechef --raw_dir ./raw_codechef
"""
from __future__ import annotations
import argparse, json, random, time, hashlib, concurrent.futures as cf
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
import threading
from tqdm import tqdm

# -------------------------- rate‑limiter ----------------------
class RateLimiter:
    """Token‑bucket rate limiter (capacity tokens / interval seconds)."""
    def __init__(self, capacity: int, interval: float):
        self.capacity = capacity
        self.interval = interval
        self.tokens = capacity
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        """Block until one token is available."""
        with self._lock:
            now = time.monotonic()
            dt = now - self.updated
            # replenish tokens
            self.tokens = min(self.capacity, self.tokens + dt * self.capacity / self.interval)
            if self.tokens < 1:
                sleep_for = (1 - self.tokens) * self.interval / self.capacity
                time.sleep(sleep_for)
                now = time.monotonic()
                dt = now - self.updated
                self.tokens = min(self.capacity, self.tokens + dt * self.capacity / self.interval)
            self.tokens -= 1
            self.updated = now

# ------------------------ helpers & const --------------------
RATING_BINS = [(0, 1400), (1400, 1800), (1800, 10000)]  # 3‑class labels
API_BASE = "https://codeforces.com/api"

# --------------------- Codeforces functions ------------------

def rating_to_label(rating: int) -> int:
    for i, (lo, hi) in enumerate(RATING_BINS):
        if lo <= rating < hi:
            return i
    return len(RATING_BINS) - 1

def cf_get(endpoint: str, rate: Optional[RateLimiter] = None, **params):
    if rate:
        rate.acquire()
    resp = requests.get(f"{API_BASE}/{endpoint}", params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK":
        raise RuntimeError(data)
    return data["result"]

def get_problemset() -> List[Dict[str, Any]]:
    return [p for p in cf_get("problemset.problems") ["problems"] if "rating" in p]

def fetch_solution(contest_id: int, index: str, api_key: str, api_secret: str,
                   rate: RateLimiter) -> str:
    """Return first accepted submission source code (may be empty)."""
    ts = int(time.time())
    rnd = random.randint(1, 1_000_000)
    method = "contest.status"
    params = f"contestId={contest_id}&from=1&count=50&apiKey={api_key}&time={ts}&rand={rnd}"
    sig = hashlib.sha512(f"{rnd}/{method}?{params}#{api_secret}".encode()).hexdigest()
    full = {"contestId": contest_id, "from": 1, "count": 50,
            "apiKey": api_key, "time": ts, "apiSig": f"{rnd}{sig}"}
    try:
        subs = cf_get(method, rate, **full)
        for s in subs:
            if s.get("verdict") == "OK" and s.get("problem", {}).get("index") == index:
                return s.get("programmingLanguage", "") + "\n" + s.get("source", "")
    except Exception:
        pass
    return ""

def _solution_worker(arg: Tuple[int, str, str, str, RateLimiter]):
    cid, idx, key, sec, rate = arg
    return (f"{cid}{idx}", fetch_solution(cid, idx, key, sec, rate))

def build_codeforces(out_path: Path, api_key: Optional[str], api_secret: Optional[str],
                     limit: int, workers: int):
    problems = get_problemset()
    random.shuffle(problems)
    problems = problems[:limit]

    samples: Dict[str, Dict[str, Any]] = {}
    for p in problems:
        pid = f"{p['contestId']}{p['index']}"
        samples[pid] = {
            "problem_id": pid,
            "statement": p["name"],
            "code": "",  # to be filled later
            "features": [p.get("timeLimit", 2000), p.get("memoryLimit", 256), 0, 0],
            "label": rating_to_label(p["rating"]),
        }

    if api_key and api_secret:
        print(f"Fetching accepted codes with {workers} threads (rate‑limited)…")
        rate = RateLimiter(60, 5)  # 60 calls / 5 s per API spec
        tasks = [(p["contestId"], p["index"], api_key, api_secret, rate) for p in problems]
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            for pid, code in tqdm(pool.map(_solution_worker, tasks), total=len(tasks)):
                samples[pid]["code"] = code
    else:
        print("API key not provided → skipping code downloads.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples.values():
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Codeforces | saved {len(samples)} problems → {out_path}")

# ----------------------------- CodeChef ----------------------
def build_codechef(raw_dir: Path, out_path: Path):
    """Expecting pre‑downloaded Kaggle dataset CSVs in raw_dir."""
    import pandas as pd
    df = pd.read_csv(raw_dir / "codechef_problems.csv")  # hypothetical file
    samples = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        rating = row["difficulty"]
        label = rating_to_label(rating)
        features = [row["timelimit"], row["memlimit"], 0, row.get("category_id", 0)]
        samples.append({
            "problem_id": row["code"],
            "statement": row["title"],
            "code": row.get("solution", ""),
            "features": features,
            "label": label,
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"CodeChef | saved {len(samples)} samples -> {out_path}")

# ----------------------------- CLI ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["codeforces", "codechef"], required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--raw_dir", type=Path)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--api_key")
    parser.add_argument("--api_secret")
    parser.add_argument("--workers", type=int, default=8, help="parallel threads (CF)")
    args = parser.parse_args()

    out = args.out or Path("data") / f"{args.dataset}.jsonl"
    if args.dataset == "codeforces":
        build_codeforces(out, args.api_key, args.api_secret, args.limit, args.workers)
    else:
        assert args.raw_dir, "--raw_dir required for CodeChef"
        build_codechef(args.raw_dir, out)
