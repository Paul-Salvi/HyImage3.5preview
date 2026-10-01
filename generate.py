#!/usr/bin/env python3
"""
generate.py - batch image generation / editing for Hy-Image-3.5-preview on GMI Cloud.

Zero third-party dependencies (stdlib only). Python 3.9+.

SETUP
    export GMI_API_KEY="your-key"          # never hard-code or paste the key anywhere

USAGE
    # single prompt, 4 variants, draft resolution
    python generate.py --name s01_cover --prompt "..." -n 4 --preset draft

    # final-quality render of one scene with a fixed seed (reproducible)
    python generate.py --name s01_cover --prompt "..." --preset final --seed 12345

    # edit mode: pass reference image URLs (must be public, <20MB each, max 5)
    python generate.py --name s01_fix --prompt "Keep everything identical; change ..." \
        --ref https://cdn.example.com/s01_cover.png

    # many scenes at once from a JSON file
    python generate.py --jobs scenes.json --workers 4

    # see exactly what would be sent, no network calls
    python generate.py --jobs scenes.json --dry-run

JOBS FILE (scenes.json) - a list of objects; only "prompt" is required:
    [
      {"name": "s01_cover", "prompt": "...", "n": 4, "size": "2560x1440"},
      {"name": "s02_hormuz", "prompt": "...", "seed": 777, "refs": ["https://.../anchor.png"]}
    ]
Per-job keys: name, prompt, n, size, seed, refs. Anything omitted falls back to CLI flags.

OUTPUT
    out/<name>/<name>_v01_s<seed>.png ...
    out/runs.jsonl   one JSON line per generation (prompt, seed, size, refs, request id,
                     timing, file) - use it as your workflow evidence for the submission.
"""

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = os.environ.get("GMI_BASE_URL", "https://console.gmicloud.ai").rstrip("/")
SUBMIT_PATH = "/api/v1/ie/requestqueue/apikey/requests"
MODEL = "hy-image-v3.5-preview"

# Exact size enum from the API docs. "" = auto (model picks, <= 2K).
VALID_SIZES = {
    "", "1024x1024", "1536x1536", "2048x2048", "1920x1080", "1080x1920",
    "1536x1152", "1152x1536", "2560x1440", "1440x2560",
    "3840x2160", "2160x3840", "4096x4096",
}
PRESETS = {"draft": "2560x1440", "final": "3840x2160"}  # 16:9; 4K is billed at the higher rate

READ_TIMEOUT = 180          # docs: set at least 2 minutes, generation blocks
MAX_RETRIES = 3
TERMINAL = {"success", "failed", "cancelled"}

_log_lock = threading.Lock()
_print_lock = threading.Lock()


def say(msg):
    with _print_lock:
        print(msg, flush=True)


def http_json(method, url, api_key, body=None, timeout=READ_TIMEOUT):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download(url, dest, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "hy-image-batch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        blob = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return len(blob)


def build_body(prompt, size, seed, refs, image_format):
    payload = {"prompt": prompt}
    if size:
        payload["size"] = size
    if seed:
        payload["seed"] = seed
    if refs:
        payload["image"] = refs[0] if (image_format == "string" and len(refs) == 1) else refs
    return {"model": MODEL, "payload": payload}


def submit_with_retry(body, api_key, tag):
    """Submit and (if needed) poll. Returns the final response dict. Raises on hard failure."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = http_json("POST", BASE_URL + SUBMIT_PATH, api_key, body)
            rid = resp.get("request_id")
            # Docs say the call is synchronous, but poll defensively if not terminal.
            polls = 0
            while resp.get("status") not in TERMINAL and rid and polls < 60:
                time.sleep(5)
                resp = http_json("GET", f"{BASE_URL}{SUBMIT_PATH}/{rid}", api_key)
                polls += 1
            return resp
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            last_err = f"HTTP {e.code}: {detail}"
            if e.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(last_err)          # 4xx (bad key, bad params): do not retry
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < MAX_RETRIES:
            wait = (2 ** attempt) + random.random()
            say(f"[{tag}] attempt {attempt} failed ({last_err}); retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"gave up after {MAX_RETRIES} attempts: {last_err}")


def run_one(job, variant, args, api_key, out_dir, log_path):
    name = job["name"]
    tag = f"{name} v{variant:02d}"
    base_seed = job.get("seed") or 0
    seed = base_seed + (variant - 1) if base_seed else random.randint(1, 2**31 - 1)
    size = job.get("size", args.size)
    refs = job.get("refs") or args.ref or []
    prompt = job["prompt"]
    body = build_body(prompt, size, seed, refs, args.image_format)

    if args.dry_run:
        say(f"[{tag}] DRY RUN body:\n{json.dumps(body, indent=2)}")
        return None

    t0 = time.time()
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "name": name, "variant": variant, "model": MODEL,
        "prompt": prompt, "size": size or "auto", "seed": seed, "refs": refs,
    }
    try:
        resp = submit_with_retry(body, api_key, tag)
        record["request_id"] = resp.get("request_id")
        record["status"] = resp.get("status")
        outcome = resp.get("outcome") or {}
        if resp.get("status") != "success":
            record["error"] = outcome.get("error") or "no error detail"
            raise RuntimeError(f"status={resp.get('status')} error={record['error']}")
        media = outcome.get("media_urls") or []
        if not media:
            raise RuntimeError("success but no media_urls in response")
        m = media[0]
        record.update(url=m.get("url"), width=m.get("width"), height=m.get("height"))
        dest = out_dir / name / f"{name}_v{variant:02d}_s{seed}.png"
        nbytes = download(m["url"], dest)            # save immediately; URL expiry is undocumented
        record.update(file=str(dest), bytes=nbytes)
        say(f"[{tag}] ok  {m.get('width')}x{m.get('height')}  {time.time()-t0:.0f}s  -> {dest}")
    except Exception as e:  # noqa: BLE001 - record every failure, keep the batch going
        record["status"] = record.get("status") or "error"
        record["error"] = record.get("error") or str(e)
        say(f"[{tag}] FAILED: {e}")
    finally:
        record["elapsed_s"] = round(time.time() - t0, 1)
        with _log_lock:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_jobs(args):
    if args.jobs:
        raw = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            sys.exit("--jobs file must contain a JSON list")
        jobs = raw
    elif args.prompt:
        jobs = [{"name": args.name or "img", "prompt": args.prompt}]
    else:
        sys.exit("provide --prompt or --jobs (see --help)")
    for i, j in enumerate(jobs):
        if not j.get("prompt"):
            sys.exit(f"job {i} has no prompt")
        j.setdefault("name", f"job{i+1:02d}")
        j["n"] = int(j.get("n", args.n))
        if j.get("size") is not None and j["size"] not in VALID_SIZES:
            sys.exit(f"job '{j['name']}': size '{j['size']}' not in {sorted(VALID_SIZES)}")
        if len(j.get("refs") or []) > 5:
            sys.exit(f"job '{j['name']}': max 5 reference images")
    return jobs


def main():
    ap = argparse.ArgumentParser(description="Hy-Image-3.5-preview batch generator (GMI Cloud)")
    ap.add_argument("--prompt", help="prompt, or edit instruction when --ref is used")
    ap.add_argument("--name", help="output name for a single --prompt run")
    ap.add_argument("--jobs", help="JSON file with a list of jobs")
    ap.add_argument("-n", type=int, default=1, help="variants per job (default 1)")
    ap.add_argument("--size", default=None, help=f"exact size, one of: {', '.join(sorted(s for s in VALID_SIZES if s))}")
    ap.add_argument("--preset", choices=PRESETS, help="draft=2560x1440, final=3840x2160 (4K billed at $0.032)")
    ap.add_argument("--seed", type=int, default=0, help="fixed seed (variants use seed, seed+1, ...)")
    ap.add_argument("--ref", action="append", default=[], help="reference image URL (repeat, max 5)")
    ap.add_argument("--image-format", choices=["list", "string"], default="list",
                    help="how a single ref is sent in the 'image' field (default list; try 'string' if rejected)")
    ap.add_argument("--workers", type=int, default=3, help="parallel requests (default 3)")
    ap.add_argument("--out", default="out", help="output directory (default ./out)")
    ap.add_argument("--dry-run", action="store_true", help="print request bodies, send nothing")
    args = ap.parse_args()

    if args.preset:
        args.size = PRESETS[args.preset]
    if args.size is None:
        args.size = PRESETS["draft"]
    if args.size not in VALID_SIZES:
        sys.exit(f"invalid --size '{args.size}'")
    if len(args.ref) > 5:
        sys.exit("max 5 reference images")

    api_key = os.environ.get("GMI_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("set GMI_API_KEY in your environment first")

    jobs = load_jobs(args)
    if args.seed:
        for j in jobs:
            j.setdefault("seed", args.seed)

    out_dir = Path(args.out)
    log_path = out_dir / "runs.jsonl"
    tasks = [(j, v) for j in jobs for v in range(1, j["n"] + 1)]
    say(f"{len(tasks)} generation(s), {args.workers} parallel, model={MODEL}, out={out_dir}/")

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(run_one, j, v, args, api_key, out_dir, log_path) for j, v in tasks]
        for f in as_completed(futs):
            results.append(f.result())

    if not args.dry_run:
        ok = sum(1 for r in results if r and r.get("status") == "success")
        say(f"done: {ok}/{len(tasks)} succeeded. run log: {log_path}")
        sys.exit(0 if ok == len(tasks) else 1)


if __name__ == "__main__":
    main()
