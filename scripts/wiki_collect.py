#!/usr/bin/env python3
"""Collect recent Wikipedia edits and build an anomaly detection CSV (label: edit later reverted)."""
# Usage: python3 scripts/wiki_collect.py --target=20000 --name=wiki_edits_nobots
# Add --include-bots to keep edits marked as bot edits, as the earlier collection did.

import argparse
import csv
import datetime as dt
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

API = "https://en.wikipedia.org/w/api.php"
UA = "COMP8240-student-project/0.1 (Macquarie University coursework)"
LABEL_TAG = "mw-reverted"   # Wikipedia's own tag for an edit that was later undone
# Revert tags are left out of tag_count so the label cannot leak into the features.
REVERT_TAGS = {"mw-reverted", "mw-undo", "mw-rollback", "mw-manual-revert"}
COLUMNS = ["size", "delta", "abs_delta", "is_minor", "is_unregistered",
           "comment_length", "tag_count"]
STAMP = "%Y-%m-%dT%H:%M:%SZ"

# Ways an edit summary can name an editor. Each is replaced before the summary is saved.
USER_PATTERNS = [
    (r"\[\[[^\]|]*User:[^\]|]*/[^\]|]*\|([^\]]*)\]\]", r"\1"),   # a tool's link: keep the tool's name
    (r"\[\[[^\]|]*User talk:[^\]]*\]\]", "talk"),
    (r"\[\[[^\]|]*(?:Special:Contrib(?:utions|s)/|User:)[^\]]*\]\]", "[user]"),
    (r"(to (?:(?:last|previous) )?(?:revision|version) (?:\d+ )?by )[^:\[\]()]+?(?=\.\s|\.$|\s*:|\s*\(|$)", r"\1[user]"),
    (r"(by )[^\[\]():]+?(?= \(talk\))", r"\1[user]"),
    (r"~\d{4}-\d+-\d+", "[user]"),                   # temporary accounts
    (r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "[user]"),      # IP addresses
]


def call(query, pause):
    url = API + "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.load(response)
            time.sleep(pause)
            return data
        except urllib.error.HTTPError as error:
            if error.code != 429:
                raise
            wait = 10 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s")
            time.sleep(wait)
    raise SystemExit("gave up after five rate-limit retries")


def collect_chunk(start, end, wanted, pause, include_bots):
    edits = []
    carry_on = None
    while len(edits) < wanted:
        query = {
            "action": "query", "format": "json", "formatversion": "2",
            "list": "recentchanges", "rcnamespace": "0", "rctype": "edit",
            "rclimit": "500", "rcdir": "older",
            "rcstart": start.strftime(STAMP), "rcend": end.strftime(STAMP),
            "rcprop": "title|ids|timestamp|user|comment|sizes|tags|flags",
        }
        if not include_bots:
            query["rcshow"] = "!bot"   # leave out edits marked as bot edits
        if carry_on:
            query["rccontinue"] = carry_on
        data = call(query, pause)
        batch = data.get("query", {}).get("recentchanges", [])
        edits.extend(batch)
        carry_on = data.get("continue", {}).get("rccontinue")
        if not carry_on or not batch:
            break
    return edits[:wanted]


def collect(target, lag_days, window_days, chunks, pause, include_bots):
    now = dt.datetime.now(dt.timezone.utc)
    newest = now - dt.timedelta(days=lag_days)
    oldest = now - dt.timedelta(days=window_days)
    step = (newest - oldest) / chunks
    print(f"collecting {target:,} edits in {chunks} chunks between "
          f"{oldest.strftime(STAMP)} and {newest.strftime(STAMP)}")

    edits = []
    for i in range(chunks):
        start = newest - i * step
        edits.extend(collect_chunk(start, start - step, target // chunks, pause, include_bots))
        print(f"  chunk {i + 1}/{chunks}: {len(edits):,} edits")
    return edits


def build_rows(edits):
    rows = []
    for edit in edits:
        tags = edit.get("tags", [])
        delta = edit.get("newlen", 0) - edit.get("oldlen", 0)
        unregistered = edit.get("temp") is True or edit.get("anon") is True
        rows.append([
            float(edit.get("newlen", 0)),
            float(delta),
            float(abs(delta)),
            1.0 if edit.get("minor") is True else 0.0,
            1.0 if unregistered else 0.0,
            float(len(edit.get("comment", "") or "")),
            float(len([t for t in tags if t not in REVERT_TAGS])),
            1.0 if LABEL_TAG in tags else 0.0,
        ])
    return np.array(rows)


def write_info(edits, path):
    # One row per edit, in the same order as the data CSV. Usernames are not kept.
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["row", "revid", "title", "timestamp", "comment", "reverted"])
        for i, edit in enumerate(edits):
            w.writerow([i, edit.get("revid"), edit.get("title", ""), edit.get("timestamp", ""),
                        hide_users(edit.get("comment")), int(LABEL_TAG in edit.get("tags", []))])


def hide_users(text):
    text = text or ""
    for pattern, replacement in USER_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def standardise(X):
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd > 0, sd, 1.0)


def report(X, y):
    norms = np.linalg.norm(X, axis=1)
    share = (X ** 2).sum(axis=0)
    share = share / share.sum()
    top = np.argsort(-share)[:3]
    print(f"\n{len(X):,} rows, {X.shape[1]} features, "
          f"{int(y.sum())} anomalies ({100 * y.mean():.3f}%)")
    print(f"mean row norm {norms.mean():,.1f}, divisor rule suggests {norms.mean() / 6:,.0f}")
    print("top 3 columns: " + ", ".join(
        f"{COLUMNS[j]} {100 * share[j]:.1f}%" for j in top))
    print(f"minor {100 * X[:, 3].mean():.1f}%, unregistered {100 * X[:, 4].mean():.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=20000)
    ap.add_argument("--lag-days", type=int, default=3)
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--pause", type=float, default=1.5)
    ap.add_argument("--outdir", default="data/constructed")
    ap.add_argument("--name", default="wiki_edits_nobots")
    ap.add_argument("--include-bots", action="store_true", help="keep edits marked as bot edits")
    args = ap.parse_args()

    edits = collect(args.target, args.lag_days, args.window_days, args.chunks, args.pause,
                    args.include_bots)
    data = build_rows(edits)
    X, y = data[:, :-1], data[:, -1]
    report(X, y)

    base = os.path.join(args.outdir, args.name)
    np.savetxt(base + ".csv", data, delimiter=",", fmt="%.6f")
    std = np.column_stack([standardise(X), y])
    np.savetxt(base + "_std.csv", std, delimiter=",", fmt="%.6f")
    write_info(edits, base + "_info.csv")

    sample = []
    for edit in edits[::max(1, len(edits) // 50)][:50]:
        record = {k: v for k, v in edit.items() if k != "user"}
        record["comment"] = hide_users(edit.get("comment"))
        sample.append(record)
    with open(base + "_sample.json", "w") as handle:
        json.dump(sample, handle, indent=2)

    print(f"wrote {base}.csv, {base}_std.csv, {base}_info.csv and {base}_sample.json")
    print("columns: " + ", ".join(COLUMNS) + ", label")


if __name__ == "__main__":
    main()
