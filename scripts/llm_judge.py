#!/usr/bin/env python3
"""Ask a local LLM (Ollama) to rate Wikipedia edits, then measure its agreement with the reverted tag."""
# Usage: python3 scripts/llm_judge.py --top=50 --random=50 --out=results/constructed/wiki_judge.csv

import argparse
import csv
import html
import json
import os
import random
import re
import time
import urllib.parse
import urllib.request

API = "https://en.wikipedia.org/w/api.php"
OLLAMA = "http://localhost:11434/api/generate"
UA = "COMP8240-student-project/0.1 (Macquarie University coursework)"
MAX_CHARS = 1500   # keeps each prompt short enough for a 7B model on a laptop

PROMPT = """You are reviewing an edit to English Wikipedia.

Article: {title}
Edit summary: {comment}
Size change: {delta} bytes

Changes (- lines were removed, + lines were added, [-x-] marks removed words, {{+y+}} marks added words):
{diff}

Rate how likely this edit is to be harmful (vandalism, nonsense, unsourced or biased claims, or removing content without reason):
1 = clearly a good-faith improvement
2 = probably fine
3 = unclear
4 = probably harmful
5 = clearly vandalism or harmful

Give a one-sentence reason first, then the score.
Answer in JSON: {{"reason": "...", "score": 1}}"""


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fetch_diff(revid):
    query = {"action": "compare", "fromrev": revid, "torelative": "prev",
             "prop": "diff|ids", "format": "json", "formatversion": "2"}
    request = urllib.request.Request(API + "?" + urllib.parse.urlencode(query),
                                     headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
    except Exception as error:
        print(f"  no diff for {revid}: {error}")
        return ""
    time.sleep(1)
    body = data.get("compare", {}).get("body", "")
    if not body:
        print(f"  no diff for {revid}: {data.get('error', {}).get('info', 'empty')}")   # e.g. a new page
        return ""
    # The diff may run from this edit back to the previous one, so flip it to read forwards.
    backwards = str(data["compare"].get("fromrevid")) == str(revid)
    removed, added = (r"{+\1+}", r"[-\1-]") if backwards else (r"[-\1-]", r"{+\1+}")
    lines = []
    for side, cell in re.findall(r'<td class="diff-(added|deleted)line[^"]*">(.*?)</td>', body, re.S):
        cell = re.sub(r"<del[^>]*>(.*?)</del>", removed, cell, flags=re.S)
        cell = re.sub(r"<ins[^>]*>(.*?)</ins>", added, cell, flags=re.S)
        text = html.unescape(re.sub(r"<[^>]+>", "", cell)).strip()
        if text:
            plus = (side == "added") != backwards
            lines.append(("+ " if plus else "- ") + text)
    return "\n".join(lines)[:MAX_CHARS]


def ask(model, prompt):
    payload = {"model": model, "prompt": prompt, "stream": False, "format": "json",
               "options": {"temperature": 0, "seed": 0}}   # fixed so reruns give the same answer
    request = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        reply = json.loads(json.load(response)["response"])
    score = int(reply.get("score", 0))
    return (score if 1 <= score <= 5 else ""), str(reply.get("reason", "")).strip()


def kappa(a, b):
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def summary(path):
    rows = [r for r in read_csv(path) if r["llm_score"]]
    llm = [int(int(r["llm_score"]) >= 4) for r in rows]   # 4 or 5 counts as harmful
    tag = [int(r["reverted"]) for r in rows]
    both = sum(l and t for l, t in zip(llm, tag))
    print(f"\n{len(rows)} edits rated")
    print(f"  harmful and reverted {both}, harmful only {sum(llm) - both}, "
          f"reverted only {sum(tag) - both}, neither {len(rows) - sum(llm) - sum(tag) + both}")
    print(f"  agreement {sum(l == t for l, t in zip(llm, tag)) / len(rows):.2f}, Cohen's kappa {kappa(llm, tag):.2f}")
    for group in ("top", "random"):
        g = [r for r in rows if r["group"] == group]
        if g:
            mean = sum(int(r["llm_score"]) for r in g) / len(g)
            harmful = sum(int(r["llm_score"]) >= 4 for r in g)
            reverted = sum(int(r["reverted"]) for r in g)
            print(f"  {group:6}: {len(g)} edits, mean LLM score {mean:.2f}, "
                  f"rated harmful {harmful}, reverted {reverted}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/constructed/wiki_edits_nobots_scores.csv")
    ap.add_argument("--info", default="data/constructed/wiki_edits_nobots_info.csv")
    ap.add_argument("--raw", default="data/constructed/wiki_edits_nobots.csv")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--top", type=int, default=50, help="edits OptIForest finds most unusual")
    ap.add_argument("--random", type=int, default=50, help="random edits for comparison")
    ap.add_argument("--out", default="results/constructed/wiki_judge.csv")
    args = ap.parse_args()

    scores = sorted(read_csv(args.scores), key=lambda r: -float(r["score"]))
    info = read_csv(args.info)
    with open(args.raw, newline="") as f:
        delta = [row[1] for row in csv.reader(f)]   # second column is the size change

    chosen = [(r, "top") for r in scores[:args.top]]
    rest = scores[args.top:]
    chosen += [(r, "random") for r in random.Random(0).sample(rest, args.random)]

    # Rows already in the output are skipped, so a stopped run can carry on where it left off.
    done = {r["row"] for r in read_csv(args.out)} if os.path.exists(args.out) else set()
    fields = ["row", "revid", "title", "group", "opt_score", "reverted", "llm_score", "reason", "seconds"]
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for n, (r, group) in enumerate(chosen, 1):
            if r["row"] in done:
                continue
            edit = info[int(r["row"])]
            t0 = time.time()
            diff = fetch_diff(edit["revid"]) or "(diff not available)"
            prompt = PROMPT.format(title=edit["title"], comment=edit["comment"] or "(none)",
                                   delta=int(float(delta[int(r["row"])])), diff=diff)
            score, reason = ask(args.model, prompt)
            seconds = round(time.time() - t0, 1)
            w.writerow({"row": r["row"], "revid": edit["revid"], "title": edit["title"], "group": group,
                        "opt_score": r["score"], "reverted": r["label"], "llm_score": score,
                        "reason": reason, "seconds": seconds})
            f.flush()
            print(f"{n}/{len(chosen)} {group:6} score {score}  reverted {r['label']}  {seconds}s  {edit['title'][:40]}")
    summary(args.out)


if __name__ == "__main__":
    main()
