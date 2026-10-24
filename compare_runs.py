"""
Extra checks on the classifier results. It does not call the model.
It only reads the result files that evaluate.py already saved.

It answers three questions:
  1. How sure is each accuracy number?           (95% confidence interval)
  2. Is one run really better than another?      (McNemar test on the same 200 comments)
  3. Do some test comments also appear in the instructions?
     Short comments like "اول" are repeated all over the channel, so a few test
     comments are nearly identical to a comment in the first 100. Accuracy is
     reported again without them.

Input:   sample_300_labeled.csv and the eval_results files listed in RUNS
Output:  eval_stats.txt

How to run (Windows CMD, inside the project folder):
    py compare_runs.py
"""

import csv
import math
import os
import re
import sys
from collections import Counter
from itertools import combinations

try:
    import classifier
except ImportError:
    sys.exit("classifier.py must be in the same folder as this file.")

LABELED_FILE = "sample_300_labeled.csv"
OUTPUT_FILE = "eval_stats.txt"
DEV_SIZE = 100
NAMES = list(classifier.LABELS)
NUMBER_TO_NAME = {str(i + 1): name for i, name in enumerate(NAMES)}

# (name shown in the report, the result files to look for, first one found is used)
RUNS = [
    ("8B short definitions", ["eval_results.csv"]),
    ("8B rules + examples", ["eval_results_qwen3-8b.csv", "eval_results_v2.csv"]),
    ("14B rules + examples", ["eval_results_qwen3-14b.csv"]),
]


def read_csv(name):
    with open(name, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def simplify(text):
    """Remove what does not change the meaning: spelling forms, emoji, stretched letters."""
    text = re.sub(r"[ً-ْـ]", "", text)                       # tashkeel and tatweel
    text = re.sub(r"[أإآ]", "ا", text).replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)               # "ههههه" -> "هه"
    text = re.sub(r"[\W_]", " ", text)                       # punctuation and emoji
    return re.sub(r"\s+", " ", text).strip().lower()


def wilson(correct, total):
    """95% confidence interval for a share (Wilson score interval), in percent."""
    z = 1.959964
    share = correct / total
    middle = (share + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, 100 * (middle - half)), min(100.0, 100 * (middle + half))


def mcnemar(only_first, only_second):
    """Exact McNemar test: the chance of a split this uneven if both runs were equally good."""
    total = only_first + only_second
    if total == 0:
        return 1.0
    smaller = min(only_first, only_second)
    tail = sum(math.comb(total, k) for k in range(smaller + 1)) / 2 ** total
    return min(1.0, 2 * tail)


def p_text(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def f1_scores(pairs):
    """pairs = [(my label, model label), ...]. Returns macro F1 and the F1 of 'request'."""
    scores = {}
    for name in NAMES:
        hit = sum(1 for mine, model in pairs if mine == name and model == name)
        said = sum(1 for _, model in pairs if model == name)
        real = sum(1 for mine, _ in pairs if mine == name)
        if not real:
            continue
        precision = hit / said if said else 0.0
        recall = hit / real
        scores[name] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return sum(scores.values()) / len(scores), scores.get("request", 0.0)


def describe(pairs):
    correct = sum(1 for mine, model in pairs if mine == model)
    low, high = wilson(correct, len(pairs))
    macro, request = f1_scores(pairs)
    return (f"{100 * correct / len(pairs):5.1f}%  ({correct:>3} of {len(pairs)})   "
            f"95% CI {low:4.1f} to {high:4.1f}   macro F1 {macro:.2f}   request F1 {request:.2f}")


def main():
    try:
        rows = read_csv(LABELED_FILE)
    except FileNotFoundError:
        sys.exit(f"{LABELED_FILE} was not found. Run this inside the project folder.")
    rows = [row for row in rows if (row.get("my_label") or "").strip() in NUMBER_TO_NAME]
    mine = {row["comment_id"]: NUMBER_TO_NAME[row["my_label"].strip()] for row in rows}
    dev, test = rows[:DEV_SIZE], rows[DEV_SIZE:]
    test_ids = [row["comment_id"] for row in test]

    runs = []       # (name, {comment_id: model label})
    for name, files in RUNS:
        found = next((f for f in files if os.path.exists(f)), None)
        if not found:
            continue
        labels = {row["comment_id"]: row["model_label"] for row in read_csv(found)}
        if all(comment_id in labels for comment_id in test_ids):
            runs.append((name, labels, found))
    if not runs:
        sys.exit("No eval_results file was found. Run evaluate.py first.")

    # Test comments that are nearly identical to a comment in the first 100,
    # or to a phrase quoted in the rules of the instructions.
    known = {simplify(row["text"]) for row in dev}
    rules = classifier.INSTRUCTIONS.split("أمثلة:")[0]
    known |= {simplify(phrase) for phrase in re.findall(r'"([^"]+)"', rules)}
    known.discard("")
    repeated = {row["comment_id"] for row in test if simplify(row["text"]) in known}
    fresh_ids = [comment_id for comment_id in test_ids if comment_id not in repeated]

    # The same text labeled by hand in two different ways.
    by_text = {}
    for row in rows:
        if simplify(row["text"]):
            by_text.setdefault(simplify(row["text"]), set()).add(mine[row["comment_id"]])
    conflicts = sum(1 for labels in by_text.values() if len(labels) > 1)

    top_label, top_count = Counter(mine[i] for i in test_ids).most_common(1)[0]
    low, high = wilson(top_count, len(test_ids))
    lines = []
    lines.append(f"Test comments: {len(test_ids)}  (the first {DEV_SIZE} labeled comments are not counted)")
    lines.append("")
    lines.append("1. Accuracy with a 95% confidence interval")
    lines.append(f"   {'baseline (always ' + top_label + ')':<24}{100 * top_count / len(test_ids):5.1f}%  "
                 f"({top_count:>3} of {len(test_ids)})   95% CI {low:4.1f} to {high:4.1f}")
    for name, labels, _ in runs:
        lines.append(f"   {name:<24}" + describe([(mine[i], labels[i]) for i in test_ids]))

    lines.append("")
    lines.append("2. Is the difference real? (McNemar exact test on the same comments)")
    for (name_a, labels_a, _), (name_b, labels_b, _) in combinations(runs, 2):
        only_a = sum(1 for i in test_ids if labels_a[i] == mine[i] != labels_b[i])
        only_b = sum(1 for i in test_ids if labels_b[i] == mine[i] != labels_a[i])
        total = len(test_ids)
        gain = (only_b - only_a) / total
        error = math.sqrt((only_a + only_b) - (only_b - only_a) ** 2 / total) / total
        lines.append(f"   {name_a}  ->  {name_b}")
        lines.append(f"      right only in the first: {only_a}, right only in the second: {only_b}")
        lines.append(f"      change: {100 * gain:+.1f} points  (95% CI {100 * (gain - 1.96 * error):+.1f} "
                     f"to {100 * (gain + 1.96 * error):+.1f}),  {p_text(mcnemar(only_a, only_b))}")

    lines.append("")
    lines.append("3. Test comments that are nearly identical to a comment in the first 100")
    lines.append(f"   or to a phrase quoted in the instructions: {len(repeated)} of {len(test_ids)}")
    lines.append(f"   Accuracy on the other {len(fresh_ids)} comments:")
    for name, labels, _ in runs:
        lines.append(f"   {name:<24}" + describe([(mine[i], labels[i]) for i in fresh_ids]))

    lines.append("")
    lines.append(f"4. Texts that I labeled by hand in two different ways (all {len(rows)} comments): {conflicts}")
    lines.append("")
    lines.append("Files used: " + ", ".join(found for _, _, found in runs))

    report = "\n".join(lines)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write(report + "\n")
    print(report)
    print()
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
