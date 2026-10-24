"""
Week 2: measure how good a local model is.

It compares YOUR labels with the MODEL's labels.

The first 100 labeled comments were used to write the instructions (classifier.py),
so they are NOT counted. Accuracy is measured only on the other 200 comments,
which the model has never seen. This is the same idea as a train/test split.

Input:   sample_300_labeled.csv        (saved from label.html)
         eval_results.csv              (optional: the first run, to show before and after)
Output:  eval_results_<model>.csv      (each test comment: your label, model label)
         eval_summary_<model>.txt      (the numbers, ready to copy)

How to run (Windows CMD, inside the project folder):
    py evaluate.py                 uses the model written in classifier.py
    py evaluate.py qwen3:14b       tries another model (download it first: ollama pull qwen3:14b)
"""

import csv
import os
import sys
import time
from collections import Counter

import requests

try:
    import classifier
except ImportError:
    sys.exit("classifier.py must be in the same folder as this file.")

LABELED_FILE = "sample_300_labeled.csv"
OLD_RESULTS_FILE = "eval_results.csv"
ALL_COMMENTS_FILE = "comments.csv"
DEV_SIZE = 100       # the first 100 comments were used to write the instructions
NAMES = list(classifier.LABELS)                        # request, praise, ...
NUMBER_TO_NAME = {str(i + 1): name for i, name in enumerate(NAMES)}   # "1" -> request


def read_test_comments():
    try:
        with open(LABELED_FILE, newline="", encoding="utf-8-sig") as file:
            rows = list(csv.DictReader(file))
    except FileNotFoundError:
        sys.exit(f"{LABELED_FILE} was not found. Move it from Downloads into this folder.")
    test = []
    for row in rows[DEV_SIZE:]:
        number = (row.get("my_label") or "").strip()
        if number in NUMBER_TO_NAME and row.get("text", "").strip():
            test.append({"comment_id": row["comment_id"], "text": row["text"],
                         "mine": NUMBER_TO_NAME[number]})
    if not test:
        sys.exit("No labeled comments were found after the first 100.")
    return test


def first_run_accuracy(test):
    """Accuracy of the very first run on the same test comments, if that file is here."""
    if not os.path.exists(OLD_RESULTS_FILE):
        return None
    with open(OLD_RESULTS_FILE, newline="", encoding="utf-8-sig") as file:
        old = {row["comment_id"]: row["model_label"] for row in csv.DictReader(file)}
    if any(r["comment_id"] not in old for r in test):
        return None
    return percent(sum(1 for r in test if old[r["comment_id"]] == r["mine"]), len(test))


def count_all_comments():
    if not os.path.exists(ALL_COMMENTS_FILE):
        return None
    with open(ALL_COMMENTS_FILE, newline="", encoding="utf-8-sig") as file:
        return sum(1 for _ in csv.DictReader(file))


def percent(part, whole):
    return 100 * part / whole if whole else 0.0


def ask_model(text):
    try:
        return classifier.classify(text)
    except requests.exceptions.ConnectionError:
        sys.exit("Ollama is not running. Open the Ollama app, then run this again.")
    except requests.exceptions.HTTPError as error:
        message = error.response.text[:300]
        if error.response.status_code == 404:
            message += f"\nDownload the model first:  ollama pull {classifier.MODEL}"
        sys.exit(f"Ollama returned an error: {message}")


def build_summary(rows, first_run, speed):
    total = len(rows)
    correct = sum(1 for r in rows if r["mine"] == r["model"])
    mine_counts = Counter(r["mine"] for r in rows)
    top_label, top_count = mine_counts.most_common(1)[0]

    lines = []
    lines.append(f"Model: {classifier.MODEL}")
    lines.append(f"Instructions: {classifier.VERSION}")
    lines.append(f"Test comments: {total}  (the first {DEV_SIZE} are not counted)")
    if first_run is not None:
        lines.append(f"First run: {first_run:.1f}%  (qwen3:8b, first instructions, same comments)")
    lines.append(f"Accuracy now: {percent(correct, total):.1f}%  ({correct} of {total})")
    lines.append(f"Baseline: {percent(top_count, total):.1f}%  (always answering '{top_label}')")
    speed_line = f"Speed: {speed:.1f} comments per second"
    all_comments = count_all_comments()
    if all_comments and speed > 0:
        hours = all_comments / speed / 3600
        speed_line += f"  (all {all_comments} comments would take about {hours:.1f} hours)"
    lines.append(speed_line)
    lines.append("")
    lines.append(f"{'label':<11}{'yours':>6}{'precision':>11}{'recall':>8}{'F1':>7}")

    f1_values = []
    for name in NAMES:
        true_positive = sum(1 for r in rows if r["mine"] == name and r["model"] == name)
        model_said = sum(1 for r in rows if r["model"] == name)
        you_said = mine_counts[name]
        precision = true_positive / model_said if model_said else 0.0
        recall = true_positive / you_said if you_said else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if you_said:
            f1_values.append(f1)
        lines.append(f"{name:<11}{you_said:>6}{precision:>11.2f}{recall:>8.2f}{f1:>7.2f}")
    lines.append(f"Macro F1: {sum(f1_values) / len(f1_values):.2f}")

    lines.append("")
    lines.append("Confusion table (rows = your label, columns = model label):")
    lines.append(f"{'':<11}" + "".join(f"{name[:8]:>9}" for name in NAMES))
    for mine in NAMES:
        counts = [sum(1 for r in rows if r["mine"] == mine and r["model"] == model)
                  for model in NAMES]
        lines.append(f"{mine:<11}" + "".join(f"{c:>9}" for c in counts))
    return "\n".join(lines)


def main():
    if len(sys.argv) > 1:
        classifier.MODEL = sys.argv[1]      # try another model without editing any file
    tag = classifier.MODEL.replace(":", "-").replace("/", "-")

    rows = read_test_comments()
    first_run = first_run_accuracy(rows)
    print(f"Model: {classifier.MODEL}. Loading it (the first time can take a minute)...")
    ask_model("اول")                        # warm-up, so loading time is not counted in the speed

    print(f"Testing on {len(rows)} comments...")
    started = time.time()
    for number, row in enumerate(rows, start=1):
        row["model"] = ask_model(row["text"])
        print(f"[{number}/{len(rows)}]")
    speed = len(rows) / max(time.time() - started, 0.001)

    with open(f"eval_results_{tag}.csv", "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["comment_id", "text", "my_label", "model_label", "correct"])
        for row in rows:
            writer.writerow([row["comment_id"], row["text"], row["mine"], row["model"],
                             "yes" if row["mine"] == row["model"] else "no"])

    summary = build_summary(rows, first_run, speed)
    with open(f"eval_summary_{tag}.txt", "w", encoding="utf-8") as file:
        file.write(summary + "\n")
    print()
    print(summary)
    print()
    print(f"Saved eval_results_{tag}.csv and eval_summary_{tag}.txt")


if __name__ == "__main__":
    main()
