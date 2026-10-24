"""
Week 2, final step: label the channel's comments with the local model.

Input:   comments.csv            (made by fetch_comments.py)
Output:  labeled_comments.csv    (every comment with its label)

Good to know:
- It saves as it goes. Stop it any time with Ctrl+C, then run the same
  command again and it continues from where it stopped.
- Comments are handled in random order. So if you stop early, what is
  already labeled is a fair random sample of the whole channel.
- A comment text that was already seen (like "اول") is not sent to the
  model again. It reuses the first answer.

How to run (Windows CMD, inside the project folder):
    py classify_all.py
"""

import csv
import os
import random
import sys
import time
from collections import Counter

import requests

try:
    import classifier
except ImportError:
    sys.exit("classifier.py must be in the same folder as this file.")

INPUT_FILE = "comments.csv"
OUTPUT_FILE = "labeled_comments.csv"
COLUMNS = ["comment_id", "video_id", "text", "likes", "published_at", "label"]
SEED = 7             # fixed order, so a second run continues the same order
REPORT_EVERY = 200   # print progress after this many comments


def read_comments():
    try:
        with open(INPUT_FILE, newline="", encoding="utf-8-sig") as file:
            comments = [row for row in csv.DictReader(file) if row["text"].strip()]
    except FileNotFoundError:
        sys.exit(f"{INPUT_FILE} was not found. Run this inside the yt-project folder.")
    random.seed(SEED)
    random.shuffle(comments)
    return comments


def read_finished():
    """Comments labeled in an earlier run: their IDs, and the answer for each text."""
    done_ids, answers = set(), {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                done_ids.add(row["comment_id"])
                answers[classifier.clean(row["text"])] = row["label"]
    return done_ids, answers


def format_time(seconds):
    hours, minutes = int(seconds // 3600), int(seconds % 3600 // 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m"


def print_summary():
    with open(OUTPUT_FILE, newline="", encoding="utf-8-sig") as file:
        counts = Counter(row["label"] for row in csv.DictReader(file))
    total = sum(counts.values())
    print()
    print(f"Labeled comments so far: {total}")
    for name in classifier.LABELS:
        share = 100 * counts[name] / total if total else 0
        print(f"  {name:<10}{counts[name]:>8}  {share:5.1f}%")


def main():
    comments = read_comments()
    done_ids, answers = read_finished()
    todo = [c for c in comments if c["comment_id"] not in done_ids]
    print(f"Model: {classifier.MODEL}")
    print(f"Comments: {len(comments)}. Already labeled: {len(done_ids)}. Left: {len(todo)}.")
    if not todo:
        print_summary()
        return
    print("Stop any time with Ctrl+C. Your progress is saved.")

    is_new_file = not os.path.exists(OUTPUT_FILE)
    started = time.time()
    finished_now = 0
    try:
        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=COLUMNS, extrasaction="ignore")
            if is_new_file:
                writer.writeheader()
            for comment in todo:
                key = classifier.clean(comment["text"])
                if key not in answers:
                    answers[key] = classifier.classify(comment["text"])
                writer.writerow(dict(comment, label=answers[key]))
                finished_now += 1
                if finished_now % REPORT_EVERY == 0:
                    file.flush()
                    speed = finished_now / (time.time() - started)
                    left = (len(todo) - finished_now) / speed
                    print(f"{len(done_ids) + finished_now} of {len(comments)} done. "
                          f"{speed:.1f} per second. About {format_time(left)} left.")
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue.")
    except requests.exceptions.ConnectionError:
        print("\nOllama is not running. Open the Ollama app, then run this again.")
    except requests.exceptions.HTTPError as error:
        print(f"\nOllama returned an error: {error.response.text[:300]}")
    else:
        print("\nAll comments are labeled.")
    print_summary()


if __name__ == "__main__":
    main()
