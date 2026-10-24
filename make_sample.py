"""
Week 2: pick 300 random comments for you to label by hand.

Input:   comments.csv
Output:  sample_300.csv  (open label.html in a browser and choose this file to label it)

Each comment gets one number in the my_label column:
    1 = request     (طلب)
    2 = praise      (مدح)
    3 = criticism   (نقد)
    4 = question    (سؤال)
    5 = spam        (سبام)
    6 = other       (أخرى)

How to run (Windows CMD, inside the project folder):
    py make_sample.py
"""

import csv
import os
import random
import sys

SAMPLE_SIZE = 300
SEED = 42            # same seed = same 300 comments every run
OUTPUT_FILE = "sample_300.csv"


def main():
    if os.path.exists(OUTPUT_FILE):
        sys.exit(f"{OUTPUT_FILE} already exists. Delete it first if you want a new one.")
    try:
        with open("comments.csv", newline="", encoding="utf-8-sig") as file:
            comments = [row for row in csv.DictReader(file) if row["text"].strip()]
    except FileNotFoundError:
        sys.exit("comments.csv was not found. Run this inside the yt-project folder.")

    random.seed(SEED)
    sample = random.sample(comments, min(SAMPLE_SIZE, len(comments)))

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["comment_id", "text", "my_label"])
        for comment in sample:
            writer.writerow([comment["comment_id"], comment["text"], ""])
    print(f"Done. Open label.html in a browser and choose {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
