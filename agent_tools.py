"""
The tools the agent can use. Each tool is a normal Python function that
answers one kind of question from the labeled comments.

The model never sees the 127,000 comments. It only chooses a tool and its
arguments, then reads the small result the tool returns.
"""

import csv
import os
import re
from collections import Counter

LABELS = ["request", "praise", "criticism", "question", "spam", "other"]
ARABIC_LABELS = {"طلب": "request", "مدح": "praise", "نقد": "criticism",
                 "سؤال": "question", "سبام": "spam", "أخرى": "other", "اخرى": "other"}

# Words that follow "مود" but are not part of a mod's name.
# They are written the way normalize() leaves them (for example "على" becomes "علي").
NOT_A_MOD = {"عن", "في", "من", "او", "الي", "اللي", "مود", "جديد", "ثاني", "يا", "و",
             "لايك", "مثل", "زي", "علي", "حق", "مع", "هذا", "ذا", "كل", "بس", "ولا",
             "جزء", "الجزء", "ضد", "تكفي", "تكفا", "تكفه", "بليز", "لو", "سمحت"}
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

comments = []    # every labeled comment, loaded once by load()
videos = {}      # video_id -> title


def normalize(text):
    """Make Arabic text easier to match: same letter forms, no stretched letters."""
    text = re.sub(r"[ً-ْـ]", "", text)                  # tashkeel and tatweel
    text = re.sub(r"[أإآ]", "ا", text).replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"(.)\1{2,}", r"\1", text)                           # "ههههه" -> "ه"
    return re.sub(r"\s+", " ", text).strip().lower()


def load(comments_file="labeled_comments.csv", videos_file="videos.csv"):
    """Read the data files into memory. Call this once when the program starts."""
    csv.field_size_limit(10 ** 8)
    comments.clear()
    videos.clear()
    if os.path.exists(videos_file):
        with open(videos_file, newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                videos[row["video_id"]] = row["title"]
    with open(comments_file, newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            title = videos.get(row["video_id"], row["video_id"])
            comments.append({
                "text": row["text"],
                "norm": normalize(row["text"]),
                "label": row["label"],
                "likes": int(row["likes"] or 0),
                "year": int(row["published_at"][:4]) if row["published_at"][:4].isdigit() else 0,
                "video_id": row["video_id"],
                "video": title,
                "video_norm": normalize(title),
            })
    return len(comments)


# ----------------------------- helpers -----------------------------

def clean_label(label):
    """Accept English or Arabic label names. Anything else means 'no filter'."""
    if not label:
        return None
    label = str(label).strip().lower()
    label = ARABIC_LABELS.get(label, label)
    return label if label in LABELS else None


def clean_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def select(label=None, year=None, video=None):
    """Return the comments that pass the filters."""
    label, year = clean_label(label), clean_int(year)
    video = normalize(str(video)) if video else None
    selected = []
    for comment in comments:
        if label and comment["label"] != label:
            continue
        if year and comment["year"] != year:
            continue
        if video and video not in comment["video_norm"]:
            continue
        selected.append(comment)
    return selected


# ------------------------------ tools ------------------------------

def label_counts(year=None, video=None):
    """How many comments of each type, for the whole channel, one year, or one video."""
    selected = select(year=year, video=video)
    counts = Counter(comment["label"] for comment in selected)
    total = len(selected)
    return {
        "total_comments": total,
        "counts": {name: counts[name] for name in LABELS},
        "percent": {name: round(100 * counts[name] / total, 1) if total else 0 for name in LABELS},
    }


def search_comments(keyword, label=None, year=None, limit=8):
    """Find comments that contain a word. Returns how many, and the most liked ones."""
    words = normalize(str(keyword or "")).split()
    # "السحبة" should also find "سحبة": drop "ال" from the start of each word.
    words = [word[2:] if word.startswith("ال") and len(word) > 4 else word for word in words]
    if not words:
        return {"error": "keyword is required"}
    limit = max(1, min(clean_int(limit, 8), 20))
    matches = [c for c in select(label=label, year=year)
               if all(word in c["norm"] for word in words)]
    matches.sort(key=lambda comment: comment["likes"], reverse=True)
    return {
        "total_matches": len(matches),
        "by_label": dict(Counter(c["label"] for c in matches)),
        "examples": [{"text": c["text"][:200], "likes": c["likes"], "year": c["year"],
                      "label": c["label"], "video": c["video"][:60]} for c in matches[:limit]],
    }


def word_only(token):
    """Keep the letters and digits of one word: "بول😳" -> "بول", "١٠" -> "10"."""
    return re.sub(r"[\W_]", "", token or "").translate(ARABIC_DIGITS)


def without_al(word):
    """ "الطيران" and "طيران" are the same name."""
    return word[2:] if word.startswith("ال") and len(word) > 4 else word


def one_letter_apart(a, b):
    """True if two words are the same, or differ by one letter (a typo)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    for i in range(len(b)):
        if len(a) == len(b):
            if a[:i] + a[i + 1:] == b[:i] + b[i + 1:]:     # one letter replaced
                return True
        elif a == b[:i] + b[i + 1:]:                       # one letter added or missing
            return True
    return False


def top_requested_mods(limit=10, year=None):
    """
    The mods the audience asks for most: how many request comments name each mod.
    The name is the word (or two words) that follows "مود".

    A name can be one word ("مود ناروتو") or two ("مود دراغون بول"). A first word is
    treated as the start of a two-word name when one second word follows it often
    (at least 20% of the time). Second words that differ by one letter are counted
    together, because they are spelling mistakes ("بول" and "بال").
    """
    limit = max(1, min(clean_int(limit, 10), 30))
    first_count, spellings, seconds = Counter(), {}, {}
    for comment in select(label="request", year=year):
        asked = {}      # each mod is counted once per comment, even if it is repeated 60 times
        # The second word is only looked at, not consumed, so "مود الطيران مود ناروتو"
        # still counts both mods.
        for match in re.finditer(r"مود\s+(\S+)(?=(?:\s+(\S+))?)", comment["norm"]):
            first, second = word_only(match.group(1)), word_only(match.group(2))
            if not first or first in NOT_A_MOD:
                continue
            if not second or second in NOT_A_MOD:
                second = None
            asked.setdefault(without_al(first), (first, second))
        for key, (first, second) in asked.items():
            first_count[key] += 1
            spellings.setdefault(key, Counter())[first] += 1
            if second:
                seconds.setdefault(key, Counter())[second] += 1

    totals = Counter()
    for key, count in first_count.items():
        name = spellings[key].most_common(1)[0][0]          # the most common spelling
        left = Counter(seconds.get(key, {}))
        groups = []                                         # (second word, how many times)
        while left:
            best = left.most_common(1)[0][0]
            same = [word for word in left
                    if one_letter_apart(without_al(word), without_al(best))]
            groups.append((best, sum(left[word] for word in same)))
            for word in same:
                del left[word]
        if groups and groups[0][1] >= 0.2 * count:          # a two-word name
            for second, pair_count in groups:
                totals[f"{name} {second}"] += pair_count
        else:
            totals[name] += count
    return {"mods": [{"mod": name, "requests": count} for name, count in totals.most_common(limit)]}


def top_videos(label=None, limit=5):
    """Videos with the most comments, or with the most comments of one type."""
    label = clean_label(label)
    limit = max(1, min(clean_int(limit, 5), 20))
    totals = Counter(comment["video_id"] for comment in comments)
    of_label = Counter(c["video_id"] for c in comments if not label or c["label"] == label)
    result = []
    for video_id, count in of_label.most_common(limit):
        result.append({
            "video": videos.get(video_id, video_id)[:80],
            "comments_of_this_type": count,
            "all_comments": totals[video_id],
            "percent": round(100 * count / totals[video_id], 1),
        })
    return {"type": label or "all", "videos": result}
