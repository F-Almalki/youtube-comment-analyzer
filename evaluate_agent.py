"""
Measure the agent with a fixed list of questions.

For every question in agent_questions.json it checks three things:
  1. Tool:      did the agent call the right tool?
  2. Arguments: with the right year, comment type, keyword or video?
                (measured twice: as the model wrote them, and after the check
                in agent.py that removes filters nobody asked for)
  3. Answer:    does the final answer contain the main number or name that the
                tool returned? For questions the comments cannot answer, does
                the agent say it does not know?

The questions are split in two sets:
  dev   20 questions. Some were used while writing the agent's instructions.
  test  40 questions. Never used to change the agent. This is the reported score.

Needs:   labeled_comments.csv, videos.csv, Ollama running with the model downloaded
Output:  agent_eval_summary_<model>.txt   the numbers (no comment text)
         agent_eval_report_<model>.html   every question and answer, to read in the browser
         agent_eval_results_<model>.json  the raw results

How to run (Windows CMD, inside the project folder):
    py evaluate_agent.py                 uses the model written in agent.py
    py evaluate_agent.py qwen3:14b       tries another model
"""

import html
import itertools
import json
import math
import re
import sys
import time

import requests

import agent
import agent_tools

QUESTIONS_FILE = "agent_questions.json"

# If the answer contains one of these, the agent said it cannot know.
DECLINE_WORDS = ["لا اعرف", "ما اعرف", "لا استطيع", "ما اقدر", "لا يمكن", "غير متوفر", "غير متاح",
                 "غير موجود", "غير محفوظ", "لا توجد", "لا يوجد", "لا تتوفر", "لا تحتوي", "ليس لدي",
                 "ما عندي", "لا املك", "لا تشمل"]
# Words that are in many video titles, so they do not identify one video.
COMMON_TITLE_WORDS = {"قراند", "مود", "الجزء", "فيديو", "تحدي", "ضد", "gta", "mod"}
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫", "0123456789.")


# ------------------------- comparing tool calls -------------------------

def simple(arguments):
    """Put a tool call's arguments in one form, so "all", 0 and "" all mean 'no filter'."""
    arguments = arguments if isinstance(arguments, dict) else {}
    return {
        "year": agent_tools.clean_int(arguments.get("year"), 0) or None,
        "label": agent_tools.clean_label(arguments.get("label")),
        "video": agent_tools.normalize(str(arguments.get("video") or "")),
        "keyword": agent_tools.normalize(str(arguments.get("keyword") or "")),
        "limit": agent_tools.clean_int(arguments.get("limit")),
    }


def as_list(value):
    return value if isinstance(value, list) else [value]


def wanted_year(value):
    if value == "this_year":
        return agent.THIS_YEAR
    if value == "last_year":
        return agent.THIS_YEAR - 1
    return value


def same_tool(wanted, step):
    return step["tool"] == wanted["tool"]


def same_call(wanted, step, which):
    """which = "arguments" (after the check in agent.py) or "asked" (as the model wrote them)."""
    if not same_tool(wanted, step):
        return False
    got = simple(step.get(which))
    if got["year"] != wanted_year(wanted.get("year")):
        return False
    if got["label"] not in as_list(wanted.get("label")):
        return False
    if "video" in wanted:
        if not any(agent_tools.normalize(part) in got["video"] for part in as_list(wanted["video"])):
            return False
    elif got["video"]:
        return False
    if "keyword" in wanted:
        if not any(agent_tools.normalize(part) in got["keyword"] for part in as_list(wanted["keyword"])):
            return False
    if "limit" in wanted and got["limit"] != wanted["limit"]:
        return False
    return True


def find_steps(wanted_calls, steps, same):
    """Give every wanted call its own step. Returns the steps, or None if that is impossible."""
    for order in itertools.permutations(range(len(steps)), len(wanted_calls)):
        if all(same(wanted, steps[index]) for wanted, index in zip(wanted_calls, order)):
            return [steps[index] for index in order]
    return None


# --------------------------- checking the answer ---------------------------

def numbers_in(text):
    text = text.translate(DIGITS)
    text = re.sub(r"(?<=\d)[,٬](?=\d{3})", "", text)          # 7,662 -> 7662
    numbers = [float(number) for number in re.findall(r"\d+(?:\.\d+)?", text)]
    # Some people write 7.662 for seven thousand. Accept that reading too.
    numbers += [float(number.replace(".", "")) for number in re.findall(r"\d{1,3}(?:\.\d{3})+", text)]
    return numbers


def has_number(text, value):
    return any(abs(number - value) < 0.051 for number in numbers_in(text))


def main_fact(check, step):
    """
    Run the tool again with the arguments the agent really used, and return
    (what to look for, description). Tools are plain Python, so this is exact.
    """
    try:
        result = agent.PYTHON_TOOLS[step["tool"]](**step["arguments"])
    except (KeyError, TypeError):
        return None
    if "error" in result:
        return None
    if check.startswith("percent:") and step["tool"] == "label_counts":
        label = check.split(":")[1]
        return ("numbers", [result["percent"][label], result["counts"][label]],
                f"{label}: {result['percent'][label]}% or {result['counts'][label]}")
    if check == "total" and step["tool"] == "label_counts":
        return ("numbers", [result["total_comments"]], f"total {result['total_comments']}")
    if check == "matches" and step["tool"] == "search_comments":
        return ("numbers", [result["total_matches"]], f"matches {result['total_matches']}")
    if check == "top_mod" and step["tool"] == "top_requested_mods" and result["mods"]:
        name = result["mods"][0]["mod"]
        return ("words", [agent_tools.without_al(word) for word in name.split()], f"top mod {name}")
    if check == "top_video" and step["tool"] == "top_videos" and result["videos"]:
        top = result["videos"][0]
        words = [word for word in agent_tools.normalize(re.sub(r"[\W_]", " ", top["video"])).split()
                 if len(word) >= 4 and word not in COMMON_TITLE_WORDS]
        return ("title", (words, top["comments_of_this_type"]), f"top video {top['video'][:50]}")
    return None


def answer_is_right(check, answer, steps):
    """Returns (True / False / None when there is nothing to check, what was expected)."""
    plain_answer = agent_tools.normalize(answer)
    if check == "decline":
        return any(words in plain_answer for words in DECLINE_WORDS), "says it cannot know"
    if not check:
        return None, ""
    if not steps:
        return False, "the right tool was not called"
    notes = []
    for step in steps:
        fact = main_fact(check, step)
        if fact is None:
            return None, ""
        kind, value, note = fact
        notes.append(note)
        if kind == "numbers":
            found = any(has_number(answer, number) for number in value)
        elif kind == "words":
            found = all(word in plain_answer for word in value)
        else:
            words, count = value
            found = has_number(answer, count) or any(word in plain_answer for word in words)
        if not found:
            return False, "; ".join(notes)
    return True, "; ".join(notes)


# ------------------------------ one question ------------------------------

def grade(question, answer, steps):
    """Compare what the agent did with what the question expects."""
    choices = [question["calls"]] + list(question["or"])
    cannot_know = question["check"] == "decline"
    graded = {"tool_ok": None, "arguments_ok": None, "asked_ok": None}
    matched = []
    if not cannot_know:
        best = None
        for wanted in choices:      # the first acceptable way of answering that fits best
            by_tool = find_steps(wanted, steps, same_tool)
            by_used = find_steps(wanted, steps, lambda w, s: same_call(w, s, "arguments"))
            by_asked = find_steps(wanted, steps, lambda w, s: same_call(w, s, "asked"))
            score = (by_used is not None, by_tool is not None)
            if best is None or score > best[0]:
                best = (score, by_tool, by_used, by_asked)
        _, by_tool, by_used, by_asked = best
        graded["tool_ok"] = by_tool is not None
        graded["arguments_ok"] = by_used is not None          # after the check in agent.py
        graded["asked_ok"] = by_asked is not None             # as the model wrote them
        matched = by_used or by_tool or []
    graded["answer_ok"], graded["expected_answer"] = answer_is_right(question["check"], answer, matched)
    parts = [graded["arguments_ok"], graded["answer_ok"]]
    graded["all_ok"] = all(part for part in parts if part is not None)
    return graded


def run_question(question, earlier):
    history = []
    if question["after"]:
        parent = earlier[question["after"]]
        history = [{"question": parent["question"], "answer": parent["answer"]}]
    started = time.time()
    try:
        answer, steps = agent.answer(question["question"], history)
    except requests.exceptions.ConnectionError:
        sys.exit("Ollama is not running. Open the Ollama app, then run this again.")
    except requests.exceptions.HTTPError as error:
        message = error.response.text[:300]
        if error.response.status_code == 404:
            message += f"\nDownload the model first:  ollama pull {agent.MODEL}"
        sys.exit(f"Ollama returned an error: {message}")
    except Exception as error:          # one broken reply should not stop the whole run
        answer, steps = f"ERROR: {error}", []
    result = dict(question, answer=answer, steps=steps, seconds=round(time.time() - started, 1))
    result.update(grade(question, answer, steps))
    return result


# ------------------------------- the report -------------------------------

def wilson(correct, total):
    """95% confidence interval for a share (Wilson score interval), in percent."""
    z = 1.959964
    share = correct / total
    middle = (share + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, 100 * (middle - half)), min(100.0, 100 * (middle + half))


def score(results, key):
    """(right, counted, text). Questions where the check does not apply are not counted."""
    counted = [r for r in results if r[key] is not None]
    right = sum(1 for r in counted if r[key])
    if not counted:
        return 0, 0, "   -"
    low, high = wilson(right, len(counted))
    return right, len(counted), (f"{100 * right / len(counted):5.1f}%  ({right:>2} of {len(counted):>2})"
                                 f"   95% CI {low:4.1f} to {high:4.1f}")


ROWS = [("tool_ok", "Right tool"),
        ("asked_ok", "Right tool and arguments, as the model wrote them"),
        ("arguments_ok", "Right tool and arguments, after the check in agent.py"),
        ("answer_ok", "Answer has the tool's main number, or says it cannot know"),
        ("all_ok", "Everything right")]


def build_summary(results):
    lines = [f"Model: {agent.MODEL}", f"Questions: {len(results)}"]
    for name, title in [("test", "TEST SET (never used to change the agent). This is the score to report."),
                        ("dev", "DEV SET (used while writing the agent's instructions).")]:
        part = [r for r in results if r["set"] == name]
        if not part:
            continue
        lines += ["", f"{title}  Questions: {len(part)}"]
        for key, label in ROWS:
            lines.append(f"  {label:<58}{score(part, key)[2]}")
        fixed = sum(1 for r in part if r["arguments_ok"] and r["asked_ok"] is False)
        broken = sum(1 for r in part if r["arguments_ok"] is False and r["asked_ok"])
        lines.append(f"  The check in agent.py fixed {fixed} question(s) and broke {broken}.")
        lines.append("")
        lines.append(f"  {'kind':<14}{'questions':>10}{'tool':>8}{'arguments':>11}{'answer':>9}{'all':>8}")
        for kind in sorted({r["kind"] for r in part}):
            rows = [r for r in part if r["kind"] == kind]
            cells = []
            for key in ["tool_ok", "arguments_ok", "answer_ok", "all_ok"]:
                right, counted, _ = score(rows, key)
                cells.append(f"{right}/{counted}" if counted else "-")
            lines.append(f"  {kind:<14}{len(rows):>10}{cells[0]:>8}{cells[1]:>11}{cells[2]:>9}{cells[3]:>8}")
        wrong = [r for r in part if not r["all_ok"]]
        lines.append("")
        lines.append(f"  Not fully right: {len(wrong)}")
        for r in wrong:
            calls = ", ".join(f"{s['tool']}({json.dumps(s['arguments'], ensure_ascii=False)})"
                              for s in r["steps"]) or "no tool"
            marks = " ".join(f"{key}={'-' if r[key + '_ok'] is None else 'ok' if r[key + '_ok'] else 'WRONG'}"
                             for key in ["tool", "arguments", "answer"])
            lines.append(f"    {r['id']}  {marks}  called: {calls}")
    seconds = sum(r["seconds"] for r in results)
    lines += ["", f"Time: {seconds:.0f} seconds, {seconds / len(results):.1f} per question"]
    return "\n".join(lines)


PAGE_STYLE = """
  :root { --bg:#f6f5f1; --card:#fff; --text:#1d1d1b; --muted:#6b6a65; --line:#dcdad2;
          --ok:#1f6f5c; --bad:#b3372f; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161715; --card:#20211e; --text:#ecebe6; --muted:#9c9a92; --line:#36372f;
            --ok:#58b79e; --bad:#ef8a80; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); line-height:1.7;
         font-family:"Segoe UI", Tahoma, system-ui, sans-serif; }
  main { max-width:820px; margin:0 auto; padding:28px 16px 60px; }
  h1 { font-size:22px; margin:0 0 12px; }
  pre { direction:ltr; text-align:left; background:var(--card); border:1px solid var(--line);
        border-radius:12px; padding:14px; overflow-x:auto; font-size:13px; line-height:1.5; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:14px 16px; margin:12px 0; border-inline-start:5px solid var(--ok); }
  .card.wrong { border-inline-start-color:var(--bad); }
  .q { font-size:18px; margin:4px 0; }
  .a { white-space:pre-wrap; overflow-wrap:anywhere; margin:4px 0; }
  .label { color:var(--muted); font-size:13px; margin-top:8px; }
  .code { direction:ltr; text-align:left; font-family:Consolas, monospace; font-size:13px;
          margin:2px 0; padding-inline-start:18px; overflow-wrap:anywhere; }
  .ok { color:var(--ok); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
  .na { color:var(--muted); font-size:13px; font-family:"Segoe UI", Tahoma, system-ui, sans-serif; }
  p { margin:6px 0; }
"""


def mark(value):
    if value is None:
        return '<span class="na">لا ينطبق</span>'
    return '<span class="ok">صح</span>' if value else '<span class="bad">خطأ</span>'


def as_text(value):
    """Tool arguments as text that is safe to put in the page."""
    return html.escape(json.dumps(value, ensure_ascii=False))


def build_card(r):
    called = []
    for step in r["steps"]:
        line = html.escape(step["tool"]) + "(" + as_text(step["arguments"]) + ")"
        if step["asked"] != step["arguments"]:
            line += ' <span class="na">النموذج كتب: ' + as_text(step["asked"]) + "</span>"
        called.append("<li>" + line + "</li>")
    wanted = [as_text(calls) for calls in [r["calls"]] + list(r["or"]) if calls]
    must_say = " (المفروض يذكر: " + html.escape(r["expected_answer"]) + ")" if r["expected_answer"] else ""
    return "\n".join([
        '<section class="card ' + ("good" if r["all_ok"] else "wrong") + '">',
        "<div><b>" + html.escape(r["id"]) + '</b> <span class="na">' + html.escape(r["set"])
        + " · " + html.escape(str(r["kind"])) + " · " + str(r["seconds"]) + " ث</span></div>",
        '<p class="q">' + html.escape(r["question"]) + "</p>",
        "<p>الأداة: " + mark(r["tool_ok"]) + " &nbsp; المدخلات: " + mark(r["arguments_ok"])
        + " &nbsp; الجواب: " + mark(r["answer_ok"]) + "</p>",
        '<div class="label">المتوقع</div>',
        '<ul class="code"><li>' + (" أو ".join(wanted) or "ما يحتاج أداة") + "</li></ul>",
        '<div class="label">اللي سواه المساعد</div>',
        '<ul class="code">' + ("".join(called) or "<li>ما استخدم أي أداة</li>") + "</ul>",
        '<div class="label">الجواب' + must_say + "</div>",
        '<p class="a">' + html.escape(r["answer"]) + "</p>",
        "</section>",
    ])


def build_page(results, summary):
    return "\n".join([
        "<!doctype html>",
        '<html lang="ar" dir="rtl">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>نتيجة اختبار المساعد</title>",
        "<style>" + PAGE_STYLE + "</style>",
        "</head>",
        "<body>",
        "<main>",
        "<h1>نتيجة اختبار المساعد</h1>",
        "<pre>" + html.escape(summary) + "</pre>",
        "\n".join(build_card(r) for r in results),
        "</main>",
        "</body>",
        "</html>",
    ])


def main():
    if len(sys.argv) > 1:
        agent.MODEL = sys.argv[1]
    tag = agent.MODEL.replace(":", "-").replace("/", "-")
    try:
        with open(QUESTIONS_FILE, encoding="utf-8-sig") as file:
            questions = json.load(file)
    except FileNotFoundError:
        sys.exit(f"{QUESTIONS_FILE} must be in the same folder as this file.")
    try:
        count = agent_tools.load()
    except FileNotFoundError:
        sys.exit("labeled_comments.csv was not found. Run this inside the project folder.")
    print(f"Loaded {count} comments. Model: {agent.MODEL}. Questions: {len(questions)}")

    results, earlier = [], {}
    for number, question in enumerate(questions, start=1):
        result = run_question(question, earlier)
        earlier[question["id"]] = result
        results.append(result)
        marks = "  ".join(
            f"{key} {'-' if result[key + '_ok'] is None else 'ok' if result[key + '_ok'] else 'WRONG'}"
            for key in ["tool", "arguments", "answer"])
        print(f"[{number:>2}/{len(questions)}] {question['id']}  {marks}  {result['seconds']}s")

    summary = build_summary(results)
    with open(f"agent_eval_summary_{tag}.txt", "w", encoding="utf-8") as file:
        file.write(summary + "\n")
    with open(f"agent_eval_report_{tag}.html", "w", encoding="utf-8") as file:
        file.write(build_page(results, summary))
    with open(f"agent_eval_results_{tag}.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=1)
    print()
    # The summary lists tool arguments, which can be Arabic. The console cannot show Arabic.
    print(summary.encode("ascii", "replace").decode("ascii"))
    print()
    print(f"Saved agent_eval_summary_{tag}.txt and agent_eval_report_{tag}.html")


if __name__ == "__main__":
    main()
