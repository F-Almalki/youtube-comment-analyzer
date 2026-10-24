"""
The agent: ask questions about the channel's comments in Arabic.

What makes it an agent: the model does not just answer. It decides which
tool to call, reads the result, and decides again, until it can answer.

    question -> model picks a tool -> Python runs the tool -> model reads the
    result -> (another tool, or) the final answer

It opens a small chat page in your browser, because the Windows console
cannot display Arabic correctly.

Needs:   labeled_comments.csv   (made by classify_all.py)
         videos.csv             (made by fetch_comments.py)
         Ollama running, with the model below downloaded

How to run (Windows CMD, inside the project folder):
    py agent.py
"""

import datetime
import json
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

import agent_tools
import channel

MODEL = "qwen3:8b"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
PORT = 8000
MAX_STEPS = 6        # stop after this many rounds, so it can never loop forever
MEMORY_TURNS = 3     # how many earlier questions the agent remembers
THIS_YEAR = datetime.date.today().year

session = requests.Session()

SYSTEM_PROMPT = f"""أنت مساعد يحلل تعليقات قناة يوتيوب اسمها "{channel.NAME}"، وهي قناة ألعاب سعودية مشهورة بمودات قراند.
عندك أدوات تقرأ تعليقات القناة المصنفة، من سنة 2013 إلى {THIS_YEAR}. السنة الحالية {THIS_YEAR}.
التصنيفات: request (طلب)، praise (مدح)، criticism (نقد)، question (سؤال)، spam (سبام)، other (أخرى).

القواعد:
- لا تجاوب من عندك. استخدم الأدوات لتحصل على الأرقام والأمثلة.
- لا تضف أي فلتر لم يطلبه المستخدم. إذا لم يذكر سنة فاجعل year يساوي 0. إذا لم يذكر تصنيفاً فاجعل label يساوي all.
- اقتراحات الجمهور وأفكار الفيديوهات وما يريده الناس كلها تعني الطلبات (request). استخدم لها top_requested_mods، أو search_comments مع label=request.
- إذا كان السؤال عن الفيديوهات نفسها، مثل "أي فيديو" أو "أكثر الفيديوهات"، فاستخدم top_videos وضع نوع التعليق في label. لا تستخدم top_requested_mods إلا إذا كان المطلوب قائمة مودات.
- لا تقل "لا أعرف" قبل أن تجرب أداة مناسبة. قلها فقط إذا كان السؤال عن شيء لا يمكن أن يوجد في التعليقات، مثل موعد الفيديو القادم.
- جاوب بالعربي وباختصار، واذكر الأرقام التي رجعتها الأدوات كما هي.
- التصنيف آلي ودقته حوالي 60%، فالأرقام تقريبية. اذكر ذلك في جملة قصيرة آخر الجواب.

أمثلة على اختيار الأداة:
- "كم نسبة المدح؟" ← label_counts(year=0, video="")
- "وش يقولون عن ماين كرافت؟" ← search_comments(keyword="ماين كرافت", label="all", year=0)
- "وش أكثر المودات المطلوبة في 2018؟" ← top_requested_mods(limit=10, year=2018)
- "أي الفيديوهات عليها أكثر مدح؟" ← top_videos(label="praise", limit=5)
- "أي الفيديوهات عليها أكثر طلبات؟" ← top_videos(label="request", limit=5)
- "وش يقترح الجمهور للفيديو الجاي؟" ← top_requested_mods(limit=10, year=0)"""

LABEL_CHOICES = ["all"] + agent_tools.LABELS

# The description of each tool is what the model reads to decide which one to call.
TOOLS = [
    {"type": "function", "function": {
        "name": "label_counts",
        "description": "Count comments of each type (request, praise, criticism, question, spam, other) "
                       "for the whole channel. Can be limited to one year or one video.",
        "parameters": {"type": "object", "properties": {
            "year": {"type": "integer", "description": "0 = all years (default). Only a year the user wrote."},
            "video": {"type": "string", "description": "Part of a video title. Empty = all videos (default)."}}},
    }},
    {"type": "function", "function": {
        "name": "search_comments",
        "description": "Find comments that contain a word or a name. Returns the number of matches "
                       "and the most liked examples. Use it to see what people say about something.",
        "parameters": {"type": "object", "properties": {
            "keyword": {"type": "string", "description": "The word to look for, in Arabic or English"},
            "label": {"type": "string", "enum": LABEL_CHOICES, "description": "all = every type (default)"},
            "year": {"type": "integer", "description": "0 = all years (default). Only a year the user wrote."},
            "limit": {"type": "integer", "description": "How many examples, default 8"}},
            "required": ["keyword"]},
    }},
    {"type": "function", "function": {
        "name": "top_requested_mods",
        "description": "The mods the audience requests most, with the number of requests for each.",
        "parameters": {"type": "object", "properties": {
            "limit": {"type": "integer", "description": "How many mods, default 10"},
            "year": {"type": "integer", "description": "0 = all years (default). Only a year the user wrote."}}},
    }},
    {"type": "function", "function": {
        "name": "top_videos",
        "description": "Videos with the most comments, or with the most comments of one type "
                       "(for example the videos with the most criticism).",
        "parameters": {"type": "object", "properties": {
            "label": {"type": "string", "enum": LABEL_CHOICES, "description": "all = every type (default)"},
            "limit": {"type": "integer", "description": "How many videos, default 5"}}},
    }},
]

PYTHON_TOOLS = {
    "label_counts": agent_tools.label_counts,
    "search_comments": agent_tools.search_comments,
    "top_requested_mods": agent_tools.top_requested_mods,
    "top_videos": agent_tools.top_videos,
}

# Words in a question that show the user really asked about one type of comment.
LABEL_WORDS = {
    "request": ["طلب", "طلوب", "قترح", "افكار", "يبون", "يبغون", "يبي", "يبغ", "تبي", "تبغ",
                "نبي", "request"],
    "praise": ["مدح", "شكر", "عجب", "تشجيع", "ايجابي", "praise"],
    "criticism": ["نقد", "نتق", "شكو", "شكاو", "شتك", "عتاب", "زعل", "سلبي", "criticism"],
    "question": ["سؤال", "سوال", "اسئل", "سال", "استفسار", "question"],
    "spam": ["سبام", "اعلان", "دعاي", "spam"],
    "other": ["اخري", "other"],
}
THIS_YEAR_WORDS = ["هذي السنه", "هذه السنه", "هالسنه", "السنه هذي", "هذا العام", "السنه الحاليه"]
LAST_YEAR_WORDS = ["السنه الماضيه", "السنه اللي فاتت", "السنه الي فاتت", "العام الماضي"]


def ask_model(messages):
    """Send the conversation and the list of tools to the model. Return its reply."""
    body = {"model": MODEL, "messages": messages, "tools": TOOLS, "think": False,
            "options": {"temperature": 0}, "keep_alive": "30m", "stream": False}
    response = session.post(OLLAMA_URL, json=body, timeout=600)
    response.raise_for_status()
    return response.json()["message"]


def remove_unasked_filters(arguments, user_text):
    """
    A small model likes to fill every argument, even ones nobody asked for.
    In testing it added "year 2023" to questions about the whole channel.
    So a year or a comment type is kept only if the user's own words mention it.
    """
    text = agent_tools.normalize(user_text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")))
    years_asked = set(re.findall(r"(?:19|20)\d\d", text))
    if any(words in text for words in THIS_YEAR_WORDS):
        years_asked.add(str(THIS_YEAR))
    if any(words in text for words in LAST_YEAR_WORDS):
        years_asked.add(str(THIS_YEAR - 1))

    cleaned = dict(arguments)
    if "year" in cleaned and str(agent_tools.clean_int(cleaned["year"], 0)) not in years_asked:
        del cleaned["year"]
    label = agent_tools.clean_label(cleaned.get("label"))
    if "label" in cleaned and not (label and any(word in text for word in LABEL_WORDS[label])):
        del cleaned["label"]
    if "video" in cleaned and not str(cleaned["video"]).strip():
        del cleaned["video"]
    return cleaned


def run_tool(name, arguments, user_text):
    """
    Run one tool the model asked for. A mistake becomes a message the model can read.
    Returns the arguments that were really used, and the tool's result.
    """
    tool = PYTHON_TOOLS.get(name)
    if tool is None:
        return {}, {"error": f"There is no tool named {name}"}
    if not isinstance(arguments, dict):
        return {}, {"error": "arguments must be an object"}
    arguments = remove_unasked_filters(arguments, user_text)
    try:
        return arguments, tool(**arguments)
    except TypeError as error:
        return arguments, {"error": f"Wrong arguments: {error}"}


def answer(question, history=None):
    """The agent loop. Returns the final answer and the list of tools it used."""
    history = (history or [])[-MEMORY_TURNS:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": question})
    # Filters are checked against everything the user typed, so "and in 2018?" still works.
    user_text = " ".join([turn["question"] for turn in history] + [question])

    steps = []
    for _ in range(MAX_STEPS):
        reply = ask_model(messages)
        messages.append(reply)
        tool_calls = reply.get("tool_calls") or []
        if not tool_calls:                      # no tool wanted: this is the final answer
            text = re.sub(r"<think>.*?</think>", "", reply.get("content", ""), flags=re.S)
            return text.strip(), steps
        for call in tool_calls:
            name = call["function"]["name"]
            asked_arguments = call["function"].get("arguments") or {}
            used_arguments, result = run_tool(name, asked_arguments, user_text)
            # "asked" is what the model wrote, "arguments" is what was used after the check.
            steps.append({"tool": name, "arguments": used_arguments, "asked": asked_arguments,
                          "failed": "error" in result})
            messages.append({"role": "tool", "tool_name": name,
                             "content": json.dumps(result, ensure_ascii=False)})
    return "ما قدرت أوصل لجواب خلال عدد الخطوات المسموح. جرب تسأل بطريقة أبسط.", steps


# --------------------------- the chat page ---------------------------

PAGE = """<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>مساعد تعليقات القناة</title>
<style>
  :root { --bg:#f6f5f1; --card:#fff; --text:#1d1d1b; --muted:#6b6a65; --line:#dcdad2;
          --accent:#1f6f5c; --accent-text:#fff; --mine:#e3f1ec; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161715; --card:#20211e; --text:#ecebe6; --muted:#9c9a92; --line:#36372f;
            --accent:#58b79e; --accent-text:#0e1513; --mine:#24392f; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); line-height:1.7;
         font-family:"Segoe UI", Tahoma, system-ui, sans-serif; }
  main { max-width:760px; margin:0 auto; padding:28px 16px 140px; }
  header { display:flex; justify-content:space-between; align-items:center; gap:12px; }
  h1 { font-size:22px; margin:0; }
  .hint { color:var(--muted); font-size:15px; margin:4px 0 16px; }
  .examples { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:20px; }
  button { font:inherit; cursor:pointer; border-radius:10px; border:1px solid var(--line);
           background:var(--card); color:var(--text); padding:8px 14px; }
  button:hover { border-color:var(--accent); }
  button:disabled { opacity:.6; cursor:default; }
  .bubble { border:1px solid var(--line); border-radius:12px; padding:14px 16px; margin:10px 0;
            background:var(--card); white-space:pre-wrap; overflow-wrap:anywhere; }
  .question { background:var(--mine); border-color:transparent; }
  .steps { margin:6px 0 0; padding:0; list-style:none; color:var(--muted); font-size:13px;
           direction:ltr; text-align:left; font-family:Consolas, monospace; overflow-wrap:anywhere; }
  .steps-title { color:var(--muted); font-size:13px; margin-top:12px; }
  form { position:fixed; bottom:0; left:0; right:0; background:var(--bg);
         border-top:1px solid var(--line); padding:12px 16px; }
  .row { max-width:760px; margin:0 auto; display:flex; gap:8px; }
  input { flex:1; min-width:0; font:inherit; padding:12px 14px; border-radius:10px;
          border:1px solid var(--line); background:var(--card); color:var(--text); }
  input:focus-visible, button:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  .send { background:var(--accent); color:var(--accent-text); border-color:var(--accent); }
</style>
</head>
<body>
<main>
  <header>
    <h1>مساعد تعليقات القناة</h1>
    <button type="button" id="reset">محادثة جديدة</button>
  </header>
  <p class="hint">اسأل عن تعليقات جمهورك. المساعد يختار الأداة المناسبة ويجاوبك من البيانات، ويتذكر آخر ثلاث أسئلة.</p>
  <div class="examples" id="examples"></div>
  <div id="chat"></div>
</main>
<form id="form">
  <div class="row">
    <input id="question" placeholder="اكتب سؤالك هنا" autocomplete="off" autofocus>
    <button class="send" id="send">اسأل</button>
  </div>
</form>
<script>
const EXAMPLES = ["وش أكثر المودات اللي يطلبها الجمهور؟", "كم نسبة النقد في القناة؟",
                  "وش يقولون الناس عن السحبة؟", "أي الفيديوهات عليها أكثر طلبات؟"];
const chat = document.getElementById("chat");
const input = document.getElementById("question");
const send = document.getElementById("send");
let history = [];   // earlier questions and answers, sent with each new question

function bubble(text, extraClass) {
  const div = document.createElement("div");
  div.className = "bubble" + (extraClass ? " " + extraClass : "");
  div.textContent = text;
  chat.append(div);
  div.scrollIntoView({ block: "end" });
  return div;
}

async function ask(question) {
  question = question.trim();
  if (!question || send.disabled) return;
  input.value = "";
  send.disabled = true;
  bubble(question, "question");
  const box = bubble("يفكر ويستخدم الأدوات...");
  try {
    const response = await fetch("/ask", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question, history: history }) });
    const data = await response.json();
    box.textContent = data.answer;
    history.push({ question: question, answer: data.answer });
    history = history.slice(-3);
    if (data.steps && data.steps.length) {
      const title = document.createElement("div");
      title.className = "steps-title";
      title.textContent = "الأدوات اللي استخدمها:";
      const list = document.createElement("ul");
      list.className = "steps";
      for (const step of data.steps) {
        const item = document.createElement("li");
        item.textContent = step.tool + "(" + JSON.stringify(step.arguments) + ")" +
                           (step.failed ? "  -> failed, tried again" : "");
        list.append(item);
      }
      box.append(title, list);
    }
  } catch (error) {
    box.textContent = "صار خطأ في الاتصال. تأكد إن نافذة البرنامج لسا مفتوحة.";
  }
  send.disabled = false;
  input.focus();
  box.scrollIntoView({ block: "end" });
}

for (const text of EXAMPLES) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = text;
  button.addEventListener("click", () => ask(text));
  document.getElementById("examples").append(button);
}
document.getElementById("form").addEventListener("submit", (event) => {
  event.preventDefault();
  ask(input.value);
});
document.getElementById("reset").addEventListener("click", () => {
  history = [];
  chat.replaceChildren();
  input.focus();
});
</script>
</body>
</html>
"""


def read_history(data):
    """Keep only well-formed earlier turns sent by the page."""
    turns = []
    for turn in data.get("history") or []:
        if isinstance(turn, dict) and turn.get("question") and turn.get("answer"):
            turns.append({"question": str(turn["question"])[:1000],
                          "answer": str(turn["answer"])[:2000]})
    return turns[-MEMORY_TURNS:]


class Handler(BaseHTTPRequestHandler):
    def send_body(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send_body(PAGE.encode("utf-8"), "text/html; charset=utf-8")

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            question = str(data.get("question", ""))[:1000]
            text, steps = answer(question, read_history(data))
        except requests.exceptions.ConnectionError:
            text, steps = "برنامج Ollama مقفل. افتحه وجرب مرة ثانية.", []
        except requests.exceptions.HTTPError as error:
            text, steps = "Ollama رجع خطأ: " + error.response.text[:300], []
        except Exception as error:      # keep the page alive whatever happens
            text, steps = f"صار خطأ: {error}", []
        body = json.dumps({"answer": text, "steps": steps}, ensure_ascii=False).encode("utf-8")
        self.send_body(body, "application/json; charset=utf-8")

    def log_message(self, *args):       # keep the console quiet
        pass


def main():
    try:
        count = agent_tools.load()
    except FileNotFoundError:
        sys.exit("labeled_comments.csv was not found. Run this inside the yt-project folder.")
    address = f"http://127.0.0.1:{PORT}"
    print(f"Loaded {count} comments. Model: {MODEL}")
    print(f"Open {address} in your browser. Press Ctrl+C here to stop.")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
