"""
The comment classifier, version 2.

Version 1 only gave the model short definitions. It scored 48% on the test comments.
Version 2 adds two things, written after reading the model's mistakes:
  1. Rules that match how the channel owner labels comments.
  2. Real example comments, taken ONLY from the first 100 labeled comments.

The other 200 labeled comments are never shown to the model.
They are the test set that evaluate.py uses to measure accuracy.

The channel's name is written as {channel} below, and its best-known series as
{series}. Both are filled in from channel.json (see channel.py), so they are not
stored in this repository. The results in the README were measured with the real
names filled in.
"""

import json
import re

import requests

import channel

MODEL = "qwen3:8b"

# Use 127.0.0.1, not "localhost". On Windows, "localhost" is tried over IPv6 first.
# Ollama does not listen there, so every request waited about 2 seconds before it
# fell back to IPv4. That delay made every model look equally slow.
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

# One session keeps the connection open instead of reconnecting for every comment.
session = requests.Session()
VERSION = "v2 (rules + examples)"

# English name (what the model answers) -> Arabic name (for reading)
LABELS = {
    "request": "طلب",
    "praise": "مدح",
    "criticism": "نقد",
    "question": "سؤال",
    "spam": "سبام",
    "other": "أخرى",
}

TEMPLATE = """أنت تصنف تعليقات الجمهور على قناة يوتيوب سعودية للألعاب اسمها "{channel}".
القناة مشهورة بمودات قراند (GTA){series}. أغلب المعلقين صغار، ويكتبون بالعامية وبأخطاء إملائية.

اختر تصنيفاً واحداً فقط:

request: يطلب من {channel} محتوى. مثل مود جديد، لعبة، جزء ثاني، تكملة سلسلة، أو أن ينزل فيديوهات أكثر.
  التعليق الذي فيه اسم مود أو لعبة فقط يعتبر طلباً، مثل "مود ناروتو".
praise: مدح أو شكر أو تشجيع، مثل "استمر" و"واصل" و"كمل يا بطل".
  الضحك فقط (ههههه) أو إيموجي الضحك والقلوب فقط يعتبر مدحاً. وكذلك "لايك" والفرح برجوعه والحنين للفيديوهات القديمة.
criticism: نقد أو شكوى أو عتاب أو تصحيح خطأ في الفيديو.
  كلمة "سحبة" أو "تسحب" معناها أنه غاب عن القناة وترك جمهوره، وهذا عتاب.
question: سؤال يحتاج جواباً، مثل اسم أغنية أو ماب أو لعبة، أو متى ينزل شيء.
spam: إعلان، رقم جوال أو حساب، مسابقات، كلام لا علاقة له بالفيديو، أو حروف ورموز بلا معنى.
other: أي شيء آخر. مثل "أول"، توقيت لقطة مع تعليق عليها، مزحة، تحية، كلام عن أحداث الفيديو، أو "قافلة باري".

أمثلة:
مود كونان ← request
كمل بولي ← request
نبي الجزء الثاني ← request
black ops 3 ps4 😊 ← request
يا خي حاول تنزل كثير 😂 ← request
استمر ← praise
كمل يا بطل😉😉 ← praise
ههههههههه ← praise
♥️♥️🥰🤩🤣😂 ← praise
لايك ← praise
اخخخخ بس ايام مودات قراند والله اشتقنا ← praise
يا سحبة ← criticism
اخيرا اتذكرتنا 😑😑 ← criticism
مقطع تسليكي 😂 ههههههه ← criticism
بس سايتما ما يطير بل يقفز لمسافة كبيرة ← criticism
شو اسم الغنيه ← question
ماكو ماكو ايش اسم الماب ← question
يا {channel} وين الجزء الثاني من valheim ← question
كتيبة #I_GRAM ← spam
يارب افووز كمبيوتر اقاريو =) ← spam
ابي ايدي مهكر قراند 5 سوني 4 هاذا رقمي ← spam
٠٠ ← spam
اول ← other
6:06 صوتو زي محمد مشيع 😂😂😂 ← other
3:35 مين لاحض مود بات مان🤣😂 ← other
قافله باري ← other
السلام عليكم ← other

أجب بصيغة JSON فقط."""

SERIES = '، مثل سلسلة "%s"' % channel.SERIES if channel.SERIES else ""
INSTRUCTIONS = TEMPLATE.replace("{channel}", channel.NAME).replace("{series}", SERIES)

# This forces the model to answer with one of our labels and nothing else.
ANSWER_FORMAT = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": list(LABELS)}},
    "required": ["label"],
}


def clean(text):
    """Turn line breaks and long gaps into single spaces, and cut very long comments."""
    return re.sub(r"\s+", " ", text).strip()[:500]


def classify(text):
    """Ask the local model for the label of one comment."""
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": clean(text)},
        ],
        "format": ANSWER_FORMAT,
        "think": False,                  # no long "Thinking..." step, much faster
        "options": {
            "temperature": 0,            # same comment gives the same answer
            "num_predict": 40,           # the answer is a few words, never let it run long
        },
        "keep_alive": "30m",             # keep the model loaded between comments
        "stream": False,
    }
    response = session.post(OLLAMA_URL, json=body, timeout=300)
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    try:
        label = json.loads(content)["label"]
    except (KeyError, ValueError, TypeError):
        # The answer was cut or not clean JSON: look for a label name inside it.
        label = next((name for name in LABELS if name in content), "other")
    return label if label in LABELS else "other"
