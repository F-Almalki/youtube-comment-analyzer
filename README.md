# YouTube Comment Analyzer (Arabic, local LLM)

I run a Saudi gaming channel on YouTube. It has more than 127,000 comments, far too many to read.
This project collects them, uses a language model running on my own PC to sort each comment into one of six types, and lets me ask questions about the result in Arabic.

Everything runs locally with [Ollama](https://ollama.com). No comment is sent to an outside AI service.

## What it does

1. **Collect.** `fetch_comments.py` downloads the public comments of every public video through the YouTube Data API.
2. **Label by hand.** `make_sample.py` picks 300 random comments. `label.html` is a small page for labeling them quickly with the keyboard.
3. **Classify.** `classifier.py` asks a local model (Qwen3 through Ollama) to label a comment as `request`, `praise`, `criticism`, `question`, `spam` or `other`.
4. **Measure the classifier.** `evaluate.py` compares the model's labels with mine. `compare_runs.py` adds confidence intervals and tests whether one run is really better than another.
5. **Run on everything.** `classify_all.py` labels the full dataset. It saves as it goes and can be stopped and continued.
6. **Ask.** `agent.py` is a tool-calling agent. I ask a question in Arabic, the model chooses a tool from `agent_tools.py`, reads the result, and answers.
7. **Measure the agent.** `evaluate_agent.py` runs 60 fixed questions and checks the tool, the arguments and the answer.

## Data

| | |
| --- | --- |
| Videos | 211 public videos |
| Comments collected | 127,894 top-level comments, from 2013 to 2026 |
| Comments labeled by the model | 127,853 (the ones that have text) |
| Language | Mostly Saudi and Gulf dialect Arabic, with many spelling mistakes |
| Labeled by hand | 300 random comments |

Two notes on the data:

- The API returned two comments twice, so there are 127,892 different comments. `fetch_comments.py` now skips repeats.
- Viewers repeat the same short comments. 22% of the comments have exactly the same text as another comment ("اول", meaning "first", appears 1,022 times).

## How accurate is the classifier

I used the first 100 hand-labeled comments to write the instructions and choose the examples.
Accuracy is measured only on the other 200 comments.

| Setup | Accuracy | 95% confidence interval | Macro F1 | F1 for `request` |
| --- | --- | --- | --- | --- |
| Baseline: always answer `other` | 42.0% | 35.4 to 48.9 | | |
| Qwen3 8B, short definitions only | 48.0% | 41.2 to 54.9 | 0.34 | 0.53 |
| Qwen3 8B, channel rules + 27 examples | 59.5% | 52.6 to 66.1 | 0.51 | 0.72 |
| Qwen3 14B, channel rules + 27 examples | 63.0% | 56.1 to 69.4 | 0.46 | 0.76 |

What the numbers say:

- **The rules and examples helped.** They added 11.5 points (95% confidence interval 3.5 to 19.5, McNemar test p = 0.008).
- **The bigger model did not clearly win.** Its gain of 3.5 points is 7 comments out of 200. The interval runs from -2.3 to +9.3 (p = 0.31), and its macro F1 is lower.
- **A few test comments are repeats.** 10 of the 200 test comments are nearly identical to a comment in the first 100, or to a phrase quoted in the instructions (for example "اول" and "يا سحبة"). This is expected, because viewers repeat the same short comments. Without those 10, accuracy is 57.4% for the 8B model and 61.1% for the 14B model.
- **Requests are found well, but over-counted.** For `request`, recall is 0.85 and precision is 0.62.

The full numbers are in `eval_summary_qwen3-8b.txt`, `eval_summary_qwen3-14b.txt` and `eval_stats.txt`.

## What the comments say

Labels from Qwen3 8B on all 127,853 comments:

| Type | Comments | Share |
| --- | --- | --- |
| other | 41,563 | 32.5% |
| praise | 32,379 | 25.3% |
| request | 28,780 | 22.5% |
| spam | 9,998 | 7.8% |
| criticism | 7,662 | 6.0% |
| question | 7,471 | 5.8% |

Most requested mods. This is the number of request comments that name each mod after the word "مود":

| Mod | Request comments |
| --- | --- |
| The channel's own mod series | 1,285 |
| Naruto | 759 |
| Dragon Ball | 262 |
| Gumball | 246 |
| Ben 10 | 226 |
| One Piece | 221 |
| Attack on Titan | 105 |

These counts are low estimates. Each row is one spelling of the name, and close typos are merged, but different spellings are not.
For example, 57 more comments write Ben 10 with digits ("بن 10"), and 57 more write Naruto as "نارتو".

## How good is the agent

The agent has four tools: count comments by type, search comments for a word, list the most requested mods, and list the top videos.
The model only chooses a tool and its arguments. Python does the counting, so the numbers in an answer come from the data and not from the model.

I measure it with 60 questions in `agent_questions.json`:

- **20 dev questions.** I may change the agent's instructions because of these.
- **40 test questions.** These are never used to change the agent.

For each question the script checks the tool, the arguments, and whether the answer contains the main number or name that the tool returned.
Four test questions cannot be answered from comments (for example ad income). For those, the right answer is "I don't know".

Results on the 40 test questions, Qwen3 8B:

| | First run | After one change |
| --- | --- | --- |
| Right tool and arguments (36 questions) | 34 (94.4%) | 36 (100%) |
| Answer has the main number, or says "I don't know" (33 questions) | 29 (87.9%) | 30 (90.9%) |
| Everything right (40 questions) | 36 (90.0%) | 37 (92.5%) |
| 95% confidence interval for "everything right" | 76.9 to 96.0 | 80.1 to 97.4 |

How to read this:

- **The first run is the fair number.** Nothing had been tuned on these questions.
- **The one change.** One dev question failed: "which videos have the most requests?" was answered with a list of mods. I added one rule and one example to the instructions. That fixed it, but a different dev question then failed, so the dev score stayed at 19 of 20.
- **The two runs are not really different.** On the test questions the change fixed two and broke one. I had already seen the first test results when I made the change, so the second column only shows that the change did no harm.

The three test questions that still fail:

- "Which video has the most questions?" The tool was right, but the model named the fourth video in the list instead of the first.
- "Who comments the most?" Commenter names are not saved, so it should say "I don't know". It gave a meaningless answer about comment types.
- "How many dislikes does the Batman video have?" It should say "I don't know". It searched for "Batman" and reported the like count of one comment.

Example answers from the test run, translated:

| Question | What the agent called | Answer |
| --- | --- | --- |
| Did criticism go up in 2026 compared with 2018? | `label_counts(year=2026)`, then `label_counts(year=2018)` | 11.7% in 2026 against 4.5% in 2018, so it went up. |
| Give me the 3 most requested mods in 2017 | `top_requested_mods(limit=3, year=2017)` | Naruto (670), the channel's own series (446), Dragon Ball (227). |
| How much does the channel earn from ads? | no tool | I don't know. |

The full numbers are in `agent_eval_summary_qwen3-8b.txt` (after the change) and `agent_eval_first_run.txt`.

## What I learned

- **Reading the mistakes helped more than changing the model.** The first version did not know the channel's slang. For example, "يا سحبة" means the creator disappeared, which is a complaint. On the full dataset, 1,252 comments contain that word and the model labels 94% of them as criticism.
- **Measure before blaming the model.** My first full run was heading for more than 80 hours. Both model sizes ran at the same 0.4 comments per second, which pointed at the code and not the model. The cause was the address `localhost`: on Windows each request waited about two seconds on IPv6 before falling back to IPv4. Switching to `127.0.0.1` and reusing one connection gave about 7 comments per second, and the full run took about five hours on one RTX 4070 Ti SUPER.
- **A small model fills in arguments nobody asked for.** In my first try of the agent, a question about the whole channel was answered for the year 2023 only, because the model added `year: 2023` on its own. I gave every filter an explicit "no filter" value, and added a check in code that drops a year or comment type unless the user's own words mention it. In the 60 test and dev questions the model still added an unasked filter twice, and the check removed both.
- **Fixing one prompt mistake can cause another.** The rule I added for one dev question broke a different one. With a small model, I cannot trust a prompt change until the whole question set is run again.
- **Check the counting code, not only the model.** One comment repeated "مود سايتما" 62 times and was counted as 62 requests. Now each mod is counted once per comment.

## Limits

- The test set for the classifier is small (200 comments). Classes like `question` have very few examples, so their scores are not reliable.
- I labeled the comments alone. One comment text appears twice in the test set with two different labels from me. This puts a ceiling on the accuracy any model can reach.
- 60% accuracy is enough to see trends, such as which mods are requested most. It is not enough to trust the label of a single comment, so every number above is approximate.
- The agent's test set is small too (40 questions), and its questions were written for this channel.
- The agent's answers are checked automatically for one number or name only. For 7 search questions nothing in the answer is checked, only the tool call. In the first run, one of those answers did not show the comment that was asked for, and it still counted as right.
- The agent does not reliably say "I don't know". It failed 1 or 2 of the 4 such questions.

## How to run

You need Python 3, [Ollama](https://ollama.com), and a YouTube Data API key (a free Google Cloud key that reads public data).

```
py -m pip install -r requirements.txt
ollama pull qwen3:8b

copy channel.example.json channel.json
```

Open `channel.json` and write your channel's handle and the name your audience uses for you. Then:

```
set YOUTUBE_API_KEY=your-key-here
py fetch_comments.py

py make_sample.py
```

Open `label.html` in a browser, choose `sample_300.csv`, label the comments, and save the file into the project folder. Then:

```
py evaluate.py
py evaluate.py qwen3:14b
py compare_runs.py
py classify_all.py
py agent.py
py evaluate_agent.py
```

`agent.py` opens a chat page in the browser at `http://127.0.0.1:8000`.

The rules and examples in `classifier.py` and the questions in `agent_questions.json` are written for my channel. Write your own for a different one.

## Privacy

- Commenter names are never saved. Only the comment text, date and like count.
- The comment files are not in this repository. You can rebuild them with your own API key.
- The channel's name, handle and best-known series are not in this repository. They are read from `channel.json`, which stays on my PC. The results above were measured with the real names filled in.
- The API key is read from an environment variable and is never written in the code.

## How it was built

I wrote the code with the help of an AI coding assistant (Claude). The agent's test questions were drafted with it too.
I collected the data, labeled the 300 comments, and ran every experiment on my own PC.

## Next steps

- Try ALLaM, the Arabic model from SDAIA, and compare it with Qwen3.
- Have a second person label the test set, to measure how much two people agree.
- Make the agent say "I don't know" more reliably, and write more test questions.
