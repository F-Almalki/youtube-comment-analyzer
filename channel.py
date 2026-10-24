"""
The channel's name and handle.

They are read from channel.json, which stays on my PC and is not in this
repository. To use the project on your own channel, copy channel.example.json
to channel.json and write your channel there.

    handle   used by fetch_comments.py to find the channel (for example "@name")
    name     what the audience calls the creator. It is put into the instructions
             of the classifier and the agent, because commenters use it a lot.
    series   optional: the channel's best-known series, named in the classifier's
             instructions so the model knows it
"""

import json
import os
import sys

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "channel.json")
DEFAULT_NAME = "صاحب القناة"      # "the channel owner", used when there is no channel.json


def read():
    try:
        with open(CONFIG_FILE, encoding="utf-8-sig") as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Note: channel.json was not found, so a generic channel name is used.",
              file=sys.stderr)
        data = {}
    except ValueError:
        sys.exit("channel.json is not valid JSON. Compare it with channel.example.json.")
    if not isinstance(data, dict):
        sys.exit("channel.json must look like channel.example.json.")
    return (str(data.get("handle") or "").strip(),
            str(data.get("name") or "").strip() or DEFAULT_NAME,
            str(data.get("series") or "").strip())


HANDLE, NAME, SERIES = read()
