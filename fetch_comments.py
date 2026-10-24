"""
Week 1: download public comments from a YouTube channel.

Output:
    videos.csv    one row per video (title, date, views, likes, comment count)
    comments.csv  one row per top-level comment (text, likes, date, replies)

Commenter names are NOT saved. We only need the text.

The script saves after every video. If it stops for any reason
(internet drops, daily quota ends), run the same command again and
it continues from where it stopped.

The channel is read from channel.json (see channel.py).

How to run (Windows CMD):
    py -m pip install requests
    set YOUTUBE_API_KEY=your-key-here
    py fetch_comments.py
"""

import csv
import os
import sys
import time

import requests

import channel

# ----------------------------- settings -----------------------------
CHANNEL = channel.HANDLE         # channel handle (@name) or channel ID (starts with UC)
MAX_VIDEOS = 1000                # reads every video on the channel (up to 1000)
MAX_COMMENTS_PER_VIDEO = 100000  # high enough to take every comment on a video
RETRIES = 5                      # tries per request when the internet drops
# ---------------------------------------------------------------------

API = "https://www.googleapis.com/youtube/v3"
API_KEY = os.environ.get("YOUTUBE_API_KEY")

VIDEOS_FILE = "videos.csv"
COMMENTS_FILE = "comments.csv"
PROGRESS_FILE = "progress.txt"   # IDs of the videos already finished

VIDEO_COLUMNS = ["video_id", "title", "published_at", "views", "likes", "comment_count"]
COMMENT_COLUMNS = ["comment_id", "video_id", "text", "likes", "published_at", "reply_count"]

# One session keeps the connection open, so we do not reconnect on every request.
session = requests.Session()


class QuotaExceeded(Exception):
    """The daily API quota is used up. It resets at midnight Pacific Time."""


class NetworkDown(Exception):
    """The request failed several times in a row."""


def call_api(endpoint, params):
    """Send one request to the YouTube API and return the JSON answer."""
    params = dict(params, key=API_KEY)
    for attempt in range(1, RETRIES + 1):
        try:
            response = session.get(f"{API}/{endpoint}", params=params, timeout=30)
        except requests.exceptions.RequestException:
            response = None  # connection dropped or timed out
        if response is not None and response.status_code < 500:
            break
        if attempt == RETRIES:
            raise NetworkDown()
        time.sleep(2 ** attempt)  # wait 2, 4, 8, 16 seconds, then try again

    if response.status_code != 200:
        try:
            reason = response.json()["error"]["errors"][0]["reason"]
        except Exception:
            reason = response.text[:200]
        if reason == "quotaExceeded":
            raise QuotaExceeded()
        raise RuntimeError(f"{endpoint} failed ({response.status_code}): {reason}")
    return response.json()


def get_uploads_playlist(channel):
    """Every channel has a hidden playlist that contains all its uploads."""
    if channel.startswith("UC"):
        params = {"part": "contentDetails", "id": channel}
    else:
        params = {"part": "contentDetails", "forHandle": channel}
    data = call_api("channels", params)
    if not data.get("items"):
        raise RuntimeError(f"Channel not found: {channel}")
    return data["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_video_ids(playlist_id, limit):
    """Return the newest video IDs, up to the limit."""
    video_ids = []
    page_token = None
    while len(video_ids) < limit:
        params = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50}
        if page_token:
            params["pageToken"] = page_token
        data = call_api("playlistItems", params)
        for item in data.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return video_ids[:limit]


def get_video_details(video_ids):
    """Return title, date and statistics for each video (50 per request)."""
    videos = []
    for start in range(0, len(video_ids), 50):
        batch = video_ids[start:start + 50]
        data = call_api("videos", {"part": "snippet,statistics", "id": ",".join(batch)})
        for item in data.get("items", []):
            stats = item.get("statistics", {})
            videos.append({
                "video_id": item["id"],
                "title": item["snippet"]["title"],
                "published_at": item["snippet"]["publishedAt"],
                "views": stats.get("viewCount", ""),
                "likes": stats.get("likeCount", ""),
                "comment_count": stats.get("commentCount", ""),
            })
    return videos


def get_comments(video_id, limit):
    """Return (comments, quota_hit) for one video, newest comments first."""
    comments = []
    seen_ids = set()     # new comments can shift the pages, so the API may repeat a comment
    page_token = None
    while len(comments) < limit:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": 100,
            "order": "time",
            "textFormat": "plainText",
        }
        if page_token:
            params["pageToken"] = page_token
        try:
            data = call_api("commentThreads", params)
        except QuotaExceeded:
            return [], True
        except RuntimeError as error:
            # Comments can be turned off on a video. Skip it and continue.
            print(f"  skipped {video_id}: {error}")
            break
        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]
            if top["id"] in seen_ids:
                continue
            seen_ids.add(top["id"])
            snippet = top["snippet"]
            comments.append({
                "comment_id": top["id"],
                "video_id": video_id,
                "text": snippet.get("textDisplay", ""),
                "likes": snippet.get("likeCount", 0),
                "published_at": snippet.get("publishedAt", ""),
                "reply_count": item["snippet"].get("totalReplyCount", 0),
            })
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.1)  # be polite to the API
    return comments[:limit], False


def save_videos(videos):
    # utf-8-sig makes Arabic text open correctly in Excel
    with open(VIDEOS_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=VIDEO_COLUMNS)
        writer.writeheader()
        writer.writerows(videos)


def append_comments(comments):
    """Add the comments of one video to the end of comments.csv."""
    is_new_file = not os.path.exists(COMMENTS_FILE)
    with open(COMMENTS_FILE, "a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=COMMENT_COLUMNS)
        if is_new_file:
            writer.writeheader()
        writer.writerows(comments)


def load_finished_videos():
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def mark_finished(video_id):
    with open(PROGRESS_FILE, "a", encoding="utf-8") as file:
        file.write(video_id + "\n")


def count_comments():
    if not os.path.exists(COMMENTS_FILE):
        return 0
    with open(COMMENTS_FILE, newline="", encoding="utf-8-sig") as file:
        return sum(1 for _ in csv.DictReader(file))


def run():
    playlist_id = get_uploads_playlist(CHANNEL)
    video_ids = get_video_ids(playlist_id, MAX_VIDEOS)
    print(f"Found {len(video_ids)} videos")

    videos = get_video_details(video_ids)
    save_videos(videos)

    finished = load_finished_videos()
    if finished:
        print(f"Continuing. {len(finished)} videos were already finished.")
    elif os.path.exists(COMMENTS_FILE):
        os.remove(COMMENTS_FILE)  # fresh start: remove the old file

    for number, video in enumerate(videos, start=1):
        if video["video_id"] in finished:
            continue
        comments, quota_hit = get_comments(video["video_id"], MAX_COMMENTS_PER_VIDEO)
        if quota_hit:
            print("Daily quota is used up. Your data is saved.")
            print("Run the same command again tomorrow. It continues from here.")
            return
        append_comments(comments)
        mark_finished(video["video_id"])
        # The video ID is printed, not the title: the Windows console cannot show Arabic.
        print(f"[{number}/{len(videos)}] {len(comments)} comments  {video['video_id']}")

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)  # everything finished, next run starts fresh
    print(f"Done. Saved {len(videos)} videos and {count_comments()} comments.")


def main():
    if not API_KEY:
        sys.exit("Set the YOUTUBE_API_KEY environment variable first.")
    if not CHANNEL:
        sys.exit("Write your channel's handle in channel.json first (see channel.example.json).")
    try:
        run()
    except NetworkDown:
        print("Internet problem. Your data is saved.")
        print("Run the same command again. It continues from where it stopped.")
    except QuotaExceeded:
        print("Daily quota is used up. Run the same command again tomorrow.")


if __name__ == "__main__":
    main()
