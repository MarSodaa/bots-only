import os
import yaml
import json
import feedparser
import random
import google.generativeai as genai
from google.generativeai import types
from datetime import datetime, timezone
from bs4 import BeautifulSoup
import time
import requests
from PIL import Image
import io

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set!")
MODEL_CHOICE = "gemini-2.5-flash-lite"
CONTEXT_WINDOW = 10000
MAX_USERS = 10
TEST_HEADLINE = None
FORCED_ENGAGEMENT = []
HISTORY_FILE = "post_history.json"
MAX_POSTS_TO_DISPLAY = 15
FILTERED_USERS = ["/u/AutoModerator", # Filters out sticky posts by these users
                  "/u/rGamesModBot",
                  "/u/AITAMod"]
NUMBER_OF_NEW_POSTS = 3

RSS_FEEDS = [
    "https://www.reddit.com/r/animenews/.rss",
    "https://www.reddit.com/r/animemes/.rss",
    "https://www.reddit.com/r/animememes/.rss",
    "https://www.reddit.com/r/anime_irl/.rss",
    "https://www.reddit.com/r/casualconversation/.rss",
    "https://www.reddit.com/r/unpopularopinion/.rss",
    "https://www.reddit.com/r/Showerthoughts/.rss",
    "https://www.reddit.com/r/changemyview/.rss",
    "https://www.reddit.com/r/gamernews/.rss",
    "https://www.reddit.com/r/trueaskreddit/.rss",
    "https://www.reddit.com/r/games/.rss",
    "https://www.reddit.com/r/gaming/.rss",
    "https://www.reddit.com/r/glitch_in_the_matrix/.rss",
    "https://www.reddit.com/r/advice/.rss",
    "https://www.reddit.com/r/amitheasshole/.rss",
    "https://www.reddit.com/r/relationships/.rss",
    "https://www.reddit.com/r/relationship_advice/.rss",
    "https://www.reddit.com/r/askreddit/.rss"
]


# Forced Feed
#RSS_FEEDS = ["https://www.reddit.com/r/anime_irl/.rss"]

# load personas
def load_personas():
    try:
        with open("personas.yml", 'r', encoding='utf-8') as ymlfile:
            personas = yaml.safe_load(ymlfile)
    except FileNotFoundError:
        print("No personas.yml found")
        personas = []
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        personas = []
    if personas:
        print(f"Found {len(personas)} personas")

    # Chooses personas
    if len(personas) > MAX_USERS:
        print(f"Total personas available: {len(personas)}. Choosing {MAX_USERS}.")
        forced_personas = [p for p in personas if p['character'] in FORCED_ENGAGEMENT]
        if forced_personas:
            print(f"Forced engagement detected for: {[p['character'] for p in forced_personas]}")
        available_for_random = [p for p in personas if p not in forced_personas]
        num_to_choose = MAX_USERS - len(forced_personas)
        if num_to_choose < 0:
            print(
                f"Warning: More forced users ({len(forced_personas)}) than MAX_USERS ({MAX_USERS}). Selecting only the forced users.")
            personas = forced_personas[:MAX_USERS]
            num_to_choose = 0  # No more to choose
        else:
            randomly_selected = random.sample(available_for_random, num_to_choose)
            personas = forced_personas + randomly_selected

    random.shuffle(personas)
    print("\n--- Final Participants ---")
    for person in personas:
        print(f"Adding {person['character']}")

    return personas


genai.configure(api_key=GEMINI_API_KEY)
client = genai.GenerativeModel(MODEL_CHOICE)


# Keep that API bill down
def context_budgeter(text, max_length, keep_length):
    words = text.split()
    if len(words) > max_length:
        last_words = words[-keep_length:]
        return "[History cut off due to context limit being hit] " + " ".join(last_words)
    else:
        return text


def get_historical_links():
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    historical_links = set()
    for post in history:
        if 'headline' in post and 'link' in post['headline']:
            historical_links.add(post['headline']['link'])
    return historical_links

def get_headline(feed_url):
    print(f"Fetching {feed_url}")
    feed = feedparser.parse(feed_url)

    if feed.bozo:
        print(f"Error parsing feed: {feed.bozo_exception}")
        return None
    if not feed.entries:
        print("No entries found in the RSS feed.")
        return None

    historical_links = get_historical_links()

    checked_post_attempts = 0
    max_checked_post_attempts = 10

    for entry in feed.entries:
        if not entry.author:
            continue
        if entry.author in FILTERED_USERS:
            continue

        checked_post_attempts += 1

        if checked_post_attempts > max_checked_post_attempts:
            print(f"Exceeded {max_checked_post_attempts} attempts to find a new headline among non-filtered posts.")
            break

        if entry.link in historical_links:
            print(f"Skipping previously posted headline: \"{entry.title}\" by {entry.author}")
            continue

        print(f"Found valid and new headline: \"{entry.title}\" by {entry.author}")

        post_body = ""
        image_object = None


        if hasattr(entry, 'content'):
            html_content = entry.content[0].value
            soup = BeautifulSoup(html_content, 'html.parser')
            post_body = soup.get_text(separator='\n', strip=True)
            image_link_tag = soup.find('a', string='[link]')
            if image_link_tag and image_link_tag['href']:
                image_url = image_link_tag['href']
                if any(ext in image_url.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
                    print(f"Found image URL: {image_url}")
                    try:
                        response = requests.get(image_url, timeout=10)
                        response.raise_for_status()

                        img = Image.open(io.BytesIO(response.content))
                        img.thumbnail((768, 768))

                        image_object = img
                        print("Successfully downloaded and resized image.")

                    except requests.exceptions.RequestException as e:
                        print(f"Could not download image: {e}")
                    except Exception as e:
                        print(f"Could not process image: {e}")

        return {
            "title": entry.title,
            "link": entry.link,
            "body": post_body,
            "image_object": image_object
        }

    print(
        "Could not find a new post (not by a filtered user and not previously posted) in the recent entries after several attempts.")
    return None


def generate_reddit_comments(post_title, post_body, image_object=None):
    post_content_prompt = f"Here is the headline: \"{post_title}\"\n"
    if post_body:
        post_content_prompt += f"Here is the body of the post:\n---\n{context_budgeter(post_body, CONTEXT_WINDOW, CONTEXT_WINDOW)}\n---\n\n"
    else:
        post_content_prompt += "\n"

    api_contents = []

    if image_object:
        print("Image object found, adding to prompt.")
        # The SDK handles the PIL Image object automatically. No Part needed.
        api_contents.append(image_object)
        post_content_prompt = "The user has provided an image along with the post title. Analyze the image first, then the text. Your comments MUST reflect that you have seen and understood the image. " + post_content_prompt

    reddit_prompt = (
        f"You are an API that generates a simulated Reddit comment section for a given headline. "
        f"Your final output must be a single, valid JSON object and nothing else. Do not include any explanatory text before or after the JSON. "
        f"The JSON object should be a list of top-level comment objects, each with 'author', 'comment', 'upvotes', and 'replies' keys.\n\n"
        f"Here are your personas: {personas}\n\n"
        f"{post_content_prompt}"
        f"### CRITICAL INSTRUCTION & EXAMPLES ###\n"
        f"The MOST IMPORTANT RULE is that each character MUST demonstrate deep, specific knowledge of the topic in the headline. They must talk as if they are true fans, critics, or experts who are deeply familiar with the subject. They are not just random people on the internet; they are part of that specific community.\n"
        f"- They MUST mention specific characters, plot points, historical context, previous installments (e.g., Season 1 vs Season 2), community memes, or well-known controversies related to the topic.\n"
        f"- The persona's traits should COLOR their expert commentary, not REPLACE it.\n\n"
        f"For example, if the headline is 'One Punch Man Season 3: 6.5 Years Wait for Same Recycled Animation':\n"
        f"**GOOD EXAMPLE (Demonstrates Knowledge):**\n"
        f"\"author\": \"Prodigy_von_Ordelia\", \"comment\": \"Six and a half years for this? After the absolute disaster that was J.C. Staff's handling of Season 2, particularly the metal shine on Genos and the slideshow-level Garou fight, I expected a complete overhaul. To hear it's 'recycled animation' suggests they learned nothing. Unacceptable. The Monster Association arc deserves cinematic quality, not this lazy cash-grab.\"\n"
        f"*(This is good because it mentions the specific animation studio for S2, specific character animation issues, a specific character, and a major story arc.)*\n\n"
        f"**BAD EXAMPLE (Generic, Lacks Knowledge):**\n"
        f"\"author\": \"Prodigy_von_Ordelia\", \"comment\": \"A 6.5-year delay is an unacceptable level of inefficiency. Studios need to be held to a higher standard. This is a fundamental mismanagement of resources.\"\n"
        f"*(This is bad because it could be about any animated show. It contains no specific details about One Punch Man.)*\n"
        f"### END OF CRITICAL INSTRUCTIONS ###\n\n"
        f"Now, generate a full, nested comment section for the provided headline. Remember:\n"
        f"- The diction should be reflective of modern brain rotted online communities.\n"
        f"- Comment length should vary from short single phrase quips to multi-paragraph rants.\n"
        f"- Do not simulate replies from the original post author.\n"
        f"- Ensure the final output is only the JSON object."
    )

    api_contents.insert(0, reddit_prompt)

    print("--- Sending Prompt to AI ---")
    print(reddit_prompt)

    response = client.generate_content(
        contents=api_contents,
        generation_config=types.GenerationConfig(
            max_output_tokens=3000,
            response_mime_type="application/json"))

    try:
        reddit_data = json.loads(response.text)
        print("--- Successfully Parsed JSON Data ---")
        return reddit_data
    except (json.JSONDecodeError, ValueError) as e:
        print(f"\n--- Error: Failed to parse JSON from the AI's response: {e} ---")
        print("Raw AI response was:")
        print(response.text)
        return None


def update_post_history(new_post):
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.insert(0, new_post)

    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    print(f"--- Post history updated. Total posts: {len(history)} ---")
    return history


def format_comment(comment, depth=0):
    author = comment.get('author', 'Anonymous')
    comment_text = comment.get('comment', '[Message has been deleted by moderator]').replace('\n', '<br>')
    upvotes = comment.get('upvotes', 1)
    replies = comment.get('replies', [])
    margin_left = f"margin-left: {depth * 20}px;"

    html = f"""
    <div class="comment" style="{margin_left}">
        <div class="comment-header">
            <span class="author">{author}</span>
            <span class="upvotes">{upvotes} points</span>
        </div>
        <div class="comment-body">
            <p>{comment_text}</p>
        </div>
        <div class="replies">
    """

    for reply in replies:
        html += format_comment(reply, depth + 1)
    html += "</div></div>"

    return html


def format_single_post_html(post_data):
    headline_title = post_data['headline']['title']
    headline_link = post_data['headline']['link']
    generation_time = post_data['timestamp']
    comments_data = post_data['comments']

    comments_html = ""
    if comments_data:
        for comment in comments_data:
            comments_html += format_comment(comment)

    return f"""
    <div class="post-container">
        <div class="headline">
            <h2><a href="{headline_link}" target="_blank">{headline_title}</a></h2>
        </div>
        <div class="timestamp">
            <p>Posted on: {generation_time}</p>
        </div>
        <hr class="post-divider">
        <div class="comments-section">
            {comments_html}
        </div>
    </div>
    """


def generate_feed_html(posts):
    all_posts_html = ""
    for post in posts:
        all_posts_html += format_single_post_html(post)

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Clankernet</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #030303; color: #d7dadc; margin: 0; padding: 20px; }}
            .main-container {{ max-width: 800px; margin: auto; }}
            .page-header {{ text-align: center; color: #fff; border-bottom: 2px solid #343536; padding-bottom: 10px; margin-bottom: 40px; }}
            .post-container {{ background-color: #1a1a1b; border: 1px solid #343536; border-radius: 8px; padding: 20px; margin-bottom: 30px; }}
            .headline h2 {{ font-size: 1.5em; margin-bottom: 5px; }}
            .headline a {{ color: #d7dadc; text-decoration: none; }}
            .headline a:hover {{ text-decoration: underline; color: #4f9eed; }}
            .timestamp p {{ font-size: 0.8em; color: #818384; margin-top: 0; }}
            .post-divider {{ border-color: #343536; }}
            .comment {{ border-left: 2px solid #343536; margin-top: 15px; padding-left: 15px; }}
            .comment-header {{ color: #818384; font-size: 0.8em; margin-bottom: 5px; }}
            .author {{ font-weight: bold; color: #a6cbe7; }}
            .upvotes {{ margin-left: 10px; }}
            .comment-body p {{ margin: 0; line-height: 1.5; }}
        </style>
    </head>
    <body>
        <div class="main-container">
            <div class="page-header">
                <h1>Clankernet</h1>
                <p>Displaying the {len(posts)} most recent posts.</p>
            </div>
            {all_posts_html}
        </div>
    </body>
    </html>
    """
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    print("--- index.html file generated successfully! ---")


if __name__ == "__main__":
    completed_posts = 0
    while completed_posts < NUMBER_OF_NEW_POSTS:
        personas = load_personas()
        print("\n--- Starting New Post Generation ---")
        update_time_utc = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")

        if TEST_HEADLINE:
            post_data = {'title': TEST_HEADLINE, 'link': '#', 'body': '#', 'image_object': None}
            print(f"Using test headline: \"{TEST_HEADLINE}\"")
        else:
            post_data = get_headline(random.choice(RSS_FEEDS))

        if post_data:
            comment_section = generate_reddit_comments(
                post_data["title"],
                post_data["body"],
                post_data.get("image_object")
            )
            if comment_section:
                new_post_data = {
                    "timestamp": update_time_utc,
                    "headline": {
                        "title": post_data['title'],
                        "link": post_data['link']
                    },
                    "comments": comment_section
                }

                full_history = update_post_history(new_post_data)
                posts_to_render = full_history[:MAX_POSTS_TO_DISPLAY]
                generate_feed_html(posts_to_render)
            else:
                print("\n--- Skipped HTML generation due to failure in comment generation. ---")
        else:
            print("\n--- Skipped all generation due to failure in fetching a headline. ---")

        completed_posts += 1
        time.sleep(2)














