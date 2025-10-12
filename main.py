import os
import yaml
import json
import feedparser
import random
import google.generativeai as genai
from datetime import datetime, timezone
from bs4 import BeautifulSoup

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set!")
MODEL_CHOICE = "gemini-2.5-flash-lite"
CONTEXT_WINDOW = 10000
MAX_USERS = 10
TEST_HEADLINE = None
FORCED_ENGAGEMENT = []
HISTORY_FILE = "post_history.json"
MAX_POSTS_TO_DISPLAY = 10
FILTERED_USERS = ["/u/Automoderator", "/u/MajorParadox", "/u/kodiak931156", "/u/AthiestComic"]

RSS_FEEDS = [
    "https://www.reddit.com/r/animenews/.rss",
    "https://www.reddit.com/r/casualconversation/.rss",
    "https://www.reddit.com/r/unpopularopinion/.rss",
    "https://www.reddit.com/r/Showerthoughts/.rss",
    "https://www.reddit.com/r/changemyview/.rss",
    "https://www.reddit.com/r/gamernews/.rss",
    "https://www.reddit.com/r/trueaskreddit/.rss",
    "https://www.reddit.com/r/games/.rss",
    "https://www.reddit.com/r/glitch_in_the_matrix/.rss",
    "https://www.reddit.com/r/advice/.rss",
    "https://www.reddit.com/r/amitheasshole/.rss",
    "https://www.reddit.com/r/relationships/.rss",
    "https://www.reddit.com/r/relationship_advice/.rss",
    "https://www.reddit.com/r/askreddit/.rss"
]

# Forced Feed
#RSS_FEEDS = ["https://www.reddit.com/r/changemyview/.rss"]

# load personas
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

def general_chat():
    print("Here are the available chatters: ")
    chatter_list = []
    for chatter in personas:
        print(chatter["character"])
        chatter_list.append(chatter["character"].lower())
    chatter_choice = input("Which chatter would you like to use? ")
    if chatter_choice.lower() in chatter_list:
        selected_persona = personas[chatter_list.index(chatter_choice.lower())]
        print(f"Here is your chatter: {selected_persona['character']}")
        opening_prompt = ""
        for persona_key in selected_persona:
            opening_prompt = opening_prompt + f"Your {persona_key} is {selected_persona[persona_key]}. "
        print(opening_prompt)
        chat_log = [opening_prompt]
    else:
        print("No such chatter. Using default chatter.")
        chat_log = ["You are an AI assistant."]
    print(f"Type \"history\" to see the history of the current session.")
    print(f"Type \"exit\" to exit the session.")
    while True:
        chat_input = input(f"\nUser: ")
        if chat_input.lower() == "exit":
            break
        if chat_input.lower() == "history":
            for chat in chat_log:
                print(chat)
        else:
            chat_log.append("USER: " + chat_input)
            response = client.generate_content(
                contents=context_budgeter("\n".join(chat_log), CONTEXT_WINDOW, CONTEXT_WINDOW),
            )
            chat_log.append("Assistant: " + response.text)
            print("AI: " + response.text, f"\n")

def get_headline(feed_url):
    print(f"Fetching {feed_url}")
    feed = feedparser.parse(feed_url)

    # Error handling for the feed
    if feed.bozo:
        print(f"Error parsing feed: {feed.bozo_exception}")
        return None
    if not feed.entries:
        print("No entries found in the RSS feed.")
        return None

    for entry in feed.entries:
        if entry.author not in FILTERED_USERS:
            print(f"Found valid headline: \"{entry.title}\" by {entry.author}")

            post_body = ""
            if hasattr(entry, 'content'):
                html_content = entry.content[0].value
                soup = BeautifulSoup(html_content, 'html.parser')
                post_body = soup.get_text(separator='\n', strip=True)
                print("--- Extracted body text ---")

            return {
                "title": entry.title,
                "link": entry.link,
                "body": post_body
            }

    print("Could not find a post not made by AutoModerator in the recent entries.")
    return None

def generate_reddit_comments(post_title, post_body):
    post_content_prompt = f"Here is the headline: \"{post_title}\"\n"
    if post_body:
        post_content_prompt += f"Here is the body of the post:\n---\n{post_body}\n---\n\n"
    else:
        post_content_prompt += "\n"
    reddit_prompt = (
        f"You are an API that generates a simulated Reddit comment section for a given headline. "
        f"Your task is to use the provided personas to create a series of comments and replies in a nested structure. "
        f"The diction and prose should be reflective of modern brain rotted youth."
        f"The final output must be a single, valid JSON object and nothing else. Do not include any explanatory text before or after the JSON. "
        f"The JSON object should be a list of top-level comment objects. "
        f"Each comment object must have the following keys:\n"
        f" - 'author': (string) The name of the persona posting the comment.\n"
        f" - 'comment': (string) The text of the comment.\n"
        f" - 'upvotes': (integer) A randomly generated number of upvotes based on how popular the comment would be in real life, e.g., between -10 and 200.\n"
        f" - 'replies': (list) A list of other comment objects that are replies to this one. This list can be empty.\n\n"
        f"You must not simulate the comments or replies of the original post. Only use the provided personas."
        f"Here are your personas: {personas}\n"
        f"Here is the content: {post_content_prompt}\n"
        f"Generate the JSON output now."
    )

    print("--- Sending Prompt to AI ---")
    print(reddit_prompt)

    response = client.generate_content(
        contents=context_budgeter(reddit_prompt, CONTEXT_WINDOW, CONTEXT_WINDOW),
        generation_config={"response_mime_type": "application/json"}
    )

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
        <title>AI Social Feed</title>
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
    print("\n--- Starting New Post Generation ---")
    update_time_utc = datetime.now(timezone.utc).strftime("%B %d, %Y at %H:%M UTC")

    if TEST_HEADLINE:
        # For testing, we can manually create a body
        post_data = {'title': TEST_HEADLINE, 'link': '#', 'body': 'This is the test body for the post.'}
        print(f"Using test headline: \"{TEST_HEADLINE}\"")
    else:
        post_data = get_headline(random.choice(RSS_FEEDS))

    if post_data:
        comment_section = generate_reddit_comments(post_data["title"], post_data["body"])
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



