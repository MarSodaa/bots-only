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
MAX_POSTS_TO_DISPLAY = 20
FILTERED_USERS = ["/u/AutoModerator", # Filters out sticky posts by these users
                  "/u/rGamesModBot",
                  "/u/AITAMod"]
NUMBER_OF_NEW_POSTS = 1

RSS_FEEDS = [
    "https://www.reddit.com/r/animenews/.rss",
    "https://www.reddit.com/r/animemes/.rss",
    "https://www.reddit.com/r/animememes/.rss",
    "https://www.reddit.com/r/anime_irl/.rss",
    "https://www.reddit.com/r/memes/.rss",
    "https://www.reddit.com/r/dankmemes/.rss",
    "https://www.reddit.com/r/me_irl/.rss",
    "https://www.reddit.com/r/funny/.rss",
    "https://www.reddit.com/r/wholesomememes/.rss",
    "https://www.reddit.com/r/tifu/.rss",
    "https://www.reddit.com/r/jokes/.rss",
    "https://www.reddit.com/r/facepalm/.rss",
    "https://www.reddit.com/r/technicallythetruth/.rss",
    "https://www.reddit.com/r/nottheonion/.rss",
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
#RSS_FEEDS = ["https://www.reddit.com/r/animememes/.rss"]


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
        if not hasattr(entry, 'author'):
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

# In case the llm abruptly ends mid-comment (Thanks, Schnitzel)
def repair_and_parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"\n--- Initial JSON parse failed: {e}. Attempting to repair. ---")

        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        if not text.startswith('['):
            print("--- Repair failed: Text does not start with a list character '['. ---")
            return None

        nesting_level = 0
        last_valid_pos = -1

        for i, char in enumerate(text):
            if char == '{' or char == '[':
                nesting_level += 1
            elif char == '}' or char == ']':
                nesting_level -= 1


            if char == '}' and nesting_level == 1:
                last_valid_pos = i

        if last_valid_pos == -1:
            print("--- Repair failed: Could not find any complete top-level objects in the JSON string. ---")
            return None

        truncated_text = text[:last_valid_pos + 1]

        repaired_json_text = truncated_text + "\n]"

        print("--- Attempting to parse repaired JSON... ---")
        try:
            repaired_data = json.loads(repaired_json_text)
            print(f"--- Successfully salvaged {len(repaired_data)} top-level comments. ---")
            return repaired_data
        except json.JSONDecodeError as e2:
            print(f"--- Final repair attempt failed: {e2} ---")
            print("--- The truncated and repaired JSON was: ---")
            print(repaired_json_text)
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
        f"Remember, any persona can make a short comment. An 'Expert Analyst' isn't limited to long paragraphs; they can also make a cutting, one-phrase joke or observation.\n"
        f"{post_content_prompt}"
        f"CRITICAL INSTRUCTION & EXAMPLES \n"
        f"The Rule of Balance: To ensure a realistic balance, you MUST adhere to a specific mix. At least 40% of the top-level comments must be 'short-form.' A short-form comment is defined as being under 25 words. These are the memes, the one-line zingers, and the gut reactions. The remaining comments can be the longer, detailed analyses. This balance is not optional.\n\n"
        f"The MOST IMPORTANT RULE is that the comment section as a whole MUST feel like it's from a genuine fan community. This means creating a realistic mix of comment types. Some comments should be the detailed analyses you've been getting, but many should be the short, punchy reactions, in-jokes, or memes that define online discussion.\n\n"
        f"All comments, regardless of length, must demonstrate specific knowledge. They must talk as if they are true fans, critics, or experts who are deeply familiar with the subject. The persona's traits should COLOR their commentary, not REPLACE it.\n\n"
        f"How to Demonstrate Knowledge in Different Comment Lengths:\n\n"
        f"1.  Detailed Analysis: Mention specific characters, plot points, historical context, previous installments (e.g., Season 1 vs Season 2), or well-known controversies.\n"
        f"2.  Short & Punchy Quips: Use community-specific memes, nicknames, or references to well-known moments that another fan would instantly understand.\n\n"
        f"For example, if the headline is 'One Punch Man Season 3: 6.5 Years Wait for Same Recycled Animation':\n\n"
        f"GOOD EXAMPLE (Detailed Analysis):\n"
        f"\"author\": \"Prodigy_von_Ordelia\", \"comment\": \"Six and a half years for this? After the absolute disaster that was J.C. Staff's handling of Season 2, particularly the metal shine on Genos and the slideshow-level Garou fight, I expected a complete overhaul. To hear it's 'recycled animation' suggests they learned nothing. Unacceptable. The Monster Association arc deserves cinematic quality, not this lazy cash-grab.\"\n"
        f"(This is good because it's a detailed rant mentioning the specific studio, character animation issues, a specific character, and a major story arc.)\n\n"
        f"GOOD EXAMPLE (Short & Knowledgeable):\n"
        f"\"author\": \"ChadThunderclap\", \"comment\": \"JC Staff and their damn metal shine, name a more iconic duo. I'll wait.\"\n"
        f"(This is also good. It's short but demonstrates knowledge of the specific, widely-criticized 'metal shine' animation from Season 2 and the studio responsible.)\n\n"
        f"ANOTHER GOOD SHORT EXAMPLE:\n"
        f"\"author\": \"PixelProwler\", \"comment\": \"So we're getting Garou vs. PowerPoint again? Cool cool cool.\"\n"
        f"(This is good because it uses a meme format to reference the specific 'slideshow' feel of a major fight in the previous season.)\n\n"
        f"BAD EXAMPLE (Generic, Lacks Knowledge):\n"
        f"\"author\": \"Prodigy_von_Ordelia\", \"comment\": \"A 6.5-year delay is an unacceptable level of inefficiency. Studios need to be held to a higher standard. This is a fundamental mismanagement of resources.\"\n"
        f"(This is bad because it could be about any animated show. It contains no specific details about One Punch Man.)\n\n"
        f"END OF CRITICAL INSTRUCTIONS \n\n"
        f"FINAL CHECKLIST BEFORE GENERATING \n"
        f"- Comment Length Variety: Is there a healthy mix of long and short comments? Did I meet the 40% short-form comment rule?\n"
        f"- Knowledge Depth: Do even the shortest comments contain a specific reference, nickname, or piece of in-community knowledge?\n"
        f"- Overall Vibe: Does this feel like a real, chaotic, and diverse fan forum, not just a collection of essays?\n\n"
        f"Now, generate a full, nested comment section for the provided headline. Remember:\n"
        f"- The diction should be reflective of modern brain rotted online communities.\n"
        f"- Do not use markdown formatting.\n"
        f"- Comment length should vary from short single phrase quips to multi-paragraph rants.\n"
        f"- Do not simulate replies from the original post author. Only use the provided fictional personas.\n"
        f"- The soul of this comment section is the personas contributing their takes or jokes and bouncing off of each other reacting to one another and replying to one another. There should be a lot of them replying to one another.\n"
        f"- Ensure the final output is only the JSON object."
    )

    api_contents.insert(0, reddit_prompt)

    print("--- Sending Prompt to AI ---")
    print(reddit_prompt)

    response = client.generate_content(
        contents=api_contents,
        generation_config=types.GenerationConfig(
        temperature=2.0,
        top_p=0.95,
            max_output_tokens=3000,
            response_mime_type="application/json"))

    reddit_data = repair_and_parse_json(response.text)

    if reddit_data:
        print("--- Successfully Parsed JSON Data ---")
        return reddit_data
    else:
        print(f"\n--- Error: Failed to parse JSON, and repair attempt was unsuccessful. ---")
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
    margin_left = f"margin-left: {depth * 5}px;"

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
    post_body = post_data['headline'].get('body')
    generation_time = post_data['timestamp']
    comments_data = post_data['comments']

    body_preview_html = ""
    if post_body:
        if len(post_body) > 100:
            preview_text = (post_body[:400] + '...') if len(post_body) > 400 else post_body
            body_preview_html = f'<div class="post-body-preview"><p>{preview_text.replace(chr(10), "<br>")}</p></div>'

    comments_html = ""
    if comments_data:
        first_comment = comments_data[0]
        comments_html += format_comment(first_comment)

        if len(comments_data) > 1:
            rest_of_comments = comments_data[1:]

            comments_html += f"""
            <button class="toggle-comments-btn" onclick="toggleComments(this)">
                Show {len(rest_of_comments)} More Comments
            </button>
            """

            rest_of_comments_html = ""
            for comment in rest_of_comments:
                rest_of_comments_html += format_comment(comment)

            comments_html += f'<div class="collapsible-comments collapsed">{rest_of_comments_html}</div>'

    return f"""
    <div class="post-container">
        <div class="headline">
            <h2><a href="{headline_link}" target="_blank">{headline_title}</a></h2>
        </div>
        {body_preview_html}
        <div class="timestamp">
            <p>Posted on: {generation_time}</p>
        </div>
        <hr class="post-divider">
        <div class="comments-section">
            {comments_html if comments_html else "<p style='color: #818384; font-style: italic;'>No comments yet.</p>"}
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

            /* --- NEW STYLES FOR COLLAPSIBLE COMMENTS --- */
            .toggle-comments-btn {{
                background-color: transparent;
                border: 1px solid #343536;
                color: #818384;
                padding: 5px 10px;
                margin-top: 15px;
                margin-left: 15px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 0.8em;
                font-weight: bold;
            }}
            .toggle-comments-btn:hover {{
                border-color: #818384;
                color: #d7dadc;
            }}
            .collapsible-comments.collapsed {{
                display: none;
            }}
        </style>
        <script>
            function toggleComments(button) {{
                // Find the div containing the rest of the comments, which is the next element after the button
                const collapsibleSection = button.nextElementSibling;
                if (collapsibleSection) {{
                    // Toggle the 'collapsed' class to show/hide it
                    collapsibleSection.classList.toggle('collapsed');

                    // Update the button text based on the new state
                    const isCollapsed = collapsibleSection.classList.contains('collapsed');
                    if (isCollapsed) {{
                        // To get the count, we count the number of direct child comments
                        const commentCount = collapsibleSection.children.length;
                        button.textContent = `Show ${{commentCount}} More Comments`;
                    }} else {{
                        button.textContent = 'Hide Comments';
                    }}
                }}
            }}
        </script>
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
                        "body": post_data['body'],
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


