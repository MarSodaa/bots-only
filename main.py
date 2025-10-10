import os
import yaml
import json
import feedparser
import random
from google import genai

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set!")
MODEL_CHOICE = "gemini-2.5-flash-lite"
CONTEXT_WINDOW = 10000

RSS_FEEDS = [
    "https://www.reddit.com/r/animenews/.rss"
]

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
client = genai.Client(api_key=GEMINI_API_KEY)
if personas:
    print(f"Found {len(personas)} personas")

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
            response = client.models.generate_content(
                model=MODEL_CHOICE,
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

    headlines = feed.entries[:3]

    # print(headlines)

    return headlines

def generate_reddit_comments(headline):
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
        f"Here are your personas: {personas}\n"
        f"Here is the headline: \"{headline}\"\n\n"
        f"Generate the JSON output now."
    )

    print("--- Sending Prompt to AI ---")

    response = client.models.generate_content(
        model=MODEL_CHOICE,
        contents=context_budgeter(reddit_prompt, CONTEXT_WINDOW, CONTEXT_WINDOW),
    )

    raw_text = response.text
    # print("\n--- AI Raw Response ---")
    # print(raw_text)

    cleaned_text = raw_text.strip().lstrip('```json').rstrip('```').strip()

    try:
        reddit_data = json.loads(cleaned_text)

        #print("\n--- Successfully Parsed JSON Data ---")

        print(json.dumps(reddit_data, indent=2))

        # Or you can access specific parts of it
        # print("\n--- Example: Accessing Data ---")
        # first_comment = reddit_data[0]
        # author = first_comment['author']
        # comment_text = first_comment['comment']
        # print(f"The author of the first comment is: {author}")
        # print(f"Their comment was: '{comment_text}'")

        #if first_comment['replies']:
        #    first_reply = first_comment['replies'][0]
        #    print(f"The first reply was from {first_reply['author']}")

        return reddit_data

    except json.JSONDecodeError:
        print("\n--- Error: Failed to parse JSON from the AI's response. ---")
        print("This usually happens if the AI includes extra text or malformed JSON.")


def format_comment(comment, depth=0):
    """Recursively formats a comment and its replies into HTML."""
    author = comment.get('author', 'Unknown')
    comment_text = comment.get('comment', '').replace('\n', '<br>')
    upvotes = comment.get('upvotes', 0)
    replies = comment.get('replies', [])

    # Indent replies
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

    # Recursively add replies
    for reply in replies:
        html += format_comment(reply, depth + 1)

    html += "</div></div>"
    return html


def generate_html_file(headline_obj, comments_data):
    """Generates a complete index.html file from the data."""
    headline_title = headline_obj['title']
    headline_link = headline_obj['link']

    comments_html = ""
    if comments_data:
        for comment in comments_data:
            comments_html += format_comment(comment)

    print("HELLOO!")

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Social Feed</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #1a1a1b; color: #d7dadc; margin: 0; padding: 20px; }}
            .container {{ max-width: 800px; margin: auto; background-color: #1a1a1b; border-radius: 8px; padding: 20px; }}
            .headline h1 {{ font-size: 1.5em; }}
            .headline a {{ color: #4f9eed; text-decoration: none; }}
            .headline a:hover {{ text-decoration: underline; }}
            .comment {{ border-left: 2px solid #343536; margin-top: 15px; padding-left: 15px; }}
            .comment-header {{ color: #818384; font-size: 0.8em; margin-bottom: 5px; }}
            .author {{ font-weight: bold; color: #a6cbe7; }}
            .upvotes {{ margin-left: 10px; }}
            .comment-body p {{ margin: 0; line-height: 1.5; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="headline">
                <h1><a href="{headline_link}" target="_blank">{headline_title}</a></h1>
            </div>
            <hr style="border-color: #343536;">
            <div class="comments-section">
                {comments_html}
            </div>
        </div>
    </body>
    </html>
    """

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    print("--- index.html file generated successfully! ---")


if __name__ == "__main__":
    headline = get_headline(random.choice(RSS_FEEDS))
    if headline:
        comment_section = generate_reddit_comments(headline[0]["title"])
        if comment_section:
            generate_html_file(headline[0], comment_section)
