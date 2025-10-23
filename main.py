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
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set!")
MODEL_CHOICE = "gemini-2.5-flash-lite"
CONTEXT_WINDOW = 10000
MAX_USERS = 10
TEST_HEADLINE = ""
FORCED_ENGAGEMENT = []
HISTORY_FILE = "post_history.json"
MAX_POSTS_TO_DISPLAY = 200
FILTERED_USERS = ["/u/AutoModerator",  # Filters out sticky posts by these users
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

YOUTH_SLANG = {
    "ahh": "Substitute for the word ass in humourous instances like goofy ahh or ts so ahh.",
    "alpha": "A man that is successful, sexually attractive, and is non-conforming to existing social norms. Used predominantly by the manosphere.",
    "aura": "Overall vibe, energy, or personality.",
    "based": "Expressing approval of someone or agreeing with someone's opinion. Similar to W. It originates from The BasedGod.",
    "basic": "Pertaining to those who prefer mainstream products, trends, and music. Derived from the term 'basic bitch'.",
    "bar(s)": "A lyric in a rap song that is considered excellent.",
    "beta": "A man who is neither alpha nor sigma. Seen as below both groups.",
    "BDE": "Abbreviation for 'big dick energy': confidence and ease.",
    "beige flag": "A behavior or personality trait that is neither good nor bad. See red flag.",
    "bestie": "Abbreviation for 'best friend'. Sometimes used humorously for someone the speaker/writer has no relationship with.",
    "bet": "Agreeing to something; yes; okay; sure.",
    "big yikes": "Used to describe something embarrassing or cringe, particularly in response to an offensive comment.",
    "bop": "(1) A derogatory term, usually for females, suggesting excessive flirtatiousness or promiscuity. (2) An exceptionally good song. (3) An acronym for 'baddie on point', meaning someone who uses their appearance to make money.",
    "brainrot": "The state of losing touch with the real world as a result of consuming hyper-stimulating or chronically online content, especially when characterized by online buzzwords (e.g., 'skibidi', 'fanum tax', 'rizz').",
    "bro": "Shortened version of brother used as a third person pronoun.",
    "bruh": "Used to express shock, embarrassment, or disappointment.",
    "bugging": "See tweaking for more context.",
    "bussin'": "Extremely good, excellent. Originated from African-American vernacular for delicious food.",
    "bussy": "Portmanteau of 'boy' and 'pussy' (slang for the vagina). Effectively a man's anus.",
    "cap": "To lie. See no cap.",
    "caught in 4K": "Refers to someone being indisputably caught doing something wrong or incriminating on camera or with evidence to prove it, referencing 4K resolution.",
    "chopped": "Ugly or unattractive.",
    "clanker": "Slur for robots, primarily used against generative artificial intelligence. Originated from Star Wars media.",
    "clapback": "Swift and witty response to an insult or critique.",
    "cook": "To 'cook' is to perform or do well. In contrast, when a person is 'cooked', they are in trouble.",
    "cracked": "To be skilled at something. Alternatively, 'getting cracked' means to have sex.",
    "crash out": "To make a reckless or regrettable decision after a bout of rage or upset.",
    "clock": "To belittle or silence someone, often in a manner that is intended to embarrass or undermine their confidence, similar to 'gag'.",
    "dank": "Excellent, high-quality.",
    "dead/ded": "Humorous to such an extent as to 'kill you'.",
    "delusionship": "A relationship in which someone holds unrealistic or overly idealistic beliefs. A person who holds such beliefs is considered 'delulu'.",
    "diddle": "To sexually assault or molest someone.",
    "dih": "Ironic Algospeak for dick. Usually used with the wilting flower emoji (🥀).",
    "drip": "Trendy high-class fashion.",
    "edge": "To maintain a high level of sexual arousal for an extended period without reaching climax (orgasm).",
    "eeffoc": "Coffee spelled backwards. Used in the context of 'not caring' or 'not giving a damn/fuck' (until I've had my coffee, I don't give eeffoc about anything).",
    "face card": "An attractive face. Sometimes defined as never declining or receding.",
    "fanum tax": "Theft of food between friends.",
    "finna": "Short for 'fixing to'. Often used interchangeably with 'gonna'.",
    "fire": "Term used to describe that something is impressive, good, or cool. Also see lit. Alternative: flame.",
    "fit/fit check": "Term used to highlight or bring attention to one's outfit or fashion. 'Fit' is a truncation of 'outfit'.",
    "41 (fourty-one)": "A nonsense word which originated from '41 Song (Saks Freestyle)' by rapper Blizzi Boi. A parallel meme to 6-7.",
    "gagged": "Shocked, amazed, or at a loss for words.",
    "gas": "To describe something as highly entertaining, pleasant, or good. See slaps.",
    "ghost": "To end communication or contact with someone without warning.",
    "glaze": "To hype, praise, or compliment someone so much that it becomes annoying or excessive.",
    "glizzy": "A hot dog. Popularized in 2020.",
    "glow-up": "A major improvement in one's self, usually an improvement in appearance, confidence, personality, and style. A 'glow-down' is a situation where someone's appearance has declined.",
    "GOAT": "Acronym for 'greatest of all time'.",
    "good boy/good girl": "A phrase that's mockingly used when one is told to do something and they do it.",
    "gooning": "The act of masturbating for long periods of time or for someone who does it chronically.",
    "green flag": "Behaviors or personality traits that are considered positive, healthy, or desirable. See red flag.",
    "Gucci": "Meaning good, cool, fashionable, or excellent. Used to express approval or satisfaction for something.",
    "gyatt": "Someone with large buttocks or an hourglass figure.",
    "grape": "An algospeak term for rape typically used online to bypass automatic filters.",
    "hawk tuah": "An onomatopoeia for spitting or expectoration on a penis as a form of oral sex.",
    "hb/hg": "An initialism of homeboy/homegirl. Slang used to refer to one's friends.",
    "hit different": "To be better in a distinctive manner.",
    "huzz": "A variation of the pejorative word 'hoes' similarly used to objectify, degrade, and/or belittle women.",
    "ick": "A sudden feeling of disgust or repulsion for something one was previously attracted to.",
    "icl": "Abbreviation of 'I can't lie'. Often used alongside ts and pmo.",
    "IJBOL": "An acronym for 'I just burst out laughing'.",
    "I oop": "Used to express shock, embarrassment, and/or amusement.",
    "iPad kid": "Term describing Generation Alpha children who spend most of their time consuming content via a phone or tablet screen.",
    "it's giving": "Used to describe an attitude or connotation.",
    "it's joever": "Replacement for 'it's over', standing for complete physical and mental defeat.",
    "iykyk": "Acronym for 'If you know, you know'. Used to describe inside jokes.",
    "jit": "An African-American term often used to describe an inexperienced or young individual.",
    "Karen": "Pejorative term for an obnoxious, angry, or entitled (usually but not exclusively white and middle-aged) woman. Also male Karen to denote a man of the same personality type.",
    "L": "Short for 'loss,' used to indicate failure, defeat, or something negative. Often contrasted with 'W' (win).",
    "lit": "Remarkable, interesting, fun, or amusing.",
    "locked in": "A state of total concentration on a task. Similar to flow state.",
    "looksmaxxing": "An attempt (often pseudoscientific) to maximize physical attractiveness.",
    "lore": "Backstory.",
    "main character (MC)": "Someone who is or wants to be the star of their life. Often refers to someone who strives to be the center of attention.",
    "mew": "A pseudoscientific method to restructure someone's jawline by pressing their tongue to the roof of their mouth. See looksmaxxing.",
    "mid": "Average, mediocre, not bad or not special. Sometimes used in a negative or insulting way.",
    "mog": "To look significantly more attractive than someone or something, causing them to appear inferior in comparison. Derived from AMOG.",
    "moot(s)": "Short for 'mutuals' or 'mutual followers'.",
    "Netflix and chill": "To engage in sexual activity, possibly during or after watching a movie or a TV series together.",
    "no cap": "'This is true'; 'I'm not lying'. See cap.",
    "Ohio": "Internet slang that refers to surreal and random phenomena that supposedly occur in Ohio. See Florida man.",
    "OK boomer": "Pejorative directed toward members of the Baby Boomer generation, used to dismiss or mock attitudes typically associated with baby boomers as out of date.",
    "oof": "Used to express discomfort, surprise, dismay, or sympathy for someone else's pain. It also is the sound of a Roblox avatar when it dies or respawns.",
    "oomf": "Abbreviation for 'One of My Followers' or 'One of My Friends'.",
    "opp": "Short for opposition or enemies; describes an individual's opponents. A secondary, older definition has the term be short for 'other peoples' pussy'.",
    "out of pocket": "To act (or say something) crazy, wild, unexpected, or extreme, sometimes to an extent that is considered too far.",
    "owned": "Used to refer to defeat in a video game, or domination of an opposition.",
    "periodt": "Used as an interjection to indicate that the preceding statement is final and that there is nothing more to be said about it.",
    "pick-me": "Someone who seeks validation by trying to stand out, often putting down others in their gender or group to gain favor or attention.",
    "pmo": "An acronym that stands for 'piss me off', used to express discontent or anger at a certain topic. Often utilized alongside ts and icl.",
    "pookie": "An endearing nickname for a close friend or lover.",
    "pushing P": "A phrase meaning acting with integrity and style while maintaining and displaying one’s success. The P in the phrase is most often interpreted as standing for the slang word 'player'.",
    "queen": "A person (usually female) deemed impressive or praiseworthy.",
    "ratio": "When a post, particularly on X (Twitter), receives more replies than retweets and likes combined, or when a reply has better reception than the original post.",
    "rage-bait": "To elicit rage within an individual or group. Usually for an increase in web traffic, or personal enjoyment.",
    "red flag": "A warning sign indicating behaviors or characteristics within a relationship that may potentially be harmful or toxic.",
    "rizz": "One's charm/seduction skills. Derived from charisma.",
    "Roman Empire": "A random event, person, incident, or thing that fascinates or intrigues one to the point that one is frequently thinking about it.",
    "salty": "Used to describe someone who is behaving or expressing themselves in a resentful, bitter, or irritated manner.",
    "SDIYBT": "Acronym for 'Start digging in your butt, twin'. Used ironically.",
    "serving cunt": "To behave in a bold, confident, feminine manner.",
    "sheesh": "To praise someone when they are doing something good. The vowels are often emphasized, as in 'sheeesh'.",
    "shook": "To be shocked, surprised, or bothered.",
    "sigma/sigma male": "A person that is individualistic, self-reliant, successful, and is non-conforming to existing social norms. Can also mean something that is good.",
    "simp": "Sycophancy, being overly affectionate in pursuit of a sexual relationship.",
    "situationship": "Refers to an ambiguous romantic relationship in which both parties have feelings for one another, but said feelings are not clearly defined: a mid-point between dating and not dating.",
    "six-seven (6-7)": "A nonsense word derived from the song Doot Doot (6 7) by Skrilla. Inspired the spinoff meme 41.",
    "skibidi": "Adjective that derives from the meme Skibidi toilet, with no real meaning.",
    "skill issue": "Refers to a situation where a person's lack of ability or proficiency is seen as the cause of their failure or difficulty in completing a task.",
    "sksksk": "Used to convey happiness/laughter. A form of keysmashing.",
    "slaps": "Used to refer to something that is perceived to be good, particularly used when referring to music.",
    "slay": "To do something well.",
    "snatched": "Amazing, attractive, or flawlessly styled.",
    "stan": "Supporting something to an extreme degree. Specifically used in cases of overzealous or obsessive support of celebrities.",
    "sus": "Short term for suspect/suspicious. Popularized by players of the online video game Among Us.",
    "sussy baka": "A combination of 'sus' and 'baka', the Japanese word for 'fool'.",
    "sybau": "Acronym for 'shut your bitch ass up'.",
    "syfm": "Acronym for 'shut your fucking mouth'.",
    "tea": "Secret information or rumors. 'Spilling the tea' means to share gossip or rumors.",
    "touch grass": "A way of telling someone to 'go outside', usually after said person is believed to have been online for too long.",
    "ts": "An abbreviation for 'this shit', or just 'this'. Often used alongside pmo and icl.",
    "tuff": "Eye dialect spelling of tough.",
    "tweaking": "To be acting strangely or thinking that someone is hallucinating.",
    "twin": "A term of endearment for a close friend, indicating a strong, sibling-like bond.",
    "unalive": "A euphemism for the word 'kill' or other death-related terms, often in the context of suicide. Used to circumvent social media algorithms.",
    "unc": "Abbreviation of uncle. Used in a mocking manner to refer to someone who is old or acting old.",
    "understood the assignment": "To understand what was supposed to be done; to do something well.",
    "uwu": "Used to portray happiness or one wanting to appear cute. Used more or less as an expression.",
    "vibe check": "To check one's personality, behavior, or attitude.",
    "vro": "Genderless synonym for bro.",
    "W": "Short for 'win,' used to indicate success, victory, or something positive. Often contrasted with 'L' (loss).",
    "who is this diva?": "An affectionate rhetorical question used to compliment people who positively embody diva-like qualities such as boldness, style, and/or confidence.",
    "yap": "To talk too much, especially without significant meaning.",
    "yeet": "To throw something with force and without regard. Also used as a generic positive exclamation.",
    "You good?": "A short hand of the usual 'Are you okay?' greeting, and is generally used to express concern for an acquaintance's well-being.",
    "zesty": "Flamboyant, effeminate, or otherwise using the stereotypical mannerisms of a gay man."
}
SLANG_NUM = 20

# Choose slang
keys = list(YOUTH_SLANG.keys())
keys_to_keep = random.sample(keys, SLANG_NUM)
YOUTH_SLANG = {key: YOUTH_SLANG[key] for key in keys_to_keep}
print(f"Using slang: {keys_to_keep}")

# Load the sentence transformer model once at the start of the script
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


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


def repair_and_parse_json(text: str):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:-3]
    elif text.startswith("```"):
        text = text[3:-3]
    text = text.strip()

    try:
        data = json.loads(text)
        return _clean_parsed_json(data)
    except json.JSONDecodeError as e:
        print(f"\n--- Initial JSON parse failed: {e}. Attempting repairs. ---")

    repaired_text = text.replace('\\"', '"')
    try:
        print("--- Trying to parse again after cleaning escaped quotes... ---")
        data = json.loads(repaired_text)
        return _clean_parsed_json(data)
    except json.JSONDecodeError as e2:
        print(f"--- Cleaning escaped quotes did not fix the issue: {e2}. ---")

    if not text.startswith('['):
        print("--- Repair failed: Text does not start with a list character '['. ---")
        return None

    last_valid_pos = text.rfind('}')
    if last_valid_pos == -1:
        print("--- Repair failed: Could not find any closing brace in the JSON string. ---")
        return None

    truncated_text = text[:last_valid_pos + 1]
    repaired_json_text = truncated_text + "\n]"

    print("--- Attempting to parse repaired truncated JSON... ---")
    try:
        repaired_data = json.loads(repaired_json_text)
        print(f"--- Successfully salvaged {len(repaired_data)} top-level comments from truncated output. ---")
        return _clean_parsed_json(repaired_data)
    except json.JSONDecodeError as e3:
        print(f"--- Final repair attempt failed: {e3} ---")
        print("--- The truncated and repaired JSON was: ---")
        print(repaired_json_text)
        return None


def _clean_parsed_json(obj):
    if isinstance(obj, dict):
        return {k: _clean_parsed_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_parsed_json(elem) for elem in obj]
    if isinstance(obj, str):
        return obj.replace('*', '').replace('\\', '')
    return obj


def generate_reddit_comments(post_title, post_body, image_object=None):
    post_content_prompt = f"Here is the headline: \"{post_title}\"\n"
    if post_body:
        post_content_prompt += f"Here is the body of the post:\n---\n{context_budgeter(post_body, CONTEXT_WINDOW, CONTEXT_WINDOW)}\n---\n\n"
    else:
        post_content_prompt += "\n"

    api_contents = []

    if image_object:
        print("Image object found, adding to prompt.")
        api_contents.append(image_object)
        post_content_prompt = "The user has provided an image along with the post title. Analyze the image first, then the text. Your comments MUST reflect that you have seen and understood the image. " + post_content_prompt

    reddit_prompt = (
        f"You are an API that generates a simulated Reddit comment section for a given headline. "
        f"Your final output must be a single, valid JSON object and nothing else. Do not include any explanatory text before or after the JSON. "
        f"The JSON object should be a list of top-level comment objects, each with 'author', 'comment', 'upvotes', and 'replies' keys. The 'replies' key contains a list of nested comment objects.\n\n"
        f"Here are your personas: {personas}\n\n"
        f"You should incorporate some of the following slang terms across many of the comments even if it is out of character in order to create a more realistic online environment: {YOUTH_SLANG} \n"
        f"Remember, any persona can make a short comment. An 'Expert Analyst' isn't limited to long paragraphs; they can also make a cutting, one-phrase joke or observation.\n"
        f"{post_content_prompt}"
        f"CRITICAL INSTRUCTIONS & EXAMPLES \n"
        f"You must follow TWO primary rules to create a realistic comment section:\n\n"
        f"RULE 1: The Rule of Balance (CRITICAL: MINIMUM 40% SHORT-FORM). To ensure realistic variety, you MUST adhere to a specific mix. At least 40% of the total comments (including replies) MUST be 'short-form.' A short-form comment is strictly defined as UNDER 15 WORDS, often a single sentence, phrase, emoji-laden response, or even just a few words. These are the memes, the one-line zingers, quick reactions, and gut feelings. This balance is not optional.\n\n"
        f"RULE 2: The Rule of Conversation. The primary goal is to simulate a conversation, not a list of disconnected statements. Therefore, you MUST create deep comment threads. At least 50% of the personas used MUST reply to another comment rather than creating a new top-level comment. A flat list of many top-level comments with no replies is a FAILED generation.\n\n"
        f"All comments, regardless of length or depth, must demonstrate specific knowledge. They must talk as if they are true fans, critics, or experts who are deeply familiar with the subject. For very short comments, this 'knowledge' can be conveyed through specific slang, inside jokes, character/lore nicknames, or an informed, immediate emotional reaction that only a true fan would have. The persona's traits should COLOR their commentary, not REPLACE it.\n\n"
        f"--- EXAMPLES OF GOOD COMMENT *CONTENT* ---\n"
        f"For example, if the headline is 'One Punch Man Season 3: 6.5 Years Wait for Same Recycled Animation':\n\n"
        f"GOOD (Detailed): \"author\": \"Prodigy_von_Ordelia\", \"comment\": \"Six and a half years for this? After the disaster of J.C. Staff's handling of S2, particularly the metal shine on Genos and the slideshow-level Garou fight, I expected a complete overhaul. To hear it's 'recycled animation' suggests they learned nothing. Unacceptable.\"\n"
        f"GOOD (Short & Knowledgeable): \"author\": \"ChadThunderclap\", \"comment\": \"JC Staff and their damn metal shine, name a more iconic duo. I'll wait.\"\n"
        f"BAD (Generic): \"author\": \"Prodigy_von_Ordelia\", \"comment\": \"A 6.5-year delay is unacceptable. Studios need to be held to a higher standard.\"\n\n"
        f"--- EXAMPLE OF GOOD COMMENT *STRUCTURE* (Following the Rule of Conversation) ---\n"
        f"This shows how personas should reply to each other in a nested thread:\n"
        f"```json\n"
        f"[\n"
        f'  {{\n'
        f'    "author": "PixelProwler",\n'
        f'    "comment": "OMG, the particle effects on the sword are insane.",\n'
        f'    "upvotes": 128,\n'
        f'    "replies": [\n'
        f'      {{\n'
        f'        "author": "Prodigy_von_Ordelia",\n'
        f'        "comment": "Incredible? The design is derivative of every dark fantasy trope from the last decade. The armor is impractical and the particle effects will just obscure the telegraphing for his attacks. It\'s style over substance.",\n'
        f'        "upvotes": 45,\n'
        f'        "replies": [\n'
        f'          {{\n'
        f'            "author": "ChadThunderclap",\n'
        f'            "comment": "lol nerd. Big sword go brrrr.",\n'
        f'            "upvotes": 250,\n'
        f'            "replies": []\n'
        f'          }}\n'
        f'        ]\n'
        f'      }}\n'
        f'    ]\n'
        f'  }}\n'
        f']\n'
        f"```\n"
        f"END OF CRITICAL INSTRUCTIONS \n\n"
        f"FINAL CHECKLIST BEFORE GENERATING \n"
        f"- Conversational Depth: Does the output feel like a conversation? Are there multiple deep comment threads (2+ replies deep)? Did I follow the Rule of Conversation, ensuring at least half the personas are replying?\n"
        f"- Comment Length Variety: Is there a healthy mix of long and short comments? Did I meet the 40% short-form rule (UNDER 15 WORDS)?\n"
        f"- Knowledge Depth: Do even the shortest comments contain a specific reference, nickname, or piece of in-community knowledge?\n"
        f"- Overall Vibe: Does this feel like a real, chaotic, and diverse fan forum, not just a collection of essays?\n\n"
        f"Now, generate a full, nested comment section for the provided headline. Remember:\n"
        f"- The diction should be reflective of modern brain rotted online communities heavily favoring short, fragmented sentences, single-phrase quips, slang, and emojis for many comments,** alongside longer, more detailed discussions.\n"
        f"- Do not use markdown formatting in the final comment text.\n"
        f"- Comment length should vary from short single phrase quips to multi-paragraph rants.\n"
        f"- Do not simulate replies from the original post author. Only use the provided fictional personas.\n"
        f"- The PRIMARY GOAL is to create conversation threads where personas react and reply to one another. A long list of un-replied, top-level comments is a failure.\n"
        f"- Ensure the final output is only the JSON object."
    )

    api_contents.insert(0, reddit_prompt)

    print("--- Sending Prompt to AI ---")

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


def format_single_post_html(post_data, topic_id=None):
    headline_title = post_data['headline']['title']
    headline_link = post_data['headline']['link']
    post_body = post_data['headline'].get('body')
    generation_time = post_data['timestamp']
    comments_data = post_data['comments']
    topic_data_attribute = f"data-topic-id='topic-{topic_id}'" if topic_id is not None else ""

    body_preview_html = ""
    if post_body:
        if len(post_body) > 100:
            preview_text = (post_body[:400] + '...') if len(post_body) > 400 else post_body
            body_preview_html = f'<div class="post-body-preview"><p>{preview_text.replace(chr(10), "<br>")}</p></div>'

    comments_html = ""
    if comments_data:
        if len(comments_data) > 0:
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
        <div class="post-container" {topic_data_attribute}>
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

def get_trending_topics(posts_to_analyze):
    if not posts_to_analyze:
        return []

    print(f"\n--- Clustering {len(posts_to_analyze)} headlines to find topics ---")
    headline_titles = [post['headline']['title'] for post in posts_to_analyze]
    headline_embeddings = embedding_model.encode(headline_titles, convert_to_tensor=False)

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5)
    clustering.fit(headline_embeddings)

    clusters = {}
    post_to_cluster_map = {i: label for i, label in enumerate(clustering.labels_)}

    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)

    trending_topics = []
    for label, indices in clusters.items():
        if len(indices) > 1:
            cluster_embeddings = headline_embeddings[np.array(indices)]
            centroid = np.mean(cluster_embeddings, axis=0)

            similarities = cosine_similarity(cluster_embeddings, [centroid])
            most_representative_index_in_cluster = np.argmax(similarities)
            original_post_index = indices[most_representative_index_in_cluster]

            topic_name = headline_titles[original_post_index]
            topic_size = len(indices)
            trending_topics.append({"name": topic_name, "count": topic_size, "id": label})

    sorted_topics = sorted(trending_topics, key=lambda x: x['count'], reverse=True)
    print(f"--- Found {len(sorted_topics)} trending topics. ---")
    return sorted_topics, post_to_cluster_map


def generate_feed_html(posts, full_post_history):
    trending_topics, post_to_cluster_map = get_trending_topics(full_post_history)

    all_posts_html = ""
    for i, post in enumerate(posts):
        topic_id = post_to_cluster_map.get(i)
        all_posts_html += format_single_post_html(post, topic_id)

    trending_list_html = "<ul>"
    trending_list_html += '<li><a href="#" class="topic-link" onclick="showAllPosts(event)"><strong>Show All Posts</strong></a></li>'
    if trending_topics:
        for topic in trending_topics:
            trending_list_html += f"""
                <li>
                    <a href="#" class="topic-link" onclick="filterByTopic('topic-{topic['id']}', event)">
                        "{topic['name']}" ({topic['count']} posts)
                    </a>
                </li>
            """
    else:
        trending_list_html += "<li>No trending topics yet.</li>"
    trending_list_html += "</ul>"

    trending_html = f"""
        <h3 class="collapsible-header" onclick="toggleTopicList(this)">Post Clusters &#9662;</h3>
        <div id="topic-list-content" class="collapsible-content collapsed">
            {trending_list_html}
        </div>
    """

    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Clankernet</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #030303; color: #d7dadc; margin: 0; }}
            .content-area {{ max-width: 900px; margin: auto; padding: 20px; }}
            .page-header {{ text-align: center; color: #fff; border-bottom: 2px solid #343536; padding-bottom: 10px; margin-bottom: 40px; }}
            .trending-topics-container {{ background-color: #1a1a1b; border: 1px solid #343536; border-radius: 8px; padding: 20px; margin-bottom: 30px; }}

            .collapsible-header {{
                margin-top: 0;
                cursor: pointer;
                user-select: none; /* Prevents text selection on click */
            }}
            .collapsible-header:hover {{
                color: #fff;
            }}
            /* MODIFICATION END */

            .trending-topics-container ul {{ padding-left: 0; list-style-type: none; margin-bottom: 0; }}
            .trending-topics-container li {{ margin-bottom: 8px; }}
            .topic-link {{
                color: #a6cbe7;
                text-decoration: none;
                transition: color 0.2s;
            }}
            .topic-link:hover {{
                color: #d7dadc;
                text-decoration: underline;
            }}
            .topic-link.active {{
                font-weight: bold;
                color: #fff;
            }}

            .main-content {{ width: 100%; }}
            .post-container {{ background-color: #1a1a1b; border: 1px solid #343536; border-radius: 8px; padding: 20px; margin-bottom: 30px; }}
            .headline h2 {{ font-size: 1.5em; margin-bottom: 5px; }}
            .headline a {{ color: #d7dadc; text-decoration: none; }}
            .headline a:hover {{ text-decoration: underline; color: #4f9eed; }}
            .timestamp p {{ font-size: 0.8em; color: #818384; margin-top: 0; }}
            .post-body-preview {{ margin-top: 10px; padding: 10px; background-color: #242425; border-radius: 4px; }}
            .post-body-preview p {{ margin: 0; font-size: 0.95em; color: #c0c0c0; }}
            .post-divider {{ border-color: #343536; }}
            .comments-section {{ margin-top: 20px; }}
            .comment {{ border-left: 2px solid #343536; margin-top: 15px; padding-left: 15px; }}
            .comment-header {{ color: #818384; font-size: 0.8em; margin-bottom: 5px; }}
            .author {{ font-weight: bold; color: #a6cbe7; }}
            .upvotes {{ margin-left: 10px; }}
            .comment-body p {{ margin: 0; line-height: 1.5; }}

            .toggle-comments-btn {{ background-color: transparent; border: 1px solid #343536; color: #818384; padding: 5px 10px; margin-top: 15px; margin-left: 15px; border-radius: 4px; cursor: pointer; font-size: 0.8em; font-weight: bold; }}
            .toggle-comments-btn:hover {{ border-color: #818384; color: #d7dadc; }}
            .collapsible-comments.collapsed, .collapsible-content.collapsed {{ display: none; }} /* Consolidated the collapsed style */
        </style>
        <script>
            function toggleComments(button) {{
                const collapsibleSection = button.nextElementSibling;
                if (collapsibleSection) {{
                    collapsibleSection.classList.toggle('collapsed');
                    const isCollapsed = collapsibleSection.classList.contains('collapsed');
                    if (isCollapsed) {{
                        const commentCount = collapsibleSection.children.length;
                        button.textContent = `Show ${{commentCount}} More Comments`;
                    }} else {{
                        button.textContent = 'Hide Comments';
                    }}
                }}
            }}

            function toggleTopicList(header) {{
                const content = header.nextElementSibling;
                const isCollapsed = content.classList.toggle('collapsed');
                if (isCollapsed) {{
                    header.innerHTML = 'Post Clusters &#9662;'; // Down arrow
                }} else {{
                    header.innerHTML = 'Post Clusters &#9652;'; // Up arrow
                }}
            }}

            function setActiveLink(clickedElement) {{
                document.querySelectorAll('.topic-link').forEach(link => {{
                    link.classList.remove('active');
                }});
                if (clickedElement) {{
                    clickedElement.classList.add('active');
                }}
            }}

            function collapseTopicList() {{
                const topicListContent = document.getElementById('topic-list-content');
                const topicListHeader = topicListContent.previousElementSibling; // This gets the <h3>
                if (!topicListContent.classList.contains('collapsed')) {{
                    topicListContent.classList.add('collapsed');
                    topicListHeader.innerHTML = 'Post Clusters &#9662;';
                }}
            }}

            function filterByTopic(topicId, event) {{
                event.preventDefault(); 
                setActiveLink(event.currentTarget);
                const allPosts = document.querySelectorAll('.post-container');
                allPosts.forEach(post => {{
                    if (post.dataset.topicId === topicId) {{
                        post.style.display = 'block';
                    }} else {{
                        post.style.display = 'none';
                    }}
                }});
                collapseTopicList();
            }}

            function showAllPosts(event) {{
                event.preventDefault(); 
                setActiveLink(event.currentTarget);
                const allPosts = document.querySelectorAll('.post-container');
                allPosts.forEach(post => {{
                    post.style.display = 'block';
                }});
                collapseTopicList(); // MODIFICATION: Collapse list after clicking "Show All"
            }}
        </script>
    </head>
    <body>
        <div class="page-header">
            <h1>Clankernet</h1>
            <p>Displaying the {len(posts)} most recent posts.</p>
        </div>
        <div class="content-area">
            <div class="trending-topics-container">
                {trending_html}
            </div>
            <div class="main-content">
                {all_posts_html}
            </div>
        </div>
    </body>
    </html>
    """
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    print("--- index.html file generated successfully! ---")


if __name__ == "__main__":
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            initial_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        initial_history = []

    posts_to_render = initial_history[:MAX_POSTS_TO_DISPLAY]
    generate_feed_html(posts_to_render, initial_history)

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
                generate_feed_html(posts_to_render, full_history)
            else:
                print("\n--- Skipped HTML generation due to failure in comment generation. ---")
        else:
            print("\n--- Skipped all generation due to failure in fetching a headline. ---")

        completed_posts += 1
        if completed_posts < NUMBER_OF_NEW_POSTS:
            time.sleep(2)
