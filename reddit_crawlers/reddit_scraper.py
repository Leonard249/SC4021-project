import requests
import csv
import time
import random
import os
import sys
from datetime import datetime

OUTPUT_FILE = 'dataset/vibe_coding_search.csv'
TARGET_COUNT = 100000 

SEARCH_QUERIES = [
    # Prominent AI IDEs & Extensions (Broad net)
    "Cursor AI", "Windsurf editor", "Cline VSCode", "Roo Code",
    "GitHub Copilot", "Supermaven", "Codeium", "Tabnine", "Continue.dev",
    "Qodo AI", "Aider AI", "PearAI", "Trae IDE", "Replit Agent",

    # Tool Comparisons (These generate massive, long-winded debate threads)
    "Cursor vs Windsurf", "Cursor vs Copilot", "Aider vs Cline",
    "Supermaven vs Copilot", "Windsurf vs VSCode", "Codeium vs Copilot",
    "Roo Code vs Cline", "Cursor vs Aider",

    # "Vibe Coding" & Emerging Workflows
    "Vibe coding", "Prompt-driven development", "AI-native development",
    "Natural language programming", "coding without typing",
    "Cursor Composer mode", "Windsurf Cascade", "Cursor rules file",
    "English as a programming language", "LLM driven development",
    "I don't write code anymore", "coding with LLMs only",

    # UI/Web Generators & Autonomous Agents
    "v0.dev", "Lovable AI", "Devin AI", "Teamblocks AI", "SWE-agent",
    "OpenHands AI", "Magic.dev", "Bolt.new", "bolt web container",

    # Models Specific to Coding (Including newer reasoning models)
    "Claude 3.5 Sonnet coding", "Claude 3.7 coding", "GPT-4o coding", 
    "GPT-4.5 code", "DeepSeek Coder", "DeepSeek R1 programming", 
    "OpenAI o1 coding benchmarks", "OpenAI o3-mini coding", 
    "Qwen2.5-Coder", "Llama 3 coding", "local LLM coding", "Ollama coding",

    # The Controversy: Job Market & Future
    "AI replacing junior devs", "is software engineering dead", "SWE jobs AI",
    "entry level tech jobs AI", "junior developers struggling with AI",
    "tech layoffs AI", "CS degree useless AI", "unable to find junior dev job AI",
    "AI replacing programmers", "post-AI software engineering",
    "AI replacing junior developers 2026",

    # The Controversy: Code Quality & Maintenance
    "AI code quality issues", "copilot hallucinations", "AI skill rot",
    "dependence on AI coding", "AI code maintainability crisis",
    "spaghetti code from LLMs", "debugging AI generated code",
    "senior engineers fixing AI code", "overreliance on AI tools",
    "AI tech debt", "AI generated code security vulnerabilities",
    "junior devs skill issue AI", "copilot hallucinating libraries",

    # Specific Use Cases & Daily Tasks (Catches the "how-to" threads)
    "refactoring with AI", "writing tests with Copilot", "ChatGPT debugging",
    "leetcode with AI", "AI code review tools", "generating unit tests AI",
    "migrating codebase AI", "understanding legacy code AI"
]

# SEARCH_QUERIES = [
#     "Claude 3.5 Sonnet coding", "Claude 3.7 coding", "GPT-4o coding", 
#     "GPT-4.5 code", "DeepSeek Coder", "DeepSeek R1 programming", 
#     "OpenAI o1 coding benchmarks", "OpenAI o3-mini coding", 
#     "Qwen2.5-Coder", "Llama 3 coding", "local LLM coding", "Ollama coding",

#     # The Controversy: Job Market & Future
#     "AI replacing junior devs", "is software engineering dead", "SWE jobs AI",
#     "entry level tech jobs AI", "junior developers struggling with AI",
#     "tech layoffs AI", "CS degree useless AI", "unable to find junior dev job AI",
#     "AI replacing programmers", "post-AI software engineering",
#     "AI replacing junior developers 2026",

#     # The Controversy: Code Quality & Maintenance
#     "AI code quality issues", "copilot hallucinations", "AI skill rot",
#     "dependence on AI coding", "AI code maintainability crisis",
#     "spaghetti code from LLMs", "debugging AI generated code",
#     "senior engineers fixing AI code", "overreliance on AI tools",
#     "AI tech debt", "AI generated code security vulnerabilities",
#     "junior devs skill issue AI", "copilot hallucinating libraries",

#     # Specific Use Cases & Daily Tasks (Catches the "how-to" threads)
#     "refactoring with AI", "writing tests with Copilot", "ChatGPT debugging",
#     "leetcode with AI", "AI code review tools", "generating unit tests AI",
#     "migrating codebase AI", "understanding legacy code AI"
# ]
# rando user agents
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
]

seen_ids = set()

def get_headers():
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }

def load_existing_ids():
    if not os.path.exists(OUTPUT_FILE):
        return 0
    count = 0
    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                if len(row) > 1:
                    seen_ids.add(row[1])
                    count += 1
    except:
        pass
    return count

def fetch_json(url, params=None):
    retries = 0
    max_retries = 10  # Try for a long time before giving up
    
    while retries < max_retries:
        try:
            response = requests.get(url, headers=get_headers(), params=params, timeout=15)
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 429:
                # Rate Limit Hit - The "Autonomous" fix is to wait it out.
                # Reddit soft-bans usually last 1-5 minutes.
                wait_time = random.randint(60, 180) * (retries + 1)
                print(f"\n[!] 429 Rate Limit. Sleeping for {wait_time}s to reset IP reputation...")
                time.sleep(wait_time)
                retries += 1
                
            elif response.status_code == 403:
                print(f"\n[!] 403 Forbidden. User-Agent might be blocked. Swapping and sleeping 60s...")
                time.sleep(60)
                retries += 1
            else:
                print(f"Error {response.status_code}: {url}")
                return None
                
        except Exception as e:
            print(f"Connection Error: {e}. Retrying...")
            time.sleep(10)
            retries += 1
            
    print("!!! Failed to fetch after multiple retries.")
    return None

def scrape_comments(post_id, subreddit, writer):
    """Fetches comments for a specific post"""
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
    data = fetch_json(url)
    
    if not data: return 0
    
    count = 0
    try:
        comments = data[1]['data']['children']
        for comment in comments:
            c_data = comment.get('data', {})
            cid = c_data.get('id')
            body = c_data.get('body', '')
            
            if cid not in seen_ids and body and body != '[deleted]' and len(body) > 20:
                writer.writerow([
                    f"r/{subreddit}", cid, 'Comment',
                    c_data.get('author'), body.replace('\n', ' '), 
                    c_data.get('score'),
                    datetime.now().strftime('%Y-%m-%d') # Reddit JSON often lacks clean dates in comments, easier to timestamp fetch
                ])
                seen_ids.add(cid)
                count += 1
                if count >= 30: break # Limit comments per post to keep moving
    except:
        pass
    return count

def main():
    current_count = load_existing_ids()
    print(f"Resuming with {current_count} records. Target: {TARGET_COUNT}")
    
    with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if current_count == 0:
            writer.writerow(['Source', 'ID', 'Type', 'Author', 'Text', 'Score', 'Date'])
        
        # Loop through keywords indefinitely until target is reached
        while current_count < TARGET_COUNT:
            for query in SEARCH_QUERIES:
                print(f"\n>>> Searching for: '{query}' [Total: {current_count}]")
                
                # Search Endpoint
                base_url = "https://www.reddit.com/search.json"
                after_token = None
                
                # Scroll depth per keyword
                for page in range(8): 
                    if current_count >= TARGET_COUNT: break
                    
                    params = {
                        'q': query,
                        'sort': 'new',   # 'new' ensures we don't just get the same top posts forever
                        'limit': 100,    # Max allowed
                        'after': after_token
                    }
                    
                    data = fetch_json(base_url, params)
                    if not data: break
                    
                    posts = data.get('data', {}).get('children', [])
                    if not posts: break
                    
                    for post in posts:
                        if current_count >= TARGET_COUNT: break
                        
                        p_data = post['data']
                        pid = p_data['id']
                        subreddit = p_data['subreddit']
                        
                        if pid not in seen_ids:
                            # Save Post
                            text = f"{p_data.get('title')} {p_data.get('selftext')}"
                            writer.writerow([
                                f"r/{subreddit}", pid, 'Post', 
                                p_data.get('author'), text.replace('\n', ' '), 
                                p_data.get('score'),
                                datetime.fromtimestamp(p_data.get('created_utc', time.time())).strftime('%Y-%m-%d')
                            ])
                            seen_ids.add(pid)
                            current_count += 1
                            
                            # Save Comments
                            if p_data.get('num_comments', 0) > 5:
                                print(f"   -> Comments for: {p_data.get('title')[:30]}...")
                                c_added = scrape_comments(pid, subreddit, writer)
                                current_count += c_added
                                # Sleep specifically after comment fetches to behave human-like
                                time.sleep(random.uniform(2, 5))
                    
                    after_token = data.get('data', {}).get('after')
                    if not after_token: break
                    
                    # Sleep between search result pages
                    time.sleep(random.uniform(3, 8))
                    
            if current_count < TARGET_COUNT:
                print("Cycle complete. Sleeping 60s before restarting keyword loop...")
                time.sleep(60)

    print("Target Reached!")

if __name__ == "__main__":
    main()