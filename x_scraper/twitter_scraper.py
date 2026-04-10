
import asyncio
from twikit import Client
import pandas as pd
from datetime import datetime
import json
import os
import re
import random
import traceback


# CONFIGS
USERNAME = 'ENTER X ACCOUNT USERNAME HERE'
EMAIL = 'ENTER X ACCOUNT EMAIL HERE'
PASSWORD = 'ENTER X ACCOUNT PW HERE'
TWEETS_PER_KEYWORD = 1000

# Date range for scraping
# Accepts formats: 'YYYY-MM-DD'
# Set to None to disable date filtering
SINCE_DATE = '2025-11-15'  # Example: '2026-02-10' - tweets from this date onwards
UNTIL_DATE = '2026-01-26' # Example: '2026-02-20' - tweets before this date (not inclusive)

# Search keywords for AI coding sentiment
SEARCH_KEYWORDS = [

    # AI coding tools
    'Claude code', #1997 tweets from Fri Feb 20 22:13:11 +0000 2026 to Sun Feb 22 16:51:42 +0000 2026
    'ChatGPT coding', #1000 tweets from Fri Feb 13 17:43:20 +0000 2026 to Sun Feb 22 17:16:53 +0000 2026
    'Cursor AI', #1971 tweets from Wed Feb 18 23:57:26 +0000 2026 to Sun Feb 22 18:06:54 +0000 2026
    'Github Copilot', #1980 tweets from Fri Feb 13 09:45:32 +0000 2026 to Mon Feb 23 02:29:53 +0000 2026
    'Gemini Antigravity', #1990 tweets from Mon Feb 16 00:23:32 +0000 2026 to Mon Feb 23 03:32:56 +0000 2026
    'AI coding assistant', #1000 tweets from Sat Jan 10 00:22:41 +0000 2026 to Mon Feb 23 03:48:13 +0000 2026
    'LLM coding', #1000 tweets from Tue Feb 10 03:05:11 +0000 2026 till Mon Feb 23 05:08:54 +0000 2026


    # Vibe and workflow
    'vibe coding', #1991 tweets from Sat Feb 21 10:59:42 +0000 2026 to Mon Feb 23 10:28:29 +0000 2026
    'AI coding workflow', #1000 tweets from Fri Jan 16 10:39:31 +0000 2026 till Mon Feb 23 11:27:46 +0000 2026

    # Likely positive comments
    'coding productivity boost', #1000 tweets from Wed Jun 11 05:15:22 +0000 2025 till Mon Feb 23 10:44:54 +0000 2026
    'AI coding amazing', #1000 tweets from Wed Sep 24 13:50:35 +0000 2025 till Mon Feb 23 11:33:23 +0000 2026
    'software engineering dead', #1000 tweets from Wed Aug 06 04:51:52 +0000 2025 till Mon Feb 23 15:43:16 +0000 2026

    # Likely negative comments
    'AI code bloat', #556 tweets
    'AI code slop', #2490 tweets from Fri Dec 19 00:35:38 +0000 2025 till Mon Feb 23 13:17:02 +0000 2026
    'debug AI code', #1986 tweets from Fri Dec 26 00:47:52 +0000 2025 till Mon Feb 23 13:53:28 +0000 2026
    'AI code hallucination', #615 tweets
    'copilot annoying', #1000 tweets from Tue Oct 08 04:54:06 +0000 2024 till Mon Feb 23 10:17:23 +0000 2026

    
    'AI coding skill rot', #4 tweets
    'depend AI coding', #308 tweets
    'overreliance AI coding', #36 tweets
    
]


# Output files
OUTPUT_FILE = 'tweets.xlsx'
BACKUP_JSON = 'tweets_backup.json'
COOKIES_FILE = 'cookies.json'

    
# ADVERTISEMENT FILTERS
def is_advertisement(text):
    """
    Returns True if the text looks like an ad, spam, or self-promotion.
    """
    # Convert to lowercase for easier matching
    text_lower = text.lower()
    
    # 1. Strong Commercial Keywords (Selling something)
    ad_keywords = [
        r"sign up now", r"subscribe to my", 
        r"link in bio", r"buy now", r"pre-order", r"preorder",
        r"limited time", r"discount code", r"use code", r"promo code",
        r"50% off", r"70% off", r"free trial", r"money back",
        r"lifetime access", r"early bird"
    ]
    
    # 2. "Guru" / Course / Newsletter Spam
    spam_keywords = [
        r"bootcamp", r"webinar", 
        r"free training", r"growth hacking", r"passive income",
        r"dm me", r"i'll send you", r"check out my course", r"my newsletter"
    ]
    
    # Combine lists
    all_patterns = ad_keywords + spam_keywords
    
    # Check if any pattern exists in the text
    for pattern in all_patterns:
        if re.search(pattern, text_lower):
            return True
        
    if re.search(r"https?://\S+", text_lower):
        return True
            
    return False



# HELPER FUNCTIONS
def clean_text(text):
    """Remove extra whitespace and normalize text"""
    if text:
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    return ''

def convert_to_twitter_date(date_input):
    """
    Convert various date formats to Twitter's search format (YYYY-MM-DD)
    Accepts:
      - 'YYYY-MM-DD' (returns as-is)
      - 'Thu Feb 19 12:02:31 +0000 2026' (tweet timestamp format)
      - datetime object
    """
    if not date_input:
        return None
    
    if isinstance(date_input, str):
        # If already in YYYY-MM-DD format, return as-is
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_input):
            return date_input
        
        # Try to parse tweet timestamp format: 'Thu Feb 19 12:02:31 +0000 2026'
        try:
            dt = datetime.strptime(date_input, '%a %b %d %H:%M:%S %z %Y')
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass
        
        # Try ISO format
        try:
            dt = datetime.fromisoformat(date_input.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            pass
    
    elif isinstance(date_input, datetime):
        return date_input.strftime('%Y-%m-%d')
    
    # If we can't parse it, return the original and let it fail with a clear error
    print(f" Warning: Could not parse date '{date_input}'. Expected format: 'YYYY-MM-DD' or 'Thu Feb 19 12:02:31 +0000 2026'")
    return str(date_input)

def check_cookies_file():
    """Check if cookies file exists and guide user if not"""
    if not os.path.exists(COOKIES_FILE):
        print("\n" + "=" * 60)
        print(" COOKIES FILE NOT FOUND")
        print("=" * 60)
        print("\nTo bypass Cloudflare, you need to manually extract cookies.")
        print("\n STEP-BY-STEP INSTRUCTIONS:")
        print("\n1. Install Cookie-Editor browser extension:")
        print("   Chrome: https://chrome.google.com/webstore/detail/cookie-editor/hlkenndednhfkekhgcdicdfddnkalmdm")
        print("   Firefox: https://addons.mozilla.org/en-US/firefox/addon/cookie-editor/")
        print("\n2. Login to twitter.com (X.com) in your browser normally")
        print("\n3. Click the Cookie-Editor extension icon")
        print("\n4. Click 'Export' button (bottom right)")
        print("\n5. Click 'Export as JSON'")
        print("\n6. Save the file as 'cookies.json' in this directory:")
        print(f"   {os.path.abspath('.')}")
        print("\n7. Run this script again")
        print("\n" + "=" * 60)
        return False
    return True

async def search_tweets_by_keyword(client, keyword, max_tweets=TWEETS_PER_KEYWORD):
    """
    Search tweets with Pagination + Auto-Wait for Rate Limits
    """
    # Build search query with date filters if specified
    search_query = keyword
    since_date_formatted = convert_to_twitter_date(SINCE_DATE)
    until_date_formatted = convert_to_twitter_date(UNTIL_DATE)
    
    if since_date_formatted:
        search_query += f" since:{since_date_formatted}"
    if until_date_formatted:
        search_query += f" until:{until_date_formatted}"
    
    print(f"\n Searching for: '{search_query}'")
    tweets_data = []
    total_scraped = 0  # Counter for 500-tweet sleep interval
    
    try:
        # Initial Search
        tweets = await client.search_tweet(search_query, 'Latest')
        
        while len(tweets_data) < max_tweets:
            try:
                # If no more tweets are available, stop
                if not tweets:
                    print(" No more tweets found.")
                    break

                # Process the current batch of tweets
                for tweet in tweets:
                    if len(tweets_data) >= max_tweets:
                        break
                        
                    # FILTER: Skip Ads, Spam, and Links
                    text_clean = clean_text(tweet.text)
                    if is_advertisement(text_clean):
                        continue
                    
                    try:
                        tweet_info = {
                            'tweet_id': tweet.id,
                            'username': tweet.user.name,
                            'screen_name': tweet.user.screen_name,
                            'tweet_text': clean_text(tweet.text),
                            'created_at': tweet.created_at,
                            'like_count': tweet.favorite_count if hasattr(tweet, 'favorite_count') else 0,
                            'retweet_count': tweet.retweet_count if hasattr(tweet, 'retweet_count') else 0,
                            'reply_count': tweet.reply_count if hasattr(tweet, 'reply_count') else 0,
                            'view_count': tweet.view_count if hasattr(tweet, 'view_count') else 0,
                            'is_retweet': tweet.retweeted_tweet is not None if hasattr(tweet, 'retweeted_tweet') else False,
                            'search_keyword': keyword,
                            'language': tweet.lang if hasattr(tweet, 'lang') else 'unknown',
                        }
                        tweets_data.append(tweet_info)
                        total_scraped += 1
                        
                        # Sleep for ~5 minutes every 500 tweets
                        if total_scraped % 500 == 0:
                            sleep_duration = random.uniform(280, 320)  # ~5 minutes (4.7-5.3 min)
                            print(f"\nReached {total_scraped} tweets - sleeping for {sleep_duration/60:.1f} minutes...")
                            await asyncio.sleep(sleep_duration)
                            print(f" Resuming scraping...")
                            
                    except Exception as e:
                        continue

                print(f"Collected {len(tweets_data)}/{max_tweets} tweets...")

                # Stop if we have enough
                if len(tweets_data) >= max_tweets:
                    break

                # SAFETY DELAY: Sleep for 5 to 8 seconds randomly
                sleep_time = random.uniform(5, 8)
                print(f" Sleeping {sleep_time:.1f}s...")
                await asyncio.sleep(sleep_time)
                
                # GET NEXT PAGE
                tweets = await tweets.next()

            except Exception as e:

                # HANDLING RATE LIMITS (Error 429)
                error_msg = str(e)
                if "429" in error_msg or "Rate limit" in error_msg:
                    print("\n" + "!"*60)
                    print(" RATE LIMIT HIT (429)")
                    print(" The script will pause for 15 minutes to cool down.")
                    print("!"*60 + "\n")
                    
                    # Wait 15 minutes (900 seconds) + 10s buffer
                    await asyncio.sleep(910) 
                    
                    print(" Resuming search...")
                    
                    # Simple retry strategy: Try fetching next page again
                    try:
                        tweets = await tweets.next()
                    except:
                        print("Could not recover session. Moving to next keyword.")
                        break
                else:
                    print(f"Error fetching page: {e}")
                    break

        # Display date range for this keyword pass
        if tweets_data:
            dates = [tweet['created_at'] for tweet in tweets_data]
            earliest_date = min(dates)
            latest_date = max(dates)
            print(f"Date range for this pass: {earliest_date} to {latest_date}")
        
        print(f"Finished '{keyword}': {len(tweets_data)} tweets collected")

    except Exception as e:
        print(f"Error searching '{keyword}': {e}")

    return tweets_data

async def main():

    print("=" * 60)
    print(" AI CODING SENTIMENT SCRAPER")
    print(" (Cloudflare Bypass - Cookie Method)")
    print("=" * 60)

    # Check for cookies file
    if not check_cookies_file():
        return

    # Initialize client
    print("\n Initializing Twitter client...")
    client = Client('en-US')

    # Load cookies from file
    try:
        print(" Loading cookies from file...")
        
        # Read the raw file first to check format
        with open(COOKIES_FILE, 'r') as f:
            raw_cookies = json.load(f)

        # Check if it's a list (Cookie-Editor format) and convert if needed
        if isinstance(raw_cookies, list):
            print("Detected Cookie-Editor JSON format. Converting...")
            converted_cookies = {}
            for cookie in raw_cookies:
                # Extract only the name and value
                if 'name' in cookie and 'value' in cookie:
                    converted_cookies[cookie['name']] = cookie['value']
            
            # Save the converted version to a temporary file
            # twikit needs a file path, not a dict object
            with open('cookies_fixed.json', 'w') as f:
                json.dump(converted_cookies, f)
            
            # Load the fixed file
            client.load_cookies(path='cookies_fixed.json')
            print(" Converted and loaded cookies successfully!")
            
            # Clean up temp file
            if os.path.exists('cookies_fixed.json'):
                os.remove('cookies_fixed.json')
                
        else:
            # It's already in the correct format (simple dict)
            client.load_cookies(path=COOKIES_FILE)
            print(" Cookies loaded successfully!")

        # Test if cookies work by making a simple request
        print(" Testing cookies validity...")
        try:
            # Try a simple search to verify cookies work
            test_tweets = await client.search_tweet('test', 'Latest')
            print(" Cookies are valid! Ready to scrape.")
        except Exception as e:
            print(f"Cookies are invalid or expired: {e}")
            print("   1. Delete the old cookies.json file")
            print("   2. Login to twitter.com again in your browser")
            print("   3. Extract fresh cookies using Cookie-Editor")
            print("   4. Run this script again")
            return

    except Exception as e:
        print(f" Error loading cookies: {e}")
        traceback.print_exc()
        return
    

    # Collect all tweets
    all_tweets = []

    print(f"\n Starting to collect tweets for {len(SEARCH_KEYWORDS)} keywords...")
    print(f" Target: {TWEETS_PER_KEYWORD} tweets per keyword")
    
    # Display date range if specified
    if SINCE_DATE or UNTIL_DATE:
        print(f" Date range filter:")
        if SINCE_DATE:
            formatted_since = convert_to_twitter_date(SINCE_DATE)
            print(f"   • From: {SINCE_DATE} → {formatted_since}")
        if UNTIL_DATE:
            formatted_until = convert_to_twitter_date(UNTIL_DATE)
            print(f"   • Until: {UNTIL_DATE} → {formatted_until} (exclusive)")
    else:
        print(f" No date filtering (searching all available tweets)")

    for i, keyword in enumerate(SEARCH_KEYWORDS, 1):
        print(f"\n[{i}/{len(SEARCH_KEYWORDS)}] Processing keyword...")
        tweets = await search_tweets_by_keyword(client, keyword, TWEETS_PER_KEYWORD)
        if tweets:
            all_tweets.extend(tweets)

        print(f" Total tweets collected so far: {len(all_tweets)}")
        sleep_time = random.uniform(3, 5)
        await asyncio.sleep(sleep_time)

    # Process and save results
    if not all_tweets:
        print("\n No tweets collected")
        return

    print(f"\nProcessing {len(all_tweets)} collected tweets...")
    new_df = pd.DataFrame(all_tweets)


    # LOAD EXISTING DATA
    if os.path.exists(OUTPUT_FILE):
        print(f"   Found existing file: {OUTPUT_FILE}")
        try:
            # Load the existing Excel file
            existing_df = pd.read_excel(OUTPUT_FILE)
            print(f"   Loaded {len(existing_df)} existing tweets.")
            
            # Combine old and new data
            df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception as e:
            print(f"   ⚠️ Could not load existing file (creating new one): {e}")
            df = new_df
    else:
        print("   No existing file found. Creating new dataset.")
        df = new_df


    # REMOVE DUPLICATES
    initial_count = len(df)
    # Remove duplicates based on tweet_id to ensure every tweet is unique
    df = df.drop_duplicates(subset=['tweet_id'], keep='last') # Keep 'last' (newest scrape) or 'first' (original)
    final_count = len(df)
    
    duplicates_removed = initial_count - final_count
    if duplicates_removed > 0:
        print(f"   Removed {duplicates_removed} duplicate tweets across old and new data.")

    # Sort by date (newest first)
    df = df.sort_values('created_at', ascending=False)

    # Save to Excel
    print(f"\n Saving {len(df)} total tweets to Excel: {OUTPUT_FILE}")
    output_path = os.path.abspath(OUTPUT_FILE)
    df.to_excel(output_path, index=False, engine='openpyxl')

    # Save backup as JSON
    print(f" Saving backup: {BACKUP_JSON}")
    backup_path = os.path.abspath(BACKUP_JSON)
    df.to_json(backup_path, orient='records', indent=2, date_format='iso', force_ascii=False)

    # Statistics
    print("\n" + "=" * 60)
    print(" SCRAPING STATISTICS")
    print("=" * 60)
    print(f"Total unique tweets: {final_count}")
    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")

    print(f"\n Tweets per keyword:")
    for keyword in SEARCH_KEYWORDS:
        count = len(df[df['search_keyword'] == keyword])
        print(f"  • {keyword}: {count}")

    print(f"\n Engagement metrics:")
    print(f"  • Total likes: {df['like_count'].sum():,}")
    print(f"  • Total retweets: {df['retweet_count'].sum():,}")
    print(f"  • Avg likes per tweet: {df['like_count'].mean():.1f}")
    print(f"  • Avg retweets per tweet: {df['retweet_count'].mean():.1f}")

    print(f"\n Language distribution:")
    lang_dist = df['language'].value_counts().head(5)
    for lang, count in lang_dist.items():
        print(f"  • {lang}: {count}")

    print(f"\n Content type:")
    retweet_count = len(df[df['is_retweet'] == True])
    original_count = len(df[df['is_retweet'] == False])
    print(f"  • Original tweets: {original_count} ({original_count/final_count*100:.1f}%)")
    print(f"  • Retweets: {retweet_count} ({retweet_count/final_count*100:.1f}%)")


    print("\n" + "=" * 60)
    print(" SCRAPING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n Files saved:")
    print(f"  • Excel: {output_path}")
    print(f"  • JSON backup: {backup_path}")



# Run Scraper
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Scraping interrupted")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()