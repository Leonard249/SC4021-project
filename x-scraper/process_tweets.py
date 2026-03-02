import json
import re
from datetime import datetime

INPUT_FILE = 'raw_tweets.json'
OUTPUT_FILE = 'twitter_standardized.json'

def clean_tweet_text(text):
    """
    Removes @ tags, converts to lowercase, and cleans up extra spacing.
    """
    if not text:
        return ""
    
    # Remove @mentions (e.g., @username)
    text = re.sub(r'@\w+', '', text)
    
    # Convert to lowercase to match the Reddit schema example
    text = text.lower()
    
    # Remove extra spaces left behind after removing tags or newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def convert_twitter_date(date_str):
    """
    Converts Twitter's timestamp format to YYYY-MM-DD.
    """
    try:
        dt = datetime.strptime(date_str, '%a %b %d %H:%M:%S %z %Y')
        return dt.strftime('%Y-%m-%d')
    except ValueError:
        try:
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except:
            return date_str

def process_data():
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            tweets = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find '{INPUT_FILE}'.")
        return

    processed_data = []
    skipped_non_english = 0

    print(" Processing tweets and converting to group schema...")
    for tweet in tweets:
        # Filter out non-English tweets
        if tweet.get('language') != 'en':
            skipped_non_english += 1
            continue

        # Extract and format data
        original_text = tweet.get('tweet_text', '')
        final_text = clean_tweet_text(original_text)
        word_count = len(final_text.split())
        
        # Construct the record using the new agreed-upon schema
        new_record = {
            "ID": str(tweet.get('tweet_id')),
            "Source": "Twitter",
            "Type": "Post",
            "Author": tweet.get('username'), 
            "Text": final_text,
            "Score": int(tweet.get('like_count', 0)), # Mapping like_count to Score
            "Date": convert_twitter_date(tweet.get('created_at')),
            "Word_Count": word_count,
            "Title": None,     # Left as None
            "Comments": None   # Left as None
        }

        processed_data.append(new_record)

    # Save to the new JSON file
    print(f" Saving formatted data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    # Output Summary
    print("\n" + "=" * 50)
    print(" PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"Original tweets:  {len(tweets)}")
    print(f"Skipped (non-EN): {skipped_non_english}")
    print(f"Saved records:    {len(processed_data)}")
    print(f"Output File:      {OUTPUT_FILE}")

if __name__ == "__main__":
    process_data()