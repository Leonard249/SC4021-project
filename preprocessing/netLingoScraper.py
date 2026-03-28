import requests
from bs4 import BeautifulSoup
import json
import re

def scrape_netlingo():
    url = "https://www.netlingo.com/acronyms.php"
    # A user-agent helps prevent the website from instantly blocking the script
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    print(f"Fetching acronyms from {url}...")
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Failed to fetch page. Status code: {response.status_code}")
        return
        
    soup = BeautifulSoup(response.text, "html.parser")
    acronym_dict = {}
    
    # NetLingo groups its alphabetical lists inside divs with the class 'list_box3'
    list_boxes = soup.find_all('div', class_='list_box3')
    
    for box in list_boxes:
        for li in box.find_all('li'):
            a_tag = li.find('a')
            if a_tag:
                acronym = a_tag.text.strip()
                
                # The definition is the remaining text inside the <li> tag
                # We replace the acronym text once to isolate the definition
                definition = li.text.replace(acronym, '', 1).strip()
                
                # Clean up the leading hyphens or spaces NetLingo uses
                if definition.startswith('-'):
                    definition = definition[1:].strip()
                    
                # Lowercase everything for consistency in the pipeline
                acronym_dict[acronym.lower()] = definition.lower()
                
    output_file = "slang_dict.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(acronym_dict, f, indent=4)
        
    print(f"Successfully scraped {len(acronym_dict)} acronyms into {output_file}!")



def scrape_netlingo_smileys():
    url = "https://www.netlingo.com/smileys.php"
    # A more complete User-Agent prevents the site from blocking us as a bot
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36'
    }
    
    print(f"Fetching emoticons from {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Throws an error if the page is down
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch page: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    emoticon_dict = {}
    
    # LOCATE THE LISTS:
    # Use Regex to find any div class that contains 'list_box' in case they changed the number
    list_boxes = soup.find_all('div', class_=re.compile(r'list_box'))
    
    # Fallback: If 'list_box' is completely gone, grab all list items in the main content area
    if not list_boxes:
        print("Warning: Target divs not found. Attempting a broader search...")
        main_content = soup.find('div', id='content') or soup.find('body')
        list_items = main_content.find_all('li')
    else:
        list_items = []
        for box in list_boxes:
            list_items.extend(box.find_all('li'))
            
    # EXTRACT THE DATA:
    for li in list_items:
        # Extract raw text and strip weird invisible characters
        full_text = li.get_text(" ", strip=True)
        if not full_text:
            continue
            
        # Scenario 1: The emoticon is wrapped in an anchor <a> tag
        a_tag = li.find('a')
        if a_tag:
            emoticon = a_tag.get_text(strip=True)
            # Remove the emoticon from the full text to isolate the definition
            meaning = full_text.replace(emoticon, '', 1).strip()
            # Clean up leading hyphens
            if meaning.startswith('-'):
                meaning = meaning[1:].strip()
                
            if emoticon and meaning:
                emoticon_dict[emoticon] = meaning.lower()
                
        # Scenario 2: It's just raw text like ":-) - smiling" or ":-( crying"
        else:
            # Try splitting by a hyphen surrounded by spaces first
            if ' - ' in full_text:
                parts = full_text.split(' - ', 1)
                emoticon_dict[parts[0].strip()] = parts[1].strip().lower()
            else:
                # Fallback Regex: Splits at the first space separating the symbol from the word
                match = re.match(r'^([^\s]+)\s+(.*)', full_text)
                if match:
                    emoticon = match.group(1).strip()
                    meaning = match.group(2).strip()
                    if meaning.startswith('-'):
                        meaning = meaning[1:].strip()
                    emoticon_dict[emoticon] = meaning.lower()

    if not emoticon_dict:
        print("Scraping failed: The HTML structure has changed too drastically.")
        return

    output_file = "emoticon_dict.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(emoticon_dict, f, indent=4)
        
    print(f"Successfully scraped {len(emoticon_dict)} emoticons into {output_file}!")


    

if __name__ == "__main__":
    scrape_netlingo()
    scrape_netlingo_smileys()