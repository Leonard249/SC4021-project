# Task
You are a skilled programmer with a simple task: to convert the JSON schema in the folder `articles_extraction/scraped_articles` to a new format. You must save this new format to `articles_extraction/articles_updated_schema` folder. So each entry in `articles_extraction/scraped_articles` should be converted to a new schema and it should be saved as JSON with the same name in `articles_extraction/articles_updated_schema`. For this task, you must create python script to perform the conversion. 

## Schema Information
Old Schema: 
```json
    {
    "url": "",
    "normalized_url": "",
    "source_type": "",
    "title": "",
    "author": "",
    "date": "",
    "text": "",
    "excerpt": "",
    "word_count": ,
    "scraped_at": "",
    "scrape_method": "",
    "queries": [""]
    }
```

New Schema: 
```json
    {
    "ID": "string",                // Unique identifier for the post
    "Source": "string",            // Platform source (e.g., Reddit)
    "Type": "string",              // Type of entry (Post, Comment, etc.)
    "Author": "string",            // Username of the author
    "Title": "string|null",        // Title of the post (nullable if not present)
    "Text": "string",              // Raw text content of the post
    "Score": "integer",            // Upvotes or score
    "Date": "string YYYY-MM-DD",   // Date of posting
    "Word_Count": "integer",       // Word count of text
    "Comments": [                  // Array of comment objects
        ""
    ]
    }

```

## Conversion Protocol
Note on conversion from Old to New Schema:
    1. Here is the mapping of data from old to new schema (format: old schema -> new schema)
        1. "source_type" -> "Source"
        2. "author" -> "Author"
        3. "title" -> "Title"
        4. "text" -> "Text" 
        5. "date" -> "Date"
        6. "word_count" -> "Word_Count" 
    2. This is how you should fill in the other fields in the new schema:
        1. For "ID", set it as `article_{json_file_name}`
            - For example, if the original JSON file name is `ffd06d8a3b6ab13d`, then the ID for that article is `article_ffd06d8a3b6ab13d`
        2. For "Type", set it to 'Article' for all entries
        3. For "Score", set it to 0 for all entries
        4. For "Comments", leave it as an empty array. In this task, we are dealing with scraped article, so there's no comment. 


