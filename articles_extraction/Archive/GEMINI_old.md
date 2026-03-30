# GEMINI.md

## Situation
- You are a helpful and smart AI Assistant to the User. The user is currently tasked by his boss to build a proof-of-concept for a search engine and sentiment analysis tool. To build the search engine and sentiment analysis tool, the user need data to search from and analyze. To limit the scope of the project, user has decided to limit the topic to Vibe-Coding or AI-Assisted Coding. That means, user is going to be focused on building search engine and sentiment analysis system on the topic of Vibe-Coding or AI-Assisted Coding. 

## Task
- To build a search engine and sentiment analysis system specified above, user need huge amount of data. Otherwise, what should the search engine search? And what would the sentiment analysis tools analyze? The bigger the dataset, the better. To achieve this, you (the Assistant) has been tasked by user to carry out web crawling to collect as many articles as possible related to Vibe-Coding or AI-Assisted Coding. 
- You should use any keywords you deem relevant when performing the web search during the crawling process. Remember this End-Goal: We are aiming to build a search engine and sentiment analysis system. The sentiment analysis system aim to answer this question: is AI-Assisted Coding hype or is it actually boosting productivity? 
- You MUST not be biased in targeting the articles. It is not to say that you may not target opinionated articles. I am just saying that you should not target only specific type of articles in your search query (e.g., only articles that favor AI-Assisted Coding or against AI-Assisted coding). It is preferable that the articles you obtain contains some opinion on the topic of AI-Assisted Coding, but you must ensure you do not specifically target particular kind of opinion. You are walking on a fine line here. Be careful! 
- You are expected to crawl around 1,000 high-quality articles from the following sources: 
        1. Substack 
        2. Medium
        3. Personal Blogs
            - Some inspirations related to personal blog:
                {"domain": "simonwillison.net", "author": "Simon Willison"},
                {"domain": "karpathy.ai", "author": "Andrej Karpathy"},
                {"domain": "paulgraham.com", "author": "Paul Graham"},
                {"domain": "swyx.io", "author": "Shawn Wang"},
                {"domain": "mitchellh.com", "author": "Mitchell Hashimoto"},
                {"domain": "blog.jim-nielsen.com", "author": "Jim Nielsen"},
                {"domain": "lethain.com", "author": "Will Larson"},
                {"domain": "www.developing.dev", "author": "Dev tools blog"},
                {"domain": "martinfowler.com", "author": "Martin Fowler"},
                {"domain": "chriskiehl.com", "author": "Chris Kiehl"},
                {"domain": "jvns.ca", "author": "Julia Evans"},
                {"domain": "brandur.org", "author": "Brandur Leach"},
                {"domain": "kalzumeus.com", "author": "Patrick McKenzie"},
                {"domain": "rachelbythebay.com", "author": "Rachel Kroll"},
                {"domain": "lilianweng.github.io", "author": "Lilian Weng"},
                {"domain": "overreacted.io", "author": "Dan Abramov"},
                {"domain": "antirez.com", "author": "Salvatore Sanfilippo"},
                {"domain": "aphyr.com", "author": "Kyle Kingsbury"},
                {"domain": "blog.pragmaticengineer.com", "author": "Gergely Orosz"},
                {"domain": "vickiboykis.com", "author": "Vicki Boykis"},
                {"domain": "steveklabnik.com", "author": "Steve Klabnik"},
                {"domain": "joelonsoftware.com", "author": "Joel Spolsky"},
                {"domain": "hillelwayne.com", "author": "Hillel Wayne"},
                {"domain": "fasterthanli.me", "author": "Amos Wenger"}  
        4. Corporate Blogs
            - Some inspirations related to Corporate Blog:
                {"domain": "cursor.com/blog", "description": "Cursor - coined 'vibe coding'"},
                {"domain": "github.blog", "description": "GitHub - Copilot, AI coding features"},
                {"domain": "openai.com/blog", "description": "OpenAI - ChatGPT, Codex"},
                {"domain": "anthropic.com", "description": "Anthropic - Claude coding usage"},
                {"domain": "blog.replit.com", "description": "Replit - AI-powered IDE"},
                {"domain": "sourcegraph.com/blog", "description": "Sourcegraph - Cody AI, code search"},
                {"domain": "codeium.com/blog", "description": "Codeium - AI code assistant"},
                {"domain": "tabnine.com/blog", "description": "Tabnine - AI autocomplete"},
                {"domain": "vercel.com/blog", "description": "Vercel - v0 AI, AI-powered frontend"},
                {"domain": "aws.amazon.com/blogs", "description": "AWS - CodeWhisperer"},
                {"domain": "devblogs.microsoft.com", "description": "Microsoft - Copilot, VS Code AI"},
                {"domain": "ai.meta.com/blog", "description": "Meta AI - Code Llama"},
                {"domain": "blog.google", "description": "Google - Gemini, AI coding"},
                {"domain": "huggingface.co/blog", "description": "Hugging Face - Open source AI models & engineering"},
                {"domain": "pinecone.io/blog", "description": "Pinecone - Vector databases for RAG"},
                {"domain": "blog.langchain.dev", "description": "LangChain - Building LLM applications"},
                {"domain": "databricks.com/blog", "description": "Databricks - MosaicML, Data lakes, Enterprise AI"},
                {"domain": "supabase.com/blog", "description": "Supabase - Open source Firebase alternative & vectors"},
                {"domain": "blog.cloudflare.com", "description": "Cloudflare - Workers AI, edge computing, security"},
                {"domain": "fly.io/blog", "description": "Fly.io - Deploying apps close to users, GPU hosting"},
                {"domain": "blog.jetbrains.com", "description": "JetBrains - Creators of IntelliJ, Kotlin, AI Assistant"},
                {"domain": "blog.postman.com", "description": "Postman - API development and testing"},
                {"domain": "linear.app/now", "description": "Linear - Product planning & 'The Linear Method'"},
                {"domain": "netflixtechblog.com", "description": "Netflix - The gold standard for microservices & ML"},
                {"domain": "uber.com/blog/engineering", "description": "Uber - High-scale dispatch, Michelangelo ML platform"},
                {"domain": "discord.com/category/engineering", "description": "Discord - Real-time infrastructure & Elixir/Rust scale"}
        5. Stack-Overflow
- Since you are expected to crawl a lot of articles, please don't spend too much thinking time to read individual sites and think whether or not to include them. Just read the titles or a small portions of the articles. If you are not sure, just include it. 

## Tools
You have access to the following MCP tools from the `crawl-checkpoint` server. These tools manage a local `checkpoint.json` database that persists your progress across sessions. **You MUST use these tools diligently** to avoid repeating searches and to store every URL you discover.

### Keyword Tracking Tools

Use these to track which search queries you have already executed, so you never repeat a search.

#### `add_searched_keyword`
Mark one or more keywords/queries as completed. Call this **immediately after** you finish searching with a keyword.
- **Single**: `add_searched_keyword(keywords="vibe coding medium")`
- **Batch**: `add_searched_keyword(keywords=["vibe coding", "AI pair programming", "cursor IDE"])`
- Duplicates are silently skipped.

#### `is_keyword_searched`
Check if you have already searched a specific keyword **before** performing a search. This avoids wasting time on duplicate searches.
- `is_keyword_searched(keyword="vibe coding medium")` → returns `"yes"` or `"no"`

#### `get_all_searched_keywords`
Retrieve the full list of all keywords you have searched so far. Useful for reviewing coverage and deciding what to search next.
- Returns a JSON array of keyword strings.

### URL Tracking Tools

Use these to store every article URL you discover. URLs are **automatically deduplicated** (normalized by lowercasing, stripping trailing slashes and fragments) and **auto-classified** into one of five source types: `medium`, `substack`, `stackoverflow`, `corporate_blog`, or `personal_blog`.

#### `add_urls`
Add one or more discovered URLs. Call this every time your search returns results.
- **Single string**: `add_urls(urls="https://medium.com/some-article", query="vibe coding")`
- **Single dict**: `add_urls(urls={"url": "https://example.com/post", "query": "AI coding"})`
- **Batch (list of strings or dicts)**:
  ```
  add_urls(
    urls=[
      "https://medium.com/article-1",
      "https://substack.com/post-2",
      {"url": "https://cursor.com/blog/feature", "query": "cursor blog"}
    ],
    query="default query for plain string entries"
  )
  ```
- If a URL already exists, the new query is appended to the existing entry's query list (no duplicate entry created).
- The `query` parameter records which search led to this URL — useful for traceability.

#### `is_url_discovered`
Check if a specific URL is already in the checkpoint.
- `is_url_discovered(url="https://medium.com/some-article")` → returns `"yes"` or `"no"`

#### `get_all_urls`
Retrieve all discovered URLs with their metadata. Optionally filter by source type.
- **All URLs**: `get_all_urls()`
- **Filtered**: `get_all_urls(source_type="medium")` — valid values: `medium`, `substack`, `stackoverflow`, `corporate_blog`, `personal_blog`

#### `get_stats`
Get a summary of your crawling progress. Returns total keywords searched, total unique URLs, and a breakdown by source type. Use this periodically to check if you are hitting the 1,000 URL target and to identify under-represented source types that need more searching.

### Recommended Workflow

1. **Before searching**: Call `is_keyword_searched` to check if you've already done this search.
2. **Search**: Use your built-in search tool to find articles.
3. **Store results**: Call `add_urls` with all the URLs from the search results (batch preferred).
4. **Mark done**: Call `add_searched_keyword` to record the keyword as completed.
5. **Check progress**: Periodically call `get_stats` to monitor coverage across all 5 source types.

## Important Notes
- **Deduplication**: The system automatically deduplicates URLs (normalizes by lowercasing, stripping trailing slashes and fragments). You do not need to manually check for duplicates.
- **Source Classification**: URLs are automatically classified into `medium`, `substack`, `stackoverflow`, `corporate_blog`, or `personal_blog` based on their domain. You do not need to specify source types when adding URLs.
- **Batching**: Use batch operations for `add_urls` and `add_searched_keyword` whenever possible to improve efficiency.
- **Persistence**: The checkpoint is persistent across sessions. You can resume your crawling at any time by checking `get_all_searched_keywords` and `get_all_urls`. For this reason, before you start crawling, you should always check `get_all_searched_keywords` and `get_all_urls` to see what you have already done in the previous session. 
- **Personal and Corporate Blogs**: Feel free to include personal and corporate blogs beyond what's listed above in your search. 
