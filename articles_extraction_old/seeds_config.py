"""
Central configuration for blog & article crawling pipeline.
Contains curated blog domains and search keywords for vibe coding discovery.
"""

# =============================================================================
# Corporate Blog Seeds
# =============================================================================
CORPORATE_BLOG_SEEDS = [
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
]

# =============================================================================
# Personal Blog Seeds
# =============================================================================
PERSONAL_BLOG_SEEDS = [
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
]

# =============================================================================
# All curated domains (convenience list for seed_discovery.py)
# =============================================================================
ALL_CURATED_DOMAINS = [s["domain"] for s in CORPORATE_BLOG_SEEDS + PERSONAL_BLOG_SEEDS]

# =============================================================================
# Search Keywords
# =============================================================================
SEARCH_KEYWORDS = [
    '"vibe coding"',
    '"vibe-coding"',
    '"AI-assisted coding"',
    '"AI pair programming"',
    '"coding with AI"',
    '"cursor IDE" coding',
    '"github copilot" experience',
    '"AI code generation"',
    '"LLM coding" workflow',
    '"agentic coding"',
    '"prompt-driven development"',
    '"AI software engineer"',
    '"Agent Engineering"',
    '"claude code"',
    '"codex"',
    '"AI slop"',
    '"AI coding"',
    '"AI coding hype"'
]

# =============================================================================
# Google Search Settings
# =============================================================================
GOOGLE_RESULTS_PER_QUERY = 20       # Sweet spot: fewer queries, low CAPTCHA risk
GOOGLE_DELAY_MIN = 5                # Minimum seconds between Google queries
GOOGLE_DELAY_MAX = 15               # Maximum seconds between Google queries
GOOGLE_BACKOFF_SECONDS = 60         # Pause duration on CAPTCHA/429
GOOGLE_MAX_RETRIES = 3              # Max retries per query on failure

# =============================================================================
# Stack Exchange API Settings
# =============================================================================
STACKOVERFLOW_TAGS = [
    "github-copilot",
    "chatgpt",
    "openai-api",
]
STACKOVERFLOW_KEYWORDS = [
    "vibe coding",
    "AI-assisted coding",
    "copilot",
    "cursor IDE",
    "AI code generation",
]

# =============================================================================
# Output Paths (relative to articles_extraction/)
# =============================================================================
DISCOVERED_URLS_DIR = "discovered_urls"
RAW_ARTICLES_DIR = "raw_articles"
FILTERED_ARTICLES_DIR = "filtered_articles"

# =============================================================================
# API Keys (Loaded from environment variables)
# =============================================================================
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "fbeae28f62511c60160ad12ed24a2a69638c977c")
