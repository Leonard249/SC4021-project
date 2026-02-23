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
