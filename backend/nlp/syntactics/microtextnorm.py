"""
microtext_normalizer.py
SC4021 Information Retrieval 2026 — Microtext Normalization Pipeline

Normalizes raw crawled JSON records (Reddit, HackerNews, X, blogs) for
sentiment analysis on AI coding productivity opinions.

Pipeline order (order matters — do not rearrange):
  Stage 0 : Code & Markdown extraction (protect before everything)
  Stage 1 : Structural cleaning (URLs, HTML, whitespace)
  Stage 2 : Mention & hashtag extraction → stored in JSON fields
  Stage 3 : Emoji & emoticon handling
  Stage 4 : Elongated word normalization
  Stage 5 : Acronym & slang expansion
  Stage 6 : Spelling correction (optional, X/Twitter only by default)
  Stage 7 : Code token restoration (replace placeholders with <CODE>)
"""

import re
import json
import logging
from html import unescape
from pathlib import Path
from typing import Optional
import emoji as emoji_lib
from spellchecker import SpellChecker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tech terms that must NEVER be passed to acronym expansion or spell correction.
# All entries should be lowercase.
TECH_ALLOWLIST: set[str] = {
    # AI tools & models
    "ai", "gpt", "claude", "chatgpt", "copilot", "gemini", "llm", "llms",
    "openai", "anthropic", "mistral", "ollama", "cursor", "tabnine", "codeium",
    "replit", "devin", "aider", "continue",
    # Dev concepts & abbreviations
    "api", "apis", "sdk", "cli", "ide", "ui", "ux", "pr", "prs", "ci", "cd",
    "lsp", "ast", "oop", "tdd", "bdd", "ddd", "ml", "nlp", "cv", "gpu", "cpu",
    "ram", "os", "vcs", "iac", "orm", "rest", "grpc", "sql", "nosql", "html",
    "css", "dom", "ssh", "http", "https", "json", "yaml", "toml", "regex",
    "devops", "mlops", "linting", "refactoring", "debugging", "autocomplete",
    # Languages, tools, platforms
    "python", "typescript", "javascript", "golang", "rust", "java", "kotlin",
    "swift", "ruby", "php", "bash", "zsh", "vscode", "neovim", "vim", "emacs",
    "git", "github", "gitlab", "bitbucket", "docker", "kubernetes", "k8s",
    "aws", "gcp", "azure", "terraform", "ansible", "npm", "pip", "cargo",
    "pytest", "jest", "webpack", "vite", "eslint", "prettier",
    # Common abbreviations safe to keep as-is
    "vs", "aka", "fyi", "eta", "asap", "ngl", "tldr",
}

# Domain-priority overrides: when NetLingo lists the wrong meaning first for
# tech contexts, specify the correct expansion here (all keys lowercase).
DOMAIN_OVERRIDES: dict[str, str] = {
    "pr":   "pull request",
    "cd":   "continuous deployment",
    "ci":   "continuous integration",
    "ml":   "machine learning",
    "dl":   "deep learning",
    "ai":   "artificial intelligence",
    "api":  "application programming interface",
    "sdk":  "software development kit",
    "ide":  "integrated development environment",
    "cli":  "command line interface",
    "lsp":  "language server protocol",
    "ast":  "abstract syntax tree",
    "oop":  "object oriented programming",
    "tdd":  "test driven development",
    "orm":  "object relational mapping",
    "iac":  "infrastructure as code",
    "llm":  "large language model",
    "nlp":  "natural language processing",
    "rag":  "retrieval augmented generation",
    "gpu":  "graphics processing unit",
    "cpu":  "central processing unit",
}

# Keywords used to pick the most tech-relevant meaning when "-or-" is present
# and no domain override exists.
TECH_HINT_WORDS: set[str] = {
    "software", "code", "coding", "program", "computer", "server", "network",
    "internet", "web", "data", "file", "system", "tech", "digital", "online",
    "application", "app", "database", "cloud", "development", "developer",
    "programming", "engineering", "platform",
}


class MicrotextNormalizer:
    """
    Full microtext normalization pipeline for AI coding productivity corpus.

    Usage
    -----
    normalizer = MicrotextNormalizer(
        emoticons_path="../data/emoticon_dict.json",
        slang_path="../data/slang_dict.json",
    )
    normalized_record = normalizer.normalize_record(record, apply_spellcheck=False)
    """

    def __init__(
        self,
        emoticons_path: str | Path,
        slang_path: str | Path,
        tech_allowlist: Optional[set[str]] = None,
        domain_overrides: Optional[dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        emoticons_path : path to emoticon_dict.json
            Format: { ":p": "smiley with tongue hanging out", ... }
        slang_path : path to slang_dict.json
            Format: { "afk": "away from keyboard -or- a free kill", ... }
        tech_allowlist : override the default TECH_ALLOWLIST set
        domain_overrides : override the default DOMAIN_OVERRIDES dict
        """
        self.tech_allowlist: set[str] = tech_allowlist or TECH_ALLOWLIST
        self.domain_overrides: dict[str, str] = domain_overrides or DOMAIN_OVERRIDES

        emoticons_file = self._resolve_path(emoticons_path)
        slang_file = self._resolve_path(slang_path)

        self.emoticon_map: dict[str, str] = self._load_emoticons(emoticons_file)
        self.slang_map: dict[str, str] = self._load_slang(slang_file)

        # Pre-sort emoticons longest-first so greedy matching works correctly.
        # e.g. ":-)" is matched before ":)" to avoid partial matches.
        self._sorted_emoticons: list[tuple[str, str]] = sorted(
            self.emoticon_map.items(), key=lambda x: -len(x[0])
        )

        # Attempt to load optional spell checker — graceful degradation if absent.
        self._spell = None
        try:

            self._spell = SpellChecker()
            logger.info("SpellChecker loaded successfully.")
        except ImportError:
            logger.warning(
                "pyspellchecker not installed — Stage 6 (spell correction) "
                "will be skipped. Install with: pip install pyspellchecker"
            )

            
    def normalize_record(
        self,
        record: dict,
        apply_spellcheck: bool = False,
        spellcheck_sources: tuple[str, ...] = ("X", "Twitter"),
    ) -> dict:
        """
        Normalize a single JSON record in-place (also returns it).

        Adds new fields:
            Normalized_Text         — cleaned version of Text
            Normalized_Word_Count   — word count of Normalized_Text
            Mentions                — list of @mentions extracted from Text
            Hashtags                — list of #hashtags extracted from Text

        The original `Text` field is never modified (preserved for reproducibility).

        Parameters
        ----------
        record : dict matching the assignment JSON schema
        apply_spellcheck : whether to run Stage 6 at all
        spellcheck_sources : only apply spellcheck if record Source is in this tuple
        """
        source = record.get("Source", "")
        run_spell = apply_spellcheck and source in spellcheck_sources
 
        # HackerNews-specific: many posts store their body in Title with an empty Text field. If Text is empty and Title is non-empty, use Title
        # as the normalization input. Raw fields are never modified.
        raw_text  = (record.get("Text")  or "").strip()
        raw_title = (record.get("Title") or "").strip()
 
        if source == "HackerNews" and not raw_text and raw_title:
            text_to_normalize = raw_title
            record["Text_Source"] = "title"   # audit flag for reproducibility
        else:
            text_to_normalize = raw_text
            record["Text_Source"] = "text"
 
        record["Normalized_Text"] = self._normalize_text(
            text_to_normalize, record, run_spell
        )
        record["Normalized_Word_Count"] = len(record["Normalized_Text"].split())
 
        # Normalize comments recursively — comments are short-form so
        # spellcheck can be applied regardless of source.
        for comment in record.get("Comments") or []:
            comment["Normalized_Text"] = self._normalize_text(
                comment.get("Text", ""), comment, apply_spellcheck
            )
            comment["Normalized_Word_Count"] = len(comment["Normalized_Text"].split())
 
        return record

    def normalize_corpus(
        self,
        records: list[dict],
        apply_spellcheck: bool = False,
    ) -> list[dict]:
        """Convenience method to normalize an entire list of records."""
        for i, record in enumerate(records):
            try:
                self.normalize_record(record, apply_spellcheck)
            except Exception as e:
                logger.error(f"Failed to normalize record {record.get('ID', i)}: {e}")
        return records

    # ------------------------------------------------------------------
    # Loader helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        """Resolve data paths from either the current cwd or this module's folder."""
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate

        # Allow paths that are already correct from the caller's cwd.
        if candidate.exists():
            return candidate.resolve()

        # Fall back to resolving relative to this file's directory.
        return (Path(__file__).resolve().parent / candidate).resolve()

    def _load_emoticons(self, path: str | Path) -> dict[str, str]:
        """
        Load NetLingo emoticons JSON.
        Expected format: { ":p": "smiley with tongue hanging out", ... }
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} emoticons from {path}")
        return data

    def _load_slang(self, path: str | Path) -> dict[str, str]:
        """
        Load NetLingo acronyms JSON and resolve multi-meaning entries.
        Expected format: { "afk": "away from keyboard -or- a free kill", ... }

        Resolution strategy for "-or-" entries:
          1. If the acronym has a DOMAIN_OVERRIDE, use that (highest priority).
          2. Otherwise scan meanings for TECH_HINT_WORDS and prefer a tech meaning.
          3. If no tech hint matches, take the first listed meaning (most common).
        """
        with open(path, encoding="utf-8") as f:
            raw: dict[str, str] = json.load(f)

        resolved: dict[str, str] = {}
        for acronym, meaning_str in raw.items():
            key = acronym.lower().strip()

            # Skip anything in the tech allowlist — it should not be expanded.
            if key in self.tech_allowlist:
                continue

            resolved[key] = self._resolve_meaning(key, meaning_str)

        logger.info(f"Loaded {len(resolved)} slang/acronym entries from {path}")
        return resolved

    def _resolve_meaning(self, acronym: str, meaning_str: str) -> str:
        """
        Resolve a potentially ambiguous meaning string to a single expansion.

        Priority:
          1. DOMAIN_OVERRIDES (manually curated tech meanings)
          2. Meaning containing TECH_HINT_WORDS
          3. First meaning in the list (NetLingo's primary definition)
        """
        # 1. Domain override takes absolute priority.
        if acronym in self.domain_overrides:
            return self.domain_overrides[acronym]

        meanings = [m.strip() for m in meaning_str.split("-or-")]

        # Single meaning — nothing to resolve.
        if len(meanings) == 1:
            return meanings[0]

        # 2. Check each meaning for tech hint words.
        for meaning in meanings:
            words_in_meaning = set(meaning.lower().split())
            if words_in_meaning & TECH_HINT_WORDS:
                return meaning

        # 3. Default to first (most common) meaning.
        return meanings[0]

    # ------------------------------------------------------------------
    # Core normalization — applies all stages in order
    # ------------------------------------------------------------------

    def _normalize_text(self, text: str, record: dict, run_spell: bool) -> str:
        # Stage 0: Protect code blocks and strip Markdown structure.
        text, protected = self._stage0_extract_code_and_markdown(text)

        # Stage 1: Structural cleaning.
        text = self._stage1_structural_clean(text)

        # Stage 2: Extract mentions and hashtags into record fields.
        text = self._stage2_mentions_hashtags(text, record)

        # Stage 3: Emoji and emoticon handling.
        text = self._stage3_emoji_emoticon(text)

        # Stage 4: Elongated word normalization.
        text = self._stage4_elongated(text)

        # Stage 5: Acronym and slang expansion.
        text = self._stage5_acronyms(text)

        # Stage 6: Optional spelling correction.
        if run_spell:
            text = self._stage6_spellcheck(text)

        # Stage 7: Replace protected code placeholders with <CODE>.
        text = self._stage7_restore_code(text, protected)

        return text.strip()

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage0_extract_code_and_markdown(
        self, text: str
    ) -> tuple[str, dict[str, str]]:
        """
        Stage 0 — Code & Markdown Extraction.

        Replaces code blocks and inline code with opaque placeholder tokens
        so subsequent stages cannot corrupt technical content.
        Strips remaining Markdown formatting (bold, italic, headers, etc.).

        Returns (cleaned_text, {placeholder: original_snippet}).
        """
        protected: dict[str, str] = {}
        counter = [0]

        def protect(match: re.Match, prefix: str) -> str:
            token = f"XPROTX{prefix}{counter[0]}XPROTX"
            protected[token] = match.group(0)
            counter[0] += 1
            return f" {token} "

        # Fenced code blocks (``` or ~~~), including language specifiers.
        text = re.sub(
            r"```[\s\S]*?```",
            lambda m: protect(m, "CODEBLOCK"),
            text,
        )
        text = re.sub(
            r"~~~[\s\S]*?~~~",
            lambda m: protect(m, "CODEBLOCK"),
            text,
        )

        # Inline code (`single backtick`).
        text = re.sub(
            r"`[^`\n]+`",
            lambda m: protect(m, "CODEINLINE"),
            text,
        )

        # Strip remaining Markdown (order matters).
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)           # **bold**
        text = re.sub(r"\*(.*?)\*", r"\1", text)               # *italic*
        text = re.sub(r"__(.*?)__", r"\1", text)               # __underline__
        text = re.sub(r"~~(.*?)~~", r"\1", text)               # ~~strikethrough~~
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # ## headers
        text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)       # > blockquote
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)  # - bullet
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)  # 1. numbered
        text = re.sub(r"---+", " ", text)                      # horizontal rule

        return text, protected

    def _stage1_structural_clean(self, text: str) -> str:
        """
        Stage 1 — Structural Cleaning.
        Removes URLs, HTML entities/tags, and normalizes whitespace.
        """
        # Remove URLs.
        text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Unescape HTML entities (&amp; &lt; etc.) then strip any residual tags.
        text = unescape(text)
        text = re.sub(r"<[^>]+>", "", text)

        # Collapse multiple spaces/tabs to a single space.
        text = re.sub(r"[ \t]+", " ", text)

        # Collapse 3+ newlines to 2 (preserve paragraph breaks).
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _stage2_mentions_hashtags(self, text: str, record: dict) -> str:
        """
        Stage 2 — Mention & Hashtag Extraction.

        Extracts @mentions and #hashtags into `record['Mentions']` and
        `record['Hashtags']` respectively, then normalises them in the text:
          - Mentions are removed (stored for later use, e.g. network analysis).
          - Hashtags are split on CamelCase and replaced with plain words.
        """
        # Only add fields if not already present (avoid overwriting on comments).
        if "Mentions" not in record:
            record["Mentions"] = re.findall(r"@(\w+)", text)
        if "Hashtags" not in record:
            record["Hashtags"] = re.findall(r"#(\w+)", text)

        # Replace hashtags with space-separated words.
        text = re.sub(
            r"#(\w+)",
            lambda m: " " + self._split_camel(m.group(1)) + " ",
            text,
        )

        # Remove mentions entirely.
        text = re.sub(r"@\w+", " ", text)

        return text

    def _stage3_emoji_emoticon(self, text: str) -> str:
        """
        Stage 3 — Emoji & Emoticon Handling.

        Replaces emoticons (ASCII art) using the NetLingo smileys dataset,
        then converts Unicode emoji to their text descriptions using the
        `emoji` library.
        """
        # 1. Emoticons first (they are ASCII, so handle before emoji demojize
        #    to avoid double-processing). Use regex with whitespace boundaries instead of str.replace()
        for emoticon, description in self._sorted_emoticons:
            # Escape special regex chars in emoticon (e.g. :) ( * ^ )
            escaped = re.escape(emoticon)
            # Match only when surrounded by whitespace or start/end of string
            pattern = r"(?<!\S)" + escaped + r"(?!\S)"
            text = re.sub(pattern, f" [{description}] ", text)

        # 2. Unicode emoji → text description.
        try:
            text = emoji_lib.demojize(text, delimiters=("__EMOJI_", "__"))
            # Clean emoji tokens: __EMOJI_thumbs_up__ → "thumbs up"
            text = re.sub(
                r"__EMOJI_([a-z0-9_-]+)__",
                lambda m: " [" + m.group(1).replace("_", " ").replace("-", " ") + "] ",
                text,
            )
        except ImportError:
            logger.warning(
                "emoji library not installed — Unicode emoji will not be converted. "
                "Install with: pip install emoji"
            )

        return text

    def _stage4_elongated(self, text: str) -> str:
        """
        Stage 4 — Elongated Word Normalization.

        Reduces any character repeated 3+ times to exactly 2 repetitions.
        Keeping 2 (rather than 1) preserves some emphasis signal useful for
        sentiment analysis (e.g., "soooo good" → "soo good" still feels emphatic).

        Protected code tokens (__PROTECTED_...__) are unaffected because they
        contain no character repeated 3+ times in the token format itself.
        """
        return re.sub(r"(.)\1{2,}", r"\1\1", text)

    def _stage5_acronyms(self, text: str) -> str:
        """
        Stage 5 — Acronym & Slang Expansion.

        For each token:
          1. Skip if it is a protected placeholder (__PROTECTED_...__).
          2. Skip if it is in the tech allowlist (preserve as-is).
          3. Expand if it matches a resolved slang/acronym entry.
          4. Otherwise keep as-is.
        """
        tokens = text.split()
        expanded = []
        for token in tokens:
            # Never touch protected code placeholders.
            if token.startswith("__PROTECTED_"):
                expanded.append(token)
                continue

            # Strip trailing punctuation for lookup but preserve it after.
            stripped, punct = self._strip_punct(token)
            lower = stripped.lower()

            if lower in self.tech_allowlist:
                expanded.append(token)  # Keep original case.
            elif lower in self.slang_map:
                expanded.append(self.slang_map[lower] + punct)
            else:
                expanded.append(token)

        return " ".join(expanded)

    def _stage6_spellcheck(self, text: str) -> str:
        """
        Stage 6 — Spelling Correction (optional).

        Skips:
          - Protected code placeholders.
          - Tech allowlisted terms.
          - Very short tokens (≤ 2 chars) — too ambiguous to correct.
          - Tokens that are purely numeric or contain digits (version numbers etc.)
        """
        if self._spell is None:
            logger.warning("Spell checker not available — skipping Stage 6.")
            return text

        tokens = text.split()
        corrected = []
        for token in tokens:
            if token.startswith("__PROTECTED_"):
                corrected.append(token)
                continue

            stripped, punct = self._strip_punct(token)
            lower = stripped.lower()

            skip = (
                lower in self.tech_allowlist
                or len(stripped) <= 2
                or re.search(r"\d", stripped)   # skip tokens with digits
            )
            if skip:
                corrected.append(token)
                continue

            correction = self._spell.correction(stripped)
            corrected.append((correction if correction else stripped) + punct)

        return " ".join(corrected)

    def _stage7_restore_code(self, text: str, protected: dict[str, str]) -> str:
        """
        Stage 7 — Code Token Restoration.

        Replaces all __PROTECTED_*__ placeholders with the unified <CODE> token.
        The original snippets live in `protected` and are intentionally discarded
        in the normalized text — they are preserved in the raw `Text` field.
        """
        for token in protected:
            text = text.replace(token, "<CODE>")

        # Collapse any double <CODE> tokens that ended up adjacent.
        text = re.sub(r"(<CODE>\s*){2,}", "<CODE> ", text)

        return text

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _split_camel(s: str) -> str:
        """Split CamelCase hashtag text into space-separated words.
        '#GitHubCopilot' → 'Git Hub Copilot'
        """
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
        return s

    @staticmethod
    def _strip_punct(token: str) -> tuple[str, str]:
        """Separate trailing punctuation from a token for clean dict lookup.
        'lol!' → ('lol', '!')
        'great...' → ('great', '...')
        """
        match = re.match(r"^(.*?)([.,!?;:\"')\]]+)$", token)
        if match:
            return match.group(1), match.group(2)
        return token, ""


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    normalizer = MicrotextNormalizer(
        emoticons_path="../../../data/lexicons/emoticon_dict.json",
        slang_path="../../../data/lexicons/slang_dict.json",
    )

    sample_record = {
        "ID": "r_001",
        "Source": "Reddit",
        "Type": "Post",
        "Author": "dev_user_42",
        "Title": "GitHub Copilot is amazinggg!!!",
        "Text": (
            "tbh copilot has made me sooooo much more productive 😍 "
            "imo it's better than tabnine. check out https://github.com/features/copilot "
            "```python\ndef hello():\n    print('hello world')\n``` "
            "lol my PR got merged in 5 mins. @alice what do you think? #AITools #CodingLife"
        ),
        "Score": 142,
        "Date": "2024-11-01",
        "Word_Count": 45,
        "Comments": [
            {
                "comment_id": "c_001",
                "parent_id": "r_001",
                "Source": "Reddit",
                "Author": "alice_dev",
                "Text": "omg same!! afk rn but will try it l8r :) thx!!",
                "Score": 23,
                "Date": "2024-11-01",
                "Word_Count": 10,
            }
        ],
    }

    result = normalizer.normalize_record(sample_record, apply_spellcheck=False)

    print("=== Normalized Post Text ===")
    print(result["Normalized_Text"])
    print(f"\nMentions: {result['Mentions']}")
    print(f"Hashtags: {result['Hashtags']}")
    print(f"Word count: {result['Normalized_Word_Count']}")

    print("\n=== Normalized Comment Text ===")
    print(result["Comments"][0]["Normalized_Text"])