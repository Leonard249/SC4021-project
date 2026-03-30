"""
spacy_utils.py
SC4021 Information Retrieval 2026 — Shared spaCy Utilities

Shared constants and helpers imported by both pos_tagger.py and ner_tagger.py.
Do not import pipeline-specific logic here — this module must remain stateless.

Exports:
    _SPECIAL_TOKEN_RE       — compiled regex matching <CODE> and [emoticon] tokens
    _extract_special_tokens — replaces pipeline tokens with spaCy-safe placeholders
    DOMAIN_ENTITY_PATTERNS  — EntityRuler patterns for AI tools, orgs, editors, PLs
    CONTENT_POS             — POS tags considered content words (used by POSTagger)
"""

import re

# ---------------------------------------------------------------------------
# Pipeline token handling
# ---------------------------------------------------------------------------

# Matches <CODE> and [bracket emoticon] tokens injected by MicrotextNormalizer.
_SPECIAL_TOKEN_RE = re.compile(r"(<CODE>|\[[^\]]+\])")


def _extract_special_tokens(text: str) -> tuple[str, dict[str, tuple[str, str]]]:
    """
    Replace pipeline-specific tokens with neutral placeholders before passing
    text to spaCy, so spaCy does not try to parse or tag them.

        <CODE>         → CODEPLACEHOLDERn
        [smiley face]  → EMOTICONPLACEHOLDERn

    Returns
    -------
    cleaned_text : str with placeholders substituted in
    special_map  : { placeholder: (original_token, custom_label) }
    """
    special_map: dict[str, tuple[str, str]] = {}
    counter = [0]

    def replace(match: re.Match) -> str:
        token = match.group(0)
        if token == "<CODE>":
            placeholder = f"CODEPLACEHOLDER{counter[0]}"
            label = "CODE"
        else:
            placeholder = f"EMOTICONPLACEHOLDER{counter[0]}"
            label = "EMOTICON"
        special_map[placeholder] = (token, label)
        counter[0] += 1
        return placeholder

    cleaned = _SPECIAL_TOKEN_RE.sub(replace, text)
    return cleaned, special_map


# ---------------------------------------------------------------------------
# Content POS filter
# ---------------------------------------------------------------------------

# Only these Universal Dependencies POS tags are stored in POS_Tags.
# Function words (DET, ADP, AUX, CCONJ, SCONJ, PART), punctuation, and
# whitespace are excluded — no downstream task in this pipeline needs them,
# and excluding them reduces POS_Tags size by ~40-50% for long blog posts.
CONTENT_POS: frozenset[str] = frozenset({
    "NOUN",   # common nouns: code, feature, bug
    "PROPN",  # proper nouns: GitHub, Copilot, Microsoft
    "VERB",   # verbs: improve, generate, suggest
    "ADJ",    # adjectives: amazing, buggy, faster
    "ADV",    # adverbs: honestly, incredibly, always
    "INTJ",   # interjections: oh, wow, ugh (sentiment signal)
    "NUM",    # numbers: sometimes relevant for benchmarks/stats
    # Custom pipeline tags — always kept regardless of filter
    "CODE",
    "EMOTICON",
})


# ---------------------------------------------------------------------------
# Domain entity patterns for NERTagger's EntityRuler
# ---------------------------------------------------------------------------

DOMAIN_ENTITY_PATTERNS: list[dict] = [
    # ------------------------------------------------------------------
    # AI_TOOL — AI coding assistants and LLMs
    # ------------------------------------------------------------------
    {"label": "AI_TOOL", "pattern": [{"LOWER": "copilot"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "github"}, {"LOWER": "copilot"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "chatgpt"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "claude"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "gemini"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "tabnine"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "codeium"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "cursor"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "devin"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "aider"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "replit"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "cody"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "sourcegraph"}, {"LOWER": "cody"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "amazon"}, {"LOWER": "codewhisperer"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "codewhisperer"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "blackbox"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "mistral"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "llama"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "gpt-4"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "gpt-4o"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "gpt-3.5"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "o1"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "o3"}]},
    {"label": "AI_TOOL", "pattern": [{"LOWER": "windsurf"}]},

    # ------------------------------------------------------------------
    # ORG — companies and platforms in the AI/dev space
    # ------------------------------------------------------------------
    {"label": "ORG", "pattern": [{"LOWER": "openai"}]},
    {"label": "ORG", "pattern": [{"LOWER": "anthropic"}]},
    {"label": "ORG", "pattern": [{"LOWER": "github"}]},
    {"label": "ORG", "pattern": [{"LOWER": "gitlab"}]},
    {"label": "ORG", "pattern": [{"LOWER": "jetbrains"}]},
    {"label": "ORG", "pattern": [{"LOWER": "microsoft"}]},
    {"label": "ORG", "pattern": [{"LOWER": "google"}]},
    {"label": "ORG", "pattern": [{"LOWER": "deepmind"}]},
    {"label": "ORG", "pattern": [{"LOWER": "deep"}, {"LOWER": "mind"}]},
    {"label": "ORG", "pattern": [{"TEXT": "Meta"}]},
    {"label": "ORG", "pattern": [{"LOWER": "amazon"}]},
    {"label": "ORG", "pattern": [{"LOWER": "aws"}]},
    {"label": "ORG", "pattern": [{"LOWER": "hugging"}, {"LOWER": "face"}]},
    {"label": "ORG", "pattern": [{"LOWER": "stack"}, {"LOWER": "overflow"}]},

    # ------------------------------------------------------------------
    # EDITOR — code editors and IDEs
    # ------------------------------------------------------------------
    {"label": "EDITOR", "pattern": [{"LOWER": "vs"}, {"LOWER": "code"}]},
    {"label": "EDITOR", "pattern": [{"LOWER": "vscode"}]},
    {"label": "EDITOR", "pattern": [{"LOWER": "visual"}, {"LOWER": "studio"}, {"LOWER": "code"}]},
    {"label": "EDITOR", "pattern": [{"LOWER": "visual"}, {"LOWER": "studio"}]},
    {"label": "EDITOR", "pattern": [{"LOWER": "neovim"}]},
    {"label": "EDITOR", "pattern": [{"LOWER": "vim"}]},
    {"label": "EDITOR", "pattern": [{"LOWER": "emacs"}]},
    {"label": "EDITOR", "pattern": [{"LOWER": "intellij"}]},
    {"label": "EDITOR", "pattern": [{"LOWER": "pycharm"}]},
    {"label": "EDITOR", "pattern": [{"LOWER": "webstorm"}]},
    {"label": "EDITOR", "pattern": [{"LOWER": "clion"}]},
    {"label": "EDITOR", "pattern": [{"LOWER": "xcode"}]},
    {"label": "EDITOR", "pattern": [{"LOWER": "eclipse"}]},
    {"label": "EDITOR", "pattern": [{"LOWER": "sublime"}, {"LOWER": "text"}]},

    # ------------------------------------------------------------------
    # PL — programming languages
    # ------------------------------------------------------------------
    {"label": "PL", "pattern": [{"LOWER": "python"}]},
    {"label": "PL", "pattern": [{"LOWER": "typescript"}]},
    {"label": "PL", "pattern": [{"LOWER": "javascript"}]},
    {"label": "PL", "pattern": [{"LOWER": "rust"}]},
    {"label": "PL", "pattern": [{"LOWER": "golang"}]},
    {"label": "PL", "pattern": [{"LOWER": "java"}]},
    {"label": "PL", "pattern": [{"LOWER": "kotlin"}]},
    {"label": "PL", "pattern": [{"LOWER": "swift"}]},
    {"label": "PL", "pattern": [{"LOWER": "c++"}]},
    {"label": "PL", "pattern": [{"LOWER": "c#"}]},
    {"label": "PL", "pattern": [{"LOWER": "ruby"}]},
    {"label": "PL", "pattern": [{"LOWER": "php"}]},
    {"label": "PL", "pattern": [{"LOWER": "scala"}]},
    {"label": "PL", "pattern": [{"LOWER": "haskell"}]},
    {"label": "PL", "pattern": [{"LOWER": "elixir"}]},

    # ------------------------------------------------------------------
    # TECH_CONCEPT — General AI/ML fields
    # ------------------------------------------------------------------
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "ai"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "ml"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "nlp"}]},
    {"label": "TECH_CONCEPT", "pattern": [{"LOWER": "llm"}]},

]

DYNAMIC_POS_PATTERNS: list[dict] = [
    # Catches "[Unknown Proper Noun] model" -> e.g., "Grok model", "Cohere model"
    {"label": "AI_TOOL", "pattern": [{"POS": "PROPN"}, {"LOWER": "model"}]},
    {"label": "AI_TOOL", "pattern": [{"POS": "PROPN"}, {"POS": "PROPN", "OP": "?"}, {"LOWER": "model"}]},
    
    # Catches "[Unknown Proper Noun] API" -> e.g., "Anthropic API"
    {"label": "AI_TOOL", "pattern": [{"POS": "PROPN"}, {"LOWER": "api"}]},
    
    # Catches "[Unknown Proper Noun] extension/plugin" -> e.g., "Bito extension"
    {"label": "AI_TOOL", "pattern": [{"POS": "PROPN"}, {"LOWER": {"IN": ["extension", "plugin"]}}]},
    
    # Catches IDEs: "[Unknown Proper Noun] editor" -> e.g., "Zed editor"
    {"label": "EDITOR", "pattern": [{"POS": "PROPN"}, {"LOWER": "editor"}]}
]