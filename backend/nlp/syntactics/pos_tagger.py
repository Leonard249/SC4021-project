"""
pos_tagger.py
SC4021 Information Retrieval 2026 — POS Tagging & NER Module

Two modular classes for the syntactics layer of the pipeline:

    POSTagger  — POS tagging on Normalized_Text (stage 2)
    NERTagger  — Named entity recognition on Normalized_Text (stage 3)

Both classes handle pipeline-specific tokens (<CODE> and [bracket emoticons])
by intercepting them before spaCy sees them and reinserting them with
custom labels afterwards.

Fields added by POSTagger:
    POS_Tags  — list of [token, POS, lemma] triples
                e.g. [["copilot", "NOUN", "copilot"], ["is", "AUX", "be"], ...]

    Special POS tags (not standard spaCy):
        <CODE>          → "CODE"
        [smiley face]   → "EMOTICON"

Fields added by NERTagger:
    NER_Tags  — list of [text, label, start_char, end_char] for each entity
                e.g. [["GitHub Copilot", "PRODUCT", 10, 23], ...]

    Standard spaCy NER labels used:
        PRODUCT  — AI tools, editors, software (Copilot, VS Code, Tabnine)
        ORG      — companies (GitHub, OpenAI, Anthropic, JetBrains)
        PERSON   — people (Linus Torvalds, Sam Altman)
        GPE      — geopolitical entities (countries, cities)
        DATE     — dates and time expressions
        CARDINAL — numbers

    Custom labels added via EntityRuler (domain-specific):
        AI_TOOL  — AI coding assistants not in spaCy's vocab
        PL       — programming languages
        EDITOR   — code editors and IDEs

Requires:
    pip install spacy
    python -m spacy download en_core_web_sm
"""

import re
import logging
from typing import Optional
import spacy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared module-level utilities (used by both POSTagger and NERTagger)
# ---------------------------------------------------------------------------

# Matches pipeline-injected tokens that spaCy should never see.
# Group 1: <CODE>
# Group 2: [bracket emoticon text]
_SPECIAL_TOKEN_RE = re.compile(r"(<CODE>|\[[^\]]+\])")


def _extract_special_tokens(text: str) -> tuple[str, dict[str, tuple[str, str]]]:
    """
    Replace pipeline-specific tokens with neutral placeholders before
    passing text to spaCy so it doesn't try to parse or tag them.

    <CODE>         → CODEPLACEHOLDER0, CODEPLACEHOLDER1, ...
    [smiley face]  → EMOTICONPLACEHOLDER0, EMOTICONPLACEHOLDER1, ...

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
# Domain-specific entity patterns for EntityRuler
# These cover AI coding tools, companies, editors, and languages that
# spaCy's statistical NER model does not recognise out of the box.
# Add new entries here as your corpus grows.
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
    {"label": "ORG", "pattern": [{"LOWER": "deep"}, {"LOWER": "mind"}]},
    {"label": "ORG", "pattern": "Meta"},
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
]


class POSTagger:
    """
    Modular POS tagging stage for the SC4021 normalization pipeline.

    Usage
    -----
    tagger = POSTagger()
    record = tagger.tag_record(record)
    # record["POS_Tags"] is now a list of [token, POS, lemma] triples.

    Or for a full corpus:
    records = tagger.tag_corpus(records)
    """

    def __init__(self, model: str = "en_core_web_sm"):
        """
        Parameters
        ----------
        model : spaCy model name.
            en_core_web_sm  — fast, sufficient for most cases (recommended)
            en_core_web_md  — adds word vectors, better OOV handling
            en_core_web_lg  — largest, most accurate, slowest
        """
        try:
            self._nlp = spacy.load(model)
            logger.info(f"Loaded spaCy model: {model}")
        except OSError:
            raise OSError(
                f"spaCy model '{model}' not found. "
                f"Install it with: python -m spacy download {model}"
            )

        # Disable pipeline components we don't need — speeds up batch tagging.
        # We keep 'tagger' (POS) and 'attribute_ruler'/'lemmatizer' (lemmas).
        # 'parser' and 'ner' are skipped here; NER may be added in a later stage.
        self._nlp.disable_pipes(
            [p for p in ["parser", "ner"] if p in self._nlp.pipe_names]
        )
        logger.info(f"Active spaCy pipes: {self._nlp.pipe_names}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag_record(self, record: dict) -> dict:
        """
        POS-tag a single record and all its nested comments in-place.

        Requires 'Normalized_Text' to be present (run MicrotextNormalizer first).
        Adds 'POS_Tags' to the record and to each comment.

        Returns the modified record.
        """
        text = record.get("Normalized_Text", "")
        if not text:
            logger.warning(f"Record {record.get('ID', '?')} has no Normalized_Text — skipping.")
            record["POS_Tags"] = []
            return record

        record["POS_Tags"] = self._tag_text(text)

        for comment in record.get("Comments") or []:
            comment_text = comment.get("Normalized_Text", "")
            comment["POS_Tags"] = self._tag_text(comment_text) if comment_text else []

        return record

    def tag_corpus(self, records: list[dict]) -> list[dict]:
        """
        POS-tag an entire list of records.

        Uses spaCy's nlp.pipe() for batch processing — significantly faster
        than calling tag_record() in a loop for large corpora.
        """
        # Collect all texts (post + comments) with their destination dicts
        # so we can write results back after batch processing.
        targets: list[tuple[dict, str]] = []  # (container_dict, field_key)
        texts: list[str] = []

        for record in records:
            post_text = record.get("Normalized_Text", "")
            targets.append((record, "POS_Tags"))
            texts.append(post_text)

            for comment in record.get("Comments") or []:
                comment_text = comment.get("Normalized_Text", "")
                targets.append((comment, "POS_Tags"))
                texts.append(comment_text)

        # Pre-process: extract special tokens before spaCy sees the texts.
        cleaned_texts, special_maps = zip(
            *[_extract_special_tokens(t) for t in texts]
        ) if texts else ([], [])

        # Batch tag with spaCy.
        logger.info(f"Batch tagging {len(texts)} texts...")
        docs = list(self._nlp.pipe(cleaned_texts, batch_size=256))

        # Write results back, reinserting special tokens.
        for (container, field), doc, special_map in zip(targets, docs, special_maps):
            container[field] = self._build_tag_list(doc, special_map)

        logger.info("Batch tagging complete.")
        return records

    # ------------------------------------------------------------------
    # Core tagging logic
    # ------------------------------------------------------------------

    def _tag_text(self, text: str) -> list[list[str]]:
        """Tag a single text string. Used by tag_record()."""
        cleaned, special_map = _extract_special_tokens(text)
        doc = self._nlp(cleaned)
        return self._build_tag_list(doc, special_map)

    def _build_tag_list(
        self,
        doc,  # spaCy Doc
        special_map: dict[str, tuple[str, str]],
    ) -> list[list[str]]:
        """
        Convert a spaCy Doc into a list of [token, POS, lemma] triples,
        restoring any special-token placeholders with their custom tags.

        POS tags follow the Universal Dependencies tagset (spaCy default):
            NOUN, VERB, ADJ, ADV, PRON, DET, ADP, AUX, CCONJ, SCONJ,
            PROPN, NUM, PUNCT, SYM, X, INTJ, PART, SPACE
        Plus our custom tags:
            CODE     — for <CODE> tokens
            EMOTICON — for [bracket emoticon] tokens
        """
        result: list[list[str]] = []

        for token in doc:
            surface = token.text

            if surface in special_map:
                # Restore the original token text and apply the custom tag.
                original, custom_tag = special_map[surface]
                result.append([original, custom_tag, original])
            else:
                # Standard spaCy token: use Universal POS tag + lemma.
                # Skip pure whitespace tokens.
                if token.pos_ == "SPACE":
                    continue
                result.append([surface, token.pos_, token.lemma_])

        return result


class NERTagger:
    """
    Modular NER stage for the SC4021 normalization pipeline.

    Runs spaCy's statistical NER model augmented with an EntityRuler containing
    domain-specific patterns for AI coding tools, companies, editors, and
    programming languages that the base model does not recognise.

    The EntityRuler runs BEFORE the statistical NER model, so domain patterns
    always take priority over spaCy's predictions (e.g., "Cursor" is tagged
    AI_TOOL rather than being misread as a generic noun).

    Usage
    -----
    ner = NERTagger()
    record = ner.tag_record(record)
    # record["NER_Tags"] is now a list of [text, label, start_char, end_char].

    Or for a full corpus:
    records = ner.tag_corpus(records)
    """

    def __init__(
        self,
        model: str = "en_core_web_sm",
        extra_patterns: Optional[list[dict]] = None,
    ):
        """
        Parameters
        ----------
        model : spaCy model name (same options as POSTagger).
        extra_patterns : additional EntityRuler patterns to merge with
            DOMAIN_ENTITY_PATTERNS. Useful for extending the entity list
            without modifying this file.
            Format: [{"label": "AI_TOOL", "pattern": "MyNewTool"}, ...]
        """

        try:
            self._nlp = spacy.load(model)
            logger.info(f"NERTagger: loaded spaCy model '{model}'")
        except OSError:
            raise OSError(
                f"spaCy model '{model}' not found. "
                f"Install it with: python -m spacy download {model}"
            )

        # Keep only the pipes needed for NER.
        # tok2vec feeds into ner, so keep it. Disable POS/lemma pipes since
        # POS tagging is handled by POSTagger in the prior stage.
        self._nlp.disable_pipes(
            [p for p in ["tagger", "attribute_ruler", "lemmatizer", "parser"]
             if p in self._nlp.pipe_names]
        )

        # Add EntityRuler BEFORE the statistical NER so domain patterns win.
        # overwrite_ents=True means the ruler's labels replace spaCy's guesses
        # when they conflict.
        ruler = self._nlp.add_pipe(
            "entity_ruler",
            after="ner",
            config={"overwrite_ents": True},
        )
        patterns = DOMAIN_ENTITY_PATTERNS + (extra_patterns or [])
        ruler.add_patterns(patterns)
        logger.info(
            f"NERTagger: EntityRuler loaded with {len(patterns)} domain patterns"
        )
        logger.info(f"NERTagger: active pipes: {self._nlp.pipe_names}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag_record(self, record: dict) -> dict:
        """
        NER-tag a single record and all its nested comments in-place.

        Requires 'Normalized_Text' to be present (run MicrotextNormalizer first).
        Adds 'NER_Tags' to the record and to each comment.

        Returns the modified record.
        """
        text = record.get("Normalized_Text", "")
        if not text:
            logger.warning(f"Record {record.get('ID', '?')} has no Normalized_Text — skipping.")
            record["NER_Tags"] = []
            return record

        record["NER_Tags"] = self._tag_text(text)

        for comment in record.get("Comments") or []:
            comment_text = comment.get("Normalized_Text", "")
            comment["NER_Tags"] = self._tag_text(comment_text) if comment_text else []

        return record

    def tag_corpus(self, records: list[dict]) -> list[dict]:
        """
        NER-tag an entire list of records using spaCy's batch pipe for speed.
        """
        targets: list[tuple[dict, str]] = []
        texts: list[str] = []

        for record in records:
            targets.append((record, "NER_Tags"))
            texts.append(record.get("Normalized_Text", ""))
            for comment in record.get("Comments") or []:
                targets.append((comment, "NER_Tags"))
                texts.append(comment.get("Normalized_Text", ""))

        cleaned_texts, special_maps = zip(
            *[_extract_special_tokens(t) for t in texts]
        ) if texts else ([], [])

        logger.info(f"NERTagger: batch tagging {len(texts)} texts...")
        docs = list(self._nlp.pipe(cleaned_texts, batch_size=256))

        for (container, field), doc, special_map in zip(targets, docs, special_maps):
            container[field] = self._build_ner_list(doc, special_map)

        logger.info("NERTagger: batch tagging complete.")
        return records

    # ------------------------------------------------------------------
    # Core NER logic
    # ------------------------------------------------------------------

    def _tag_text(self, text: str) -> list[list]:
        """NER-tag a single text string."""
        cleaned, special_map = _extract_special_tokens(text)
        doc = self._nlp(cleaned)
        return self._build_ner_list(doc, special_map)

    def _build_ner_list(self, doc, special_map: dict[str, tuple[str, str]]) -> list[list]:
        """
        Build the NER_Tags list from a spaCy Doc.

        Each entry is [text, label, start_char, end_char].
        start_char and end_char refer to character offsets in Normalized_Text
        (after placeholder substitution), which is sufficient for downstream
        use in aspect detection.

        Special tokens (<CODE>, [emoticon]) are included if they appear as
        named spans — in practice they won't, since placeholders are opaque
        to spaCy, but the map is checked for safety.
        """
        result: list[list] = []

        for ent in doc.ents:
            surface = ent.text

            # If the entire span is a placeholder, restore the original token.
            if surface in special_map:
                original, custom_label = special_map[surface]

                # If this span was tagged as an entity by spaCy but is actually a special token,
                # we trust the special token label over spaCy's guess and skip it.
                if custom_label in ("CODE", "EMOTICON"):
                    continue
                result.append([original, custom_label, ent.start_char, ent.end_char])
            else:
                # Restore any placeholders that ended up inside a multi-token span
                # (edge case, but handles e.g. "OpenAI <CODE> model").
                restored = surface
                for placeholder, (original, _) in special_map.items():
                    restored = restored.replace(placeholder, original)
                result.append([restored, ent.label_, ent.start_char, ent.end_char])

        return result


# ---------------------------------------------------------------------------
# Example usage / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample_record = {
        "ID": "r_001",
        "Source": "Reddit",
        "Normalized_Text": (
            "to be honest copilot has made me soo much more productive "
            "[smiling face with heart eyes] in my opinion it is better than tabnine . "
            "check out <CODE> laughing out loud my PR got merged in 5 minutes . "
            "GitHub Copilot and ChatGPT are both made by OpenAI and Microsoft . "
            "I use VS Code with Python and TypeScript every day ."
        ),
        "Comments": [
            {
                "comment_id": "c_001",
                "Normalized_Text": (
                    "oh my god same !! away from keyboard right now "
                    "but will try Cursor later [smiley face] thanks !!"
                ),
            }
        ],
    }

    # --- POS Tagging ---
    pos_tagger = POSTagger(model="en_core_web_sm")
    sample_record = pos_tagger.tag_record(sample_record)

    print("\n=== POS Tags (Post) ===")
    for token, pos, lemma in sample_record["POS_Tags"]:
        print(f"  {token:<40} {pos:<12} {lemma}")

    print("\n=== POS Tags (Comment) ===")
    for token, pos, lemma in sample_record["Comments"][0]["POS_Tags"]:
        print(f"  {token:<40} {pos:<12} {lemma}")

    # --- NER Tagging ---
    ner_tagger = NERTagger(model="en_core_web_sm")
    sample_record = ner_tagger.tag_record(sample_record)


    print("\n=== NER Tags (Post) ===")
    print(f"  {'Entity':<35} {'Label':<12} {'Start':>6} {'End':>6}")
    print(f"  {'-'*35} {'-'*12} {'-'*6} {'-'*6}")
    for text, label, start, end in sample_record["NER_Tags"]:
        print(f"  {text:<35} {label:<12} {start:>6} {end:>6}")

    print("\n=== NER Tags (Comment) ===")
    print(f"  {'Entity':<35} {'Label':<12} {'Start':>6} {'End':>6}")
    print(f"  {'-'*35} {'-'*12} {'-'*6} {'-'*6}")
    for text, label, start, end in sample_record["Comments"][0]["NER_Tags"]:
        print(f"  {text:<35} {label:<12} {start:>6} {end:>6}")