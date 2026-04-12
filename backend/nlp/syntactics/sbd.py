"""
sbd.py
SC4021 Information Retrieval 2026 — Sentence Boundary Disambiguation

Standard sentence tokenizers (including NLTK punkt) fail on several patterns
common in tech/developer text:

  1. Abbreviations    "vs.", "e.g.", "i.e.", "Dr.", "Fig.", "No."
  2. Version numbers  "Python 3.11.2 is great" — the dots are NOT boundaries
  3. File extensions  "install the .exe file" — not a boundary
  4. Ellipsis         "hmm..." followed by more text — ambiguous
  5. Code tokens      "<CODE>" from our pipeline — never a boundary
  6. Emoticon tokens  "[smiley face]" — never a boundary mid-sentence
  7. URL fragments    residual after Stage 1 cleaning — not boundaries
  8. All-caps words   "NLP. ML. AI." — likely a list, each IS a boundary
  9. Lowercase after  "it works. honestly though..." — IS a boundary despite
     period           the lowercase (our normalizer uppercases these, but
                      raw comments from HackerNews may not go through norm)

Pipeline position:
    MicrotextNormalizer → SBD → POSTagger → NERTagger → SubjectivityDetector

The SBD is used by SubjectivityDetector internally. You do not need to call
it manually — SubjectivityDetector accepts an SBD instance and delegates
sentence splitting to it.

Usage (standalone):
    sbd = SentenceBoundaryDisambiguator()
    sentences = sbd.split("Python 3.11.2 is fast. I use it vs. PyPy.")
    # ["Python 3.11.2 is fast.", "I use it vs. PyPy."]

Usage (integrated with SubjectivityDetector):
    from sbd import SentenceBoundaryDisambiguator
    sbd = SentenceBoundaryDisambiguator()
    detector = SubjectivityDetector(sbd=sbd)
"""

import re
import logging
from typing import Optional
import nltk

logger = logging.getLogger(__name__)


# Abbreviation lists

# These tokens followed by a period are NEVER sentence boundaries.
# All entries lowercase, without the trailing period.
ABBREVIATIONS: frozenset[str] = frozenset({
    # General English abbreviations
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc",
    "e.g", "i.e", "cf", "ca", "approx", "dept", "est", "fig",
    "no", "vol", "p", "pp", "ed", "eds", "repr", "rev", "sec",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "oct", "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    # Tech-specific abbreviations
    "ver", "v", "rel", "ref", "impl", "init", "config", "env",
    "min", "max", "avg", "std", "msg", "err", "req", "resp",
    "auth", "api", "sdk", "cli", "ui", "ux", "os", "db",
    "pkg", "lib", "src", "bin", "obj", "tmp", "dir",
    # Common in HackerNews / blog writing
    "approx", "incl", "excl", "esp", "re", "via", "nb",
})

# Regex patterns that, when matched at a potential boundary, suppress splitting.
# Each pattern is applied to the text around the candidate period.
_SUPPRESS_PATTERNS: list[re.Pattern] = [
    re.compile(r"\d+\.\d+"),
    # File extensions: .py, .js, .exe, .md
    re.compile(r"\.\w{1,5}(?=\s|$)"),
    # Ellipsis: ... or .. (keep together)
    re.compile(r"\.{2,}"),
    # Domain names / URLs (residual): github.com, huggingface.co
    re.compile(r"[a-z]+\.[a-z]{2,4}(?:[/\w]*)"),
    # Single uppercase letter abbreviations: U.S.A., N.L.P.
    re.compile(r"(?:[A-Z]\.){2,}"),
    # Code pipeline tokens — never split inside these
    re.compile(r"<CODE>"),
    re.compile(r"\[[^\]]+\]"),   # [smiley face], [heart]
]

# Regex to find candidate sentence boundaries: period/!/? followed by space
_CANDIDATE_BOUNDARY = re.compile(r"([.!?]+)\s+")

# Minimum token length to be considered a real sentence (filters empty splits)
_MIN_SENTENCE_TOKENS = 2


class SentenceBoundaryDisambiguator:
    """
    Rule-based sentence boundary disambiguator tailored for developer/tech text.

    Wraps NLTK's punkt tokenizer with a pre-pass that protects known
    non-boundary patterns, and a post-pass that merges incorrectly split
    fragments.

    Falls back to regex splitting if NLTK is unavailable.
    """

    def __init__(
        self,
        extra_abbreviations: Optional[set[str]] = None,
        min_sentence_tokens: int = _MIN_SENTENCE_TOKENS,
    ):
        """
        Parameters
        ----------
        extra_abbreviations : additional lowercase abbreviations (without period)
            to treat as non-boundaries, merged with the built-in ABBREVIATIONS set.
        min_sentence_tokens : sentences with fewer tokens than this are merged
            into the previous sentence (handles fragments like "Yes." or "Same.")
            unless they are the only sentence.
        """
        self.abbreviations: frozenset[str] = (
            ABBREVIATIONS | frozenset(a.lower() for a in extra_abbreviations)
            if extra_abbreviations else ABBREVIATIONS
        )
        self.min_tokens = min_sentence_tokens
        self._tokenizer = self._load_nltk()


    # Public APIs
    def split(self, text: str) -> list[str]:
        """
        Split text into sentences with boundary disambiguation applied.

        Returns a list of sentence strings. Empty/whitespace-only strings
        are filtered out. The pipeline tokens <CODE> and [emoticon] are
        never split across sentence boundaries.

        Parameters
        ----------
        text : normalized text (output of MicrotextNormalizer)
        """
        if not text or not text.strip():
            return []

        # Step 1: Protect pipeline tokens and known non-boundary patterns
        #         from the sentence splitter.
        protected_text, restore_map = self._protect_non_boundaries(text)

        # Step 2: Run sentence tokenizer on the protected text.
        raw_sentences = self._run_tokenizer(protected_text)

        # Step 3: Post-process — merge fragments, restore protected spans.
        sentences = self._postprocess(raw_sentences, restore_map)

        return sentences

    def tag_record(self, record: dict) -> dict:
        """
        Split Normalized_Text into sentences and store in record["Sentences"].
        Also processes all nested comments.
        Returns the modified record.
        """
        text = record.get("Normalized_Text", "")
        record["Sentences"] = self.split(text) if text else []

        for comment in record.get("Comments") or []:
            c_text = comment.get("Normalized_Text", "")
            comment["Sentences"] = self.split(c_text) if c_text else []

        return record

    def tag_corpus(self, records: list[dict]) -> list[dict]:
        """Split sentences for an entire list of records."""
        for i, record in enumerate(records):
            try:
                self.tag_record(record)
            except Exception as e:
                logger.error(f"SBD failed on record {record.get('ID', i)}: {e}")
        logger.info(f"SBD complete: {len(records)} records processed.")
        return records


    # Protection pass (pre-tokenization)
    def _protect_non_boundaries(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Replace non-boundary period patterns with placeholder strings that
        contain no punctuation, preventing the tokenizer from splitting on them.

        Returns (protected_text, {placeholder: original_span}).
        """
        restore_map: dict[str, str] = {}
        counter = [0]

        def protect(original: str, category: str) -> str:
            placeholder = f"SBDPROTECT{category}{counter[0]}END"
            restore_map[placeholder] = original
            counter[0] += 1
            return placeholder

        # 1. Protect <CODE> tokens
        text = re.sub(
            r"<CODE>",
            lambda m: protect(m.group(0), "CODE"),
            text,
        )

        # 2. Protect [emoticon] tokens
        text = re.sub(
            r"\[[^\]]+\]",
            lambda m: protect(m.group(0), "EMOT"),
            text,
        )

        # 3. Protect version numbers: 3.11, v1.0.3, Python 3.11.2
        text = re.sub(
            r"\b([vV]?\d+(?:\.\d+)+)\b",
            lambda m: protect(m.group(0), "VER"),
            text,
        )

        # 4. Protect known abbreviations: "vs.", "e.g.", "i.e."
        #    Match word at end of token list followed by period and space/end.
        def protect_abbrev(match: re.Match) -> str:
            word = match.group(1).lower()
            period = match.group(2)
            after = match.group(3)
            if word in self.abbreviations:
                original = match.group(1) + period
                return protect(original, "ABBR") + after
            return match.group(0)

        text = re.sub(
            r"\b(\w+)(\.)([\s]|$)",
            protect_abbrev,
            text,
        )

        # 5. Protect ellipsis (2+ dots)
        text = re.sub(
            r"\.{2,}",
            lambda m: protect(m.group(0), "ELPS"),
            text,
        )

        # 6. Protect file extensions: .py, .js, .exe
        text = re.sub(
            r"(?<=\S)(\.\w{1,5})(?=\s|$)",
            lambda m: protect(m.group(0), "EXT"),
            text,
        )

        # 7. Protect domain-like patterns: github.com, huggingface.co
        text = re.sub(
            r"\b([a-zA-Z0-9\-]+\.[a-z]{2,6})(?:/\S*)?\b",
            lambda m: protect(m.group(0), "DOM"),
            text,
        )

        return text, restore_map


    # Tokenization
    def _run_tokenizer(self, text: str) -> list[str]:
        """Run the sentence tokenizer on pre-processed text."""
        if self._tokenizer:
            return self._tokenizer(text)
        return self._regex_split(text)

    def _regex_split(self, text: str) -> list[str]:
        """
        Fallback regex splitter used when NLTK is unavailable.
        Splits on .  !  ? followed by whitespace and an uppercase letter.
        """
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [p.strip() for p in parts if p.strip()]


    # Post-processing pass
    def _postprocess(
        self,
        sentences: list[str],
        restore_map: dict[str, str],
    ) -> list[str]:
        """
        1. Restore all protected placeholders back to their original text.
        2. Merge very short sentences (fragments) into the previous sentence.
        3. Filter empty strings.
        """
        # Restore placeholders
        restored: list[str] = []
        for sentence in sentences:
            for placeholder, original in restore_map.items():
                sentence = sentence.replace(placeholder, original)
            restored.append(sentence.strip())

        # Merge fragments: sentences below min_tokens are appended to previous.
        # Exception: if there is only one sentence, always keep it.
        if len(restored) <= 1:
            return [s for s in restored if s]

        merged: list[str] = []
        for sentence in restored:
            if not sentence:
                continue
            token_count = len(sentence.split())
            if token_count < self.min_tokens and merged:
                # Append fragment to previous sentence with a space
                merged[-1] = merged[-1].rstrip() + " " + sentence
            else:
                merged.append(sentence)

        return [s for s in merged if s.strip()]


    # NLTK loader
    def _load_nltk(self):
        """
        Load NLTK punkt tokenizer. Returns the tokenize function or None
        if NLTK is unavailable (graceful degradation to regex fallback).
        """
        try:
            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                logger.info("Downloading NLTK punkt_tab tokenizer...")
                nltk.download("punkt_tab", quiet=True)
            logger.info("SBD: NLTK punkt tokenizer ready.")
            return nltk.sent_tokenize
        except ImportError:
            logger.warning(
                "SBD: nltk not installed — using regex fallback splitter. "
                "Install with: pip install nltk"
            )
            return None


# Smoke test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sbd = SentenceBoundaryDisambiguator()

    test_cases = [
        # Basic split
        (
            "GitHub Copilot was released in 2021. It changed how I write code.",
            ["GitHub Copilot was released in 2021.", "It changed how I write code."]
        ),
        # Version numbers — should NOT split
        (
            "I upgraded to Python 3.11.2. It is much faster.",
            ["I upgraded to Python 3.11.2.", "It is much faster."]
        ),
        # Abbreviation — should NOT split on "vs."
        (
            "Copilot vs. Tabnine is a common debate. Both are useful.",
            ["Copilot vs. Tabnine is a common debate.", "Both are useful."]
        ),
        # Ellipsis — should NOT split mid-thought
        (
            "hmm... I think copilot is better. Tabnine feels slow.",
            ["hmm... I think copilot is better.", "Tabnine feels slow."]
        ),
        # Code token — never a boundary
        (
            "use <CODE> to install dependencies. Then run the tests.",
            ["use <CODE> to install dependencies.", "Then run the tests."]
        ),
        # Emoticon token — never a boundary
        (
            "I love this tool [smiley face]. It saves so much time.",
            ["I love this tool [smiley face].", "It saves so much time."]
        ),
        # Fragment merging — "Yes." should merge with next sentence
        (
            "Did it work? Yes. I ran the tests and everything passed.",
            2  # expect 2 sentences (Yes. merged into next)
        ),
        # Domain name — should NOT split on "github.com"
        (
            "Check out github.com for the source. It is open source.",
            ["Check out github.com for the source.", "It is open source."]
        ),
        # Multiple sentences
        (
            "I think copilot is great. It helps me write code faster. "
            "However, it sometimes hallucinates. I still recommend it.",
            4
        ),
    ]

    print("=" * 65)
    print("SBD SMOKE TEST")
    print("=" * 65)

    passed = 0
    for i, test in enumerate(test_cases, 1):
        if len(test) == 2:
            text, expected = test
        else:
            text, expected = test[0], test[1]

        result = sbd.split(text)

        if isinstance(expected, int):
            ok = len(result) == expected
            detail = f"expected {expected} sentences, got {len(result)}"
        else:
            ok = result == expected
            detail = f"expected {expected}" if not ok else "ok"

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] Test {i}: {text[:55]}{'...' if len(text) > 55 else ''}")
        if not ok:
            print(f"         → {detail}")
            print(f"         → got: {result}")

    print(f"\n{passed}/{len(test_cases)} tests passed.")
    print("=" * 65)