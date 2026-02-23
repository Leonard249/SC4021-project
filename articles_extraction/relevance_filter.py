"""
Stage 3: LLM Relevance Filtering

Reads extracted articles from Stage 2 and classifies each as relevant
or irrelevant to "vibe coding" using an LLM (Groq).

Usage:
    python relevance_filter.py

Input:
    raw_articles/raw_articles.csv

Output:
    filtered_articles/filtered_articles.csv
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seeds_config import RAW_ARTICLES_DIR, FILTERED_ARTICLES_DIR
from utils import (
    setup_logger,
    load_articles_from_csv,
    save_articles_to_csv,
    ensure_dir,
)

logger = setup_logger("relevance_filter", os.path.join(FILTERED_ARTICLES_DIR, "filtering.log"))


# =============================================================================
# Configuration
# =============================================================================
# Change this to your preferred model/provider
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Fast and free on Groq
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

RELEVANCE_PROMPT = """You are classifying articles for relevance to "Vibe Coding".

Vibe coding refers to the practice of using AI tools (like Cursor, GitHub Copilot, ChatGPT, Claude, Windsurf, Replit Agent, etc.) to assist with or generate code, often with minimal manual coding. Related topics include:
- AI pair programming
- LLM-assisted development
- Agentic coding
- Prompt-driven development
- AI code generation tools and experiences
- Discussions about the impact of AI on software development workflows

Title: {title}
Content (first 500 words): {content_preview}

Is this article relevant to vibe coding? Reply with ONLY a JSON object:
{{"relevant": true/false, "confidence": 0.0-1.0, "reason": "one sentence explanation"}}"""


# =============================================================================
# LLM Classification
# =============================================================================
def classify_with_groq(title: str, content: str) -> dict:
    """
    Classify an article's relevance using Groq API.
    Returns {"relevant": bool, "confidence": float, "reason": str}
    """
    import requests

    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY environment variable not set!")
        logger.error("Set it with: export GROQ_API_KEY='your-key-here'")
        sys.exit(1)

    # Take first 500 words for classification
    words = content.split()
    content_preview = " ".join(words[:500])

    prompt = RELEVANCE_PROMPT.format(title=title, content_preview=content_preview)

    try:
        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 150,
            },
            timeout=30,
        )
        response.raise_for_status()
        result_text = response.json()["choices"][0]["message"]["content"].strip()

        # Parse JSON response
        # Handle potential markdown code blocks in response
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]

        result = json.loads(result_text)
        return {
            "relevant": bool(result.get("relevant", False)),
            "confidence": float(result.get("confidence", 0.0)),
            "reason": str(result.get("reason", "")),
        }

    except json.JSONDecodeError:
        # If LLM didn't return valid JSON, try to parse the text
        result_lower = result_text.lower()
        relevant = "true" in result_lower or '"relevant": true' in result_lower
        return {
            "relevant": relevant,
            "confidence": 0.5,
            "reason": f"Parsed from non-JSON response: {result_text[:100]}",
        }

    except Exception as e:
        logger.warning(f"  ⚠ Groq API error: {e}")
        return {
            "relevant": False,
            "confidence": 0.0,
            "reason": f"API error: {str(e)[:100]}",
        }


# =============================================================================
# Progress Tracking
# =============================================================================
def load_classification_cache(cache_file: str) -> dict:
    """Load previously classified URLs from cache."""
    if not os.path.exists(cache_file):
        return {}
    with open(cache_file, "r") as f:
        return json.load(f)


def save_classification_cache(cache: dict, cache_file: str):
    """Save classification cache to disk."""
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Read extracted articles and classify relevance with LLM."""
    start_time = time.time()

    # Setup
    ensure_dir(FILTERED_ARTICLES_DIR)
    input_file = os.path.join(RAW_ARTICLES_DIR, "raw_articles.csv")
    output_file = os.path.join(FILTERED_ARTICLES_DIR, "filtered_articles.csv")
    cache_file = os.path.join(FILTERED_ARTICLES_DIR, "classification_cache.json")

    # Load articles
    articles = load_articles_from_csv(input_file)
    if not articles:
        logger.error(f"No articles found in {input_file}. Run content_extractor.py first.")
        return

    # Load classification cache
    cache = load_classification_cache(cache_file)

    logger.info("=" * 60)
    logger.info("RELEVANCE FILTERING — Starting")
    logger.info(f"  Total articles: {len(articles)}")
    logger.info(f"  Already classified (cache): {len(cache)}")
    logger.info(f"  Model: {GROQ_MODEL}")
    logger.info("=" * 60)

    relevant_articles = []
    classified_count = 0
    relevant_count = 0
    cached_count = 0

    for i, article in enumerate(articles):
        url = article["url"]
        title = article.get("title", "")
        content = article.get("content", "")

        # Check cache
        if url in cache:
            cached_count += 1
            if cache[url]["relevant"]:
                article["relevance_confidence"] = cache[url]["confidence"]
                article["relevance_reason"] = cache[url]["reason"]
                relevant_articles.append(article)
                relevant_count += 1
            continue

        logger.info(f"[{i+1}/{len(articles)}] Classifying: {title[:60]}...")

        # Classify with LLM
        result = classify_with_groq(title, content)

        # Cache the result
        cache[url] = result

        if result["relevant"]:
            article["relevance_confidence"] = result["confidence"]
            article["relevance_reason"] = result["reason"]
            relevant_articles.append(article)
            relevant_count += 1
            logger.info(f"  ✓ RELEVANT (confidence: {result['confidence']:.2f})")
        else:
            logger.info(f"  ✗ Not relevant — {result['reason'][:60]}")

        classified_count += 1

        # Save cache periodically
        if classified_count % 20 == 0:
            save_classification_cache(cache, cache_file)
            save_articles_to_csv(relevant_articles, output_file)
            logger.info(f"  💾 Saved checkpoint ({relevant_count} relevant so far)")

        # Small delay to respect Groq rate limits
        time.sleep(0.5)

    # Final save
    save_classification_cache(cache, cache_file)
    save_articles_to_csv(relevant_articles, output_file)

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("FILTERING COMPLETE")
    logger.info(f"  Newly classified: {classified_count}")
    logger.info(f"  From cache: {cached_count}")
    logger.info(f"  Total relevant articles: {relevant_count}")
    logger.info(f"  Total irrelevant: {len(articles) - relevant_count}")
    logger.info(f"  Relevance rate: {relevant_count/len(articles)*100:.1f}%")
    logger.info(f"  Time elapsed: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"  Output saved to: {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
