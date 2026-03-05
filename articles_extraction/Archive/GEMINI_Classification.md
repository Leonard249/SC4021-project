# GEMINI_Classification.md

## Situation
- You are a helpful AI assistant. The user has scraped ~1,800 articles related to Vibe-Coding / AI-Assisted Coding. 
- Not all articles may be truly relevant — some might be tangentially related or entirely off-topic.
- The user wants to classify every article along two dimensions:
  1. **Relevance**: Is this article about Vibe-Coding or AI-Assisted Coding? 
    - Details: There are many terminologies related to Vibe-Coding / AI-Assisted Coding. For example: Agent Engineering, AI Pair Programming, AI Code Generation, etc. In this case, I refer to AI-assisted coding in its broadest sense. Any article that discusses the usage of AI tools to assist in coding is relevant. I make no restriction on the level of "assistance" an AI tool provides – it could fully build an app, only do autocomplete, only do partial development, whatever. 
  2. **Opinion**: Does this article express an opinion/sentiment about Vibe-Coding or AI-Assisted Coding?
    - Details: Any article that take some kind of stance, either positive or negative about AI-Assisted coding is considered to express an opinion. The question is not whether the article is good or bad, but whether it takes a stance on the topic. Even a weak stance is enough to count as an opinion. 

## Your Task
You must classify articles **one at a time** using the MCP tools from the `article-classifier` server. Your progress is saved automatically — you can stop and resume at any time without repeating work.

## Tools Available

| Tool | Purpose |
|------|---------|
| `get_next_article()` | Returns the next unclassified article (title, URL, text capped at 2,000 words). Returns `"ALL_DONE"` when finished. |
| `submit_classification(article_id, is_relevant, has_opinion)` | Record your classification for the article. |
| `skip_article(article_id, reason)` | Skip an unreadable/garbage article. |
| `get_classification_stats()` | Check progress and distribution. |
| `get_classification_results(filter_relevant, filter_opinion)` | Retrieve past classifications. |

## Classification Criteria

### `is_relevant` — Is this article about Vibe-Coding or AI-Assisted Coding?

| Value | Use when... |
|-------|-------------|
| `"yes"` | The article is **primarily** about vibe-coding, AI-assisted coding, AI code generation, AI pair programming, tools like Copilot/Cursor/Cody/Tabnine, or the broader impact of LLMs on software development. |
| `"partially"` | The article **mentions** vibe-coding or AI coding tools but is primarily about something else (e.g., a general AI article that briefly discusses code generation). |
| `"no"` | The article has **nothing to do** with vibe-coding or AI-assisted coding. |

### `has_opinion` — Does the article express a sentiment or opinion?

| Value | Use when... |
|-------|-------------|
| `"yes"` | The article expresses a **clear opinion** — positive, negative, or mixed — about AI-assisted coding. Examples: "Copilot makes me 10x productive", "vibe coding is dangerous", "AI coding has pros and cons". |
| `"no"` | The article is **neutral**: a tutorial, documentation, changelog, benchmark report, or purely factual coverage with no editorial stance. |

### When to Skip
Use `skip_article` only for articles where the text is:
- Empty or near-empty (just references, links, or boilerplate)
- Completely garbled or unreadable
- Obviously not an article (e.g., a JSON dump, error page)

**Do NOT skip articles just because they are irrelevant** — classify them as `is_relevant: "no"` instead.

## Workflow

Repeat the following loop:

1. Call `get_next_article()`
2. Read the title and article text
3. Decide your classification:
   - Is it about vibe-coding / AI-assisted coding? → `is_relevant`
   - Does it have an opinion? → `has_opinion`
4. Call `submit_classification(article_id, is_relevant, has_opinion)`
5. Go to step 1

**Periodically** (every ~50 articles), call `get_classification_stats()` to report progress.

**Stop** when `get_next_article()` returns `"ALL_DONE"`.

## Important Notes
- **Do NOT overthink**. Read the title and skim the text. If the answer is obvious from the title, just classify it. Don't spend time analyzing every paragraph. Speed is important — there are ~1,800 articles to process.
- **Be consistent**. Apply the criteria uniformly. When in doubt about relevance, lean toward `"partially"`. When in doubt about opinion, lean toward `"no"`.
- **Progress is saved automatically**. You can stop at any time and resume later. The system will pick up right where you left off.
- **Do not re-classify**. Once you submit, the article is done. Move on.
