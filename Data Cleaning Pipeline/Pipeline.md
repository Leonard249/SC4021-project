# Purpose

This pipeline filters raw scraped data into a cleaner dataset for later sentiment analysis on **vibe-coding / AI-assisted coding**.

# Problem

The raw dataset is noisy. Many items:

- are only loosely related to vibe-coding
- are advertisements or product announcements
- are fragmentary and not meaningful on their own
- do not clearly express a sentiment toward AI-assisted coding as a practice

If these items are sent directly into the downstream classification stage, the labeling quality becomes poor.

# Solution

Use a controller script to process the dataset **one item at a time**. For each item, the controller sends a **fresh stateless request to an OpenAI-compatible endpoint** so context does not accumulate across items.

Each item is classified into exactly one of:

- `Irrelevant`
- `Neutral`
- `Positive`
- `Negative`

# High-Level Pipeline

```text
raw_data.json
  -> controller picks one item
  -> stateless LLM request classifies that one item
  -> controller saves result
  -> repeat
```

The current runner for this is `run_headless_pipeline.py`.

# Why Headless

The model endpoint is no longer responsible for running its own loop or calling tools. Instead:

- the pipeline script owns item selection
- the pipeline script owns persistence
- the model only reads one item and returns one structured classification

This keeps each classification independent and avoids long-session context growth.

# State Files

- `raw_data.json`
  The full scraped dataset.
- `index.json`
  Remaining unprocessed IDs grouped by source. This shrinks over time.
- `processed_item.json`
  IDs that have already been processed by the controller.
- `selected_data.json`
  Accepted items grouped into the four target buckets.
- `failed_item.json`
  Items where the model returned invalid output or the run failed after retries.
- `GEMINI.md`
  The classification rubric loaded as the system instruction for every stateless request.

# Selection Logic

The controller uses source-balancing logic similar to the earlier tool-based design:

- look at how many already-selected items come from each source
- compute which sources are underrepresented
- randomly pick the next item from one of the most underrepresented sources

This helps avoid the final dataset being dominated by a single source.

# Classification Logic

For each selected item:

1. The controller builds a prompt containing the item data.
2. The runner loads `GEMINI.md` as the system instruction.
3. The model returns raw JSON containing:
   - `classification`
   - `confidence`
   - `reasoning`
   - `evidence`
4. The controller validates the JSON.
5. If the bucket is not full, the item is saved into `selected_data.json`.
6. If the bucket is already full, the item is discarded but still marked as processed.

# Default Bucket Targets

The current default targets in the runner are:

- `Irrelevant`: 300
- `Neutral`: 300
- `Positive`: 300
- `Negative`: 300

Total target size: `1200`

# Notes

- The model does not call tools in this design.
- Each item gets a fresh stateless request.
- `Comments` are not sent to the model in the current pipeline.
- Older legacy `Opinionated` entries may still exist in `selected_data.json`, but the active pipeline now uses only the four labels above.
