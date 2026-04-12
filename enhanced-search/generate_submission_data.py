#!/usr/bin/env python3

import json
import os
import time
import zipfile
import requests
from datetime import datetime

BASE = "http://localhost:5001"
OUT  = "submission_data"
os.makedirs(OUT, exist_ok=True)

def get(endpoint, params=None):
    r = requests.get(f"{BASE}{endpoint}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def save(name, data):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {name}")
    return path

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

section("0. Index Statistics")
stats = get("/api/stats")
save("00_index_stats.json", stats)


section("1. Five Queries × Three Methods (Q2)")

QUERIES = [
    {"id": "Q1", "text": "Claude productivity for coding"},
    {"id": "Q2", "text": "GitHub Copilot efficiency comparison"},
    {"id": "Q3", "text": "AI coding assistant performance"},
    {"id": "Q4", "text": "best practices AI pair programming"},
    {"id": "Q5", "text": "developer experience with Claude Code"},
]

METHODS = [
    ("/api/search/keyword",  "keyword_BM25"),
    ("/api/search/semantic", "semantic_embeddings"),
    ("/api/search/hybrid",   "hybrid_fusion"),
]

SIZE = 20   # top-20 results per method per query

all_query_results = []

for q in QUERIES:
    qdata = {"query_id": q["id"], "query_text": q["text"], "methods": {}}
    print(f"\n  {q['id']}: \"{q['text']}\"")

    for endpoint, method_key in METHODS:
        resp = get(endpoint, {"q": q["text"], "size": SIZE})
        qdata["methods"][method_key] = {
            "total_hits":     resp.get("total_hits"),
            "search_time_ms": resp.get("search_time_ms"),
            "top_score":      resp["results"][0]["score"] if resp["results"] else None,
            "results":        resp["results"],
        }
        print(f"    {method_key:25s}  hits={resp['total_hits']:>7}  "
              f"time={resp['search_time_ms']:>7.1f}ms  "
              f"top_score={resp['results'][0]['score']:.4f}")
        time.sleep(0.1)

    all_query_results.append(qdata)

save("01_five_queries_all_methods.json", all_query_results)

# Summary table (TSV — easy to paste into spreadsheet)
tsv_lines = ["Query_ID\tQuery_Text\tMethod\tTime_ms\tTotal_Hits\tTop_Score"]
for qd in all_query_results:
    for method_key, mdata in qd["methods"].items():
        tsv_lines.append(
            f"{qd['query_id']}\t{qd['query_text']}\t{method_key}\t"
            f"{mdata['search_time_ms']}\t{mdata['total_hits']}\t{mdata['top_score']}"
        )
with open(os.path.join(OUT, "01_query_summary_table.tsv"), "w") as f:
    f.write("\n".join(tsv_lines))
print("  Saved: 01_query_summary_table.tsv")

section("2. Precision–Recall Evaluation (Q1 Hybrid)")

# Manual relevance judgements for Q1 hybrid top-20
# Criterion: substantive discussion of AI coding tool productivity impact
RELEVANCE = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0]
N_R = sum(RELEVANCE)

pr_rows = []
cum_rel = 0
for i, rel in enumerate(RELEVANCE, 1):
    cum_rel += rel
    P = cum_rel / i
    R = cum_rel / N_R
    F = (2 * P * R / (P + R)) if (P + R) > 0 else 0
    M = (P + R) / 2
    pr_rows.append({
        "N": i,
        "relevant": rel,
        "cumulative_relevant": cum_rel,
        "precision_P": round(P, 4),
        "recall_R":    round(R, 4),
        "f1_measure":  round(F, 4),
        "arith_mean_M": round(M, 4),
    })

# attach actual document excerpts from earlier results
q1_hybrid = next(
    qd["methods"]["hybrid_fusion"]
    for qd in all_query_results if qd["query_id"] == "Q1"
)
for row, doc in zip(pr_rows, q1_hybrid["results"]):
    row["document_excerpt"] = doc["text"][:200]
    row["source"]           = doc["source"]
    row["sentiment_label"]  = doc["label"]
    row["score"]            = doc["score"]

pr_eval = {
    "query":              "Claude productivity for coding",
    "method":             "hybrid_fusion",
    "N_R_total_relevant": N_R,
    "binary_string":      "".join(str(r) for r in RELEVANCE),
    "relevance_criterion": (
        "Relevant (1) if the document provides substantive discussion of "
        "an AI coding tool's concrete impact on developer productivity "
        "(actual experiences, comparisons, or workflow descriptions). "
        "Non-relevant (0) for incidental mentions, philosophical statements, "
        "or rhetorical/sarcastic content."
    ),
    "non_relevant_positions": {
        "N=11": "Abstract/philosophical — no concrete productivity data",
        "N=15": "Lists multiple tools without discussing productivity impact",
        "N=20": "Rhetorical/sarcastic tweet, no substantive content",
    },
    "metrics_at_each_rank": pr_rows,
    "summary": {
        "precision_at_10":  pr_rows[9]["precision_P"],
        "recall_at_10":     pr_rows[9]["recall_R"],
        "f1_peak":          max(r["f1_measure"] for r in pr_rows),
        "f1_peak_at_N":     max(pr_rows, key=lambda r: r["f1_measure"])["N"],
        "perfect_precision_run": "N=1 to N=10 (P=1.0 throughout)",
    }
}
save("02_precision_recall_evaluation.json", pr_eval)

# Also save as TSV
tsv = ["N\tRelevant\tCumR\tPrecision\tRecall\tF1\tArithMean\tSource\tSentiment\tScore\tExcerpt"]
for r in pr_rows:
    tsv.append(
        f"{r['N']}\t{r['relevant']}\t{r['cumulative_relevant']}\t"
        f"{r['precision_P']}\t{r['recall_R']}\t{r['f1_measure']}\t{r['arith_mean_M']}\t"
        f"{r.get('source','')}\t{r.get('sentiment_label','')}\t"
        f"{r.get('score','')}\t{r.get('document_excerpt','').replace(chr(9),' ')}"
    )
with open(os.path.join(OUT, "02_precision_recall_table.tsv"), "w") as f:
    f.write("\n".join(tsv))
print("  Saved: 02_precision_recall_table.tsv")


section("3. Date-Filtered Queries (Timeline Search)")

date_experiments = [
    {"query": "Claude Code",          "date_from": "2025-01-01", "date_to": "2025-06-30"},
    {"query": "Claude Code",          "date_from": "2026-01-01", "date_to": "2026-03-31"},
    {"query": "GitHub Copilot",       "date_from": "2023-01-01", "date_to": "2023-12-31"},
    {"query": "GitHub Copilot",       "date_from": "2025-01-01", "date_to": "2026-03-31"},
    {"query": "cursor IDE",           "date_from": "2025-06-01", "date_to": "2026-03-31"},
    {"query": "AI coding assistant",  "date_from": "2024-01-01", "date_to": "2024-12-31"},
]

date_results = []
for exp in date_experiments:
    for endpoint, method_key in METHODS:
        resp = get(endpoint, {
            "q": exp["query"], "size": 10,
            "date_from": exp["date_from"], "date_to": exp["date_to"]
        })
        date_results.append({
            "query":        exp["query"],
            "date_from":    exp["date_from"],
            "date_to":      exp["date_to"],
            "method":       method_key,
            "total_hits":   resp["total_hits"],
            "time_ms":      resp["search_time_ms"],
            "top_results":  resp["results"][:5],
        })
        print(f"  \"{exp['query']}\" [{exp['date_from']}→{exp['date_to']}] "
              f"{method_key}: {resp['total_hits']} hits")
        time.sleep(0.1)

save("03_date_filtered_queries.json", date_results)


section("4. Timeline Aggregations")

timeline_queries = [
    "Claude Code", "GitHub Copilot", "cursor", "AI coding assistant",
    "windsurf", "chatgpt coding"
]
timeline_results = {}
for q in timeline_queries:
    resp = get("/api/timeline", {"q": q})
    timeline_results[q] = resp["timeline"]
    print(f"  \"{q}\": {len(resp['timeline'])} months of data")

save("04_timeline_aggregations.json", timeline_results)


section("5. Faceted Breakdowns (Multifaceted Search)")

facet_queries = [
    "Claude productivity for coding",
    "GitHub Copilot efficiency comparison",
    "AI coding assistant performance",
    "cursor IDE review",
    "windsurf vs cursor",
    "",   # global facets across entire index
]

facet_results = {}
for q in facet_queries:
    label = q if q else "__ALL_DOCUMENTS__"
    resp = get("/api/facets", {"q": q})
    facet_results[label] = resp["facets"]
    print(f"  \"{label}\": sources={len(resp['facets']['sources'])}, "
          f"labels={resp['facets']['labels']}")
    time.sleep(0.1)

save("05_faceted_breakdowns.json", facet_results)


section("6. Sentiment Trend Over Time")

sentiment_queries = [
    "Claude Code", "GitHub Copilot", "cursor", "AI coding",
    "windsurf", "gemini coding", ""
]
sentiment_results = {}
for q in sentiment_queries:
    label = q if q else "__ALL_DOCUMENTS__"
    resp = get("/api/sentiment_trend", {"q": q})
    sentiment_results[label] = resp["trend"]
    total = sum(r["total"] for r in resp["trend"])
    print(f"  \"{label}\": {len(resp['trend'])} months, {total:,} total docs")
    time.sleep(0.1)

save("06_sentiment_trends.json", sentiment_results)


section("7. AI Tool Sentiment Comparison")

tool_resp = get("/api/tool_comparison")
tool_comparison = {
    "description": "Sentiment distribution per major AI coding tool across entire index",
    "tools": tool_resp["tools"]
}
save("07_tool_comparison.json", tool_comparison)
for t in tool_resp["tools"]:
    print(f"  {t['tool']:12s}: total={t['total']:6d}  "
          f"pos={t['positive_pct']}%  neg={t['negative_pct']}%  neu={t['neutral_pct']}%")


# TOP ENTITIES PER QUERY
section("8. Top Entities and Aspects Per Query")

entity_queries = [
    "AI coding assistant", "productivity tools", "code generation",
    "cursor windsurf comparison", "Claude vs Copilot"
]
entity_results = {}
for q in entity_queries:
    resp = get("/api/top_entities", {"q": q})
    entity_results[q] = {
        "top_tools":         resp["top_tools"][:10],
        "top_aspects":       resp["top_aspects"][:10],
        "label_distribution": resp["label_distribution"],
    }
    print(f"  \"{q}\": top_tool={resp['top_tools'][0]['tool'] if resp['top_tools'] else 'n/a'}")
    time.sleep(0.1)

save("08_top_entities_per_query.json", entity_results)

section("9. Source-Filtered Queries (Platform Drill-Down)")

source_experiments = [
    {"query": "GitHub Copilot",        "source": "HackerNews"},
    {"query": "GitHub Copilot",        "source": "Reddit"},
    {"query": "Claude productivity",   "source": "Twitter"},
    {"query": "Claude productivity",   "source": "HackerNews"},
    {"query": "AI coding assistant",   "source": "Reddit"},
]

source_results = []
for exp in source_experiments:
    resp = get("/api/search/by_source", {
        "q": exp["query"], "source": exp["source"], "size": 10
    })
    source_results.append({
        "query":       exp["query"],
        "source":      exp["source"],
        "total_hits":  resp["total_hits"],
        "time_ms":     resp["search_time_ms"],
        "top_results": resp["results"][:5],
    })
    print(f"  \"{exp['query']}\" on {exp['source']}: {resp['total_hits']} hits")
    time.sleep(0.1)

save("09_source_filtered_queries.json", source_results)


section("10. Entity Search (Named AI Tool Drill-Down)")

entities = ["claude", "copilot", "cursor", "gemini", "chatgpt", "windsurf"]
entity_search_results = {}
for entity in entities:
    resp = get("/api/entity_search", {"entity": entity, "size": 10})
    entity_search_results[entity] = {
        "total_hits": resp["total_hits"],
        "time_ms":    resp["search_time_ms"],
        "top_results": resp["results"][:5],
    }
    print(f"  entity='{entity}': {resp['total_hits']:,} docs")
    time.sleep(0.1)

save("10_entity_search_results.json", entity_search_results)


section("11. Period-Based Queries (by_period)")

period_experiments = [
    {"query": "Claude Code",     "year_month": "2025-01"},
    {"query": "Claude Code",     "year_month": "2026-02"},
    {"query": "cursor",          "year_month": "2025-11"},
    {"query": "GitHub Copilot",  "year_month": "2024-06"},
]

period_results = []
for exp in period_experiments:
    resp = get("/api/search/by_period", {
        "q": exp["query"], "year_month": exp["year_month"], "size": 10
    })
    period_results.append({
        "query":       exp["query"],
        "year_month":  exp["year_month"],
        "total_hits":  resp["total_hits"],
        "time_ms":     resp["search_time_ms"],
        "top_results": resp["results"][:5],
    })
    print(f"  \"{exp['query']}\" in {exp['year_month']}: {resp['total_hits']} hits")
    time.sleep(0.1)

save("11_period_based_queries.json", period_results)


section("12. Master Summary")

summary = {
    "generated_at": datetime.now().isoformat(),
    "system": {
        "elasticsearch_index": "ai_coding_search",
        "total_documents": stats["total_documents"],
        "embedding_model":  stats["embedding_model"],
        "embedding_dims":   stats["embedding_dimensions"],
        "api_base":         BASE,
    },
    "files": {
        "00_index_stats.json":            "Elasticsearch index metadata",
        "01_five_queries_all_methods.json":"Top-20 results for all 5 queries × 3 methods",
        "01_query_summary_table.tsv":     "TSV summary of query performance metrics",
        "02_precision_recall_evaluation.json": "P/R/F1/M evaluation for Q1 hybrid (top-20)",
        "02_precision_recall_table.tsv":  "TSV of P/R metrics at each rank",
        "03_date_filtered_queries.json":  "Results with date range filters (Timeline innovation)",
        "04_timeline_aggregations.json":  "Month-by-month document counts per query",
        "05_faceted_breakdowns.json":     "Source/type/label facet aggregations per query",
        "06_sentiment_trends.json":       "Monthly sentiment breakdowns per query",
        "07_tool_comparison.json":        "Cross-tool sentiment distributions (5 tools)",
        "08_top_entities_per_query.json": "Top AI tools and aspects per query",
        "09_source_filtered_queries.json":"Platform drill-down results (Multifaceted innovation)",
        "10_entity_search_results.json":  "Documents per named AI tool entity",
        "11_period_based_queries.json":   "Results filtered to specific year-month periods",
        "12_master_summary.json":         "This file — overview and index of all outputs",
    },
    "query_performance_summary": [
        {
            "query_id": qd["query_id"],
            "query_text": qd["query_text"],
            "keyword_time_ms":  qd["methods"]["keyword_BM25"]["search_time_ms"],
            "semantic_time_ms": qd["methods"]["semantic_embeddings"]["search_time_ms"],
            "hybrid_time_ms":   qd["methods"]["hybrid_fusion"]["search_time_ms"],
            "keyword_hits":     qd["methods"]["keyword_BM25"]["total_hits"],
            "keyword_top_score":  qd["methods"]["keyword_BM25"]["top_score"],
            "hybrid_top_score":   qd["methods"]["hybrid_fusion"]["top_score"],
        }
        for qd in all_query_results
    ],
    "precision_recall_summary": pr_eval["summary"],
    "tool_sentiment_summary": [
        {k: v for k, v in t.items() if k in ("tool","total","positive_pct","negative_pct","neutral_pct")}
        for t in tool_resp["tools"]
    ],
}
save("12_master_summary.json", summary)


section("Packaging into submission_data.zip")

zip_path = "submission_data.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fname in sorted(os.listdir(OUT)):
        zf.write(os.path.join(OUT, fname), arcname=os.path.join("submission_data", fname))

size_mb = os.path.getsize(zip_path) / 1_048_576
print(f"\n  Created: {zip_path}  ({size_mb:.1f} MB)")
print(f"  Files included: {len(os.listdir(OUT))}")
print("\nDone.")
