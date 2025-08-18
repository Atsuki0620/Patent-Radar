# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a patent screening system called "注目特許仕分けくん" (Notable Patent Classifier) that performs binary classification (hit/miss) of prior patents against invention ideas, specifically for liquid separation equipment patents.

## Key Architecture

### Core Processing Pipeline
1. **Normalization**: Load and normalize JSON patent data (handle encoding, units, case)
2. **Extraction**: Extract claim 1 (independent) + abstract from each patent
3. **LLM Processing**: 
   - Summarize invention idea (once)
   - Summarize each patent (claims + abstract)
   - Binary classification with confidence score and hit reasons
4. **Ranking**: Sort by LLM_confidence descending
5. **Export**: Generate CSV and JSONL outputs

### Directory Structure
```
src/
├── core/        # normalize, extract, export functionality
├── llm/         # LLM client, prompts, pipeline
├── ranking/     # Sorting by LLM_confidence
├── retrieval/   # Future expansion (unused in MVP)
└── verify/      # Future expansion (unused in MVP)
```

## Input/Output Specifications

### Input
- **Invention idea**: Text or JSON with optional keys: `title, problem, solution, effects, key_elements[], constraints`
- **Patent JSONs**: Required fields: `publication_number, title, assignee, pub_date, claims[], abstract`
  - Claims format: `{ no: <int>, text: <string>, is_independent: <bool> }`

### Output
- **CSV**: `rank, pub_number, title, assignee, pub_date, decision, LLM_confidence, hit_reason_1, hit_src_1, hit_reason_2, hit_src_2, url_hint`
- **JSONL**: Detailed JSON per patent with decision, confidence, reasons, and flags

## MVP Configuration
```yaml
run:
  use_topk: false
  use_retrieval_score: false
  verify_quotes: false
llm:
  temperature: 0.0
  response_format: json
  max_tokens: 320
ranking:
  method: "llm_only"  # final = LLM_confidence
```

## Important Notes
- MVP uses LLM_confidence only for scoring (no TopK, no retrieval score, no evidence verification)
- System outputs binary hit/miss decisions (borderline cases marked for human review)
- Stability requirement: Consistent sort order on re-runs (tiebreaker by pub_number)
- All processing is semantic-based, avoiding keyword-heavy approaches like BM25