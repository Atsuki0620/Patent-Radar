# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a patent screening system called "注目特許仕分けくん" (Notable Patent Classifier) that performs binary classification (hit/miss) of prior patents against invention ideas, specifically for liquid separation equipment patents.

## Common Development Commands

### Setup and Dependencies
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r .venv/requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_extraction.py

# Run specific test
pytest tests/unit/test_extraction.py::test_extract_claim_1
```

### Running the System
```python
# Example usage (to be implemented)
from src.core import PatentScreener

screener = PatentScreener()
results = screener.analyze(
    invention="path/to/invention.json",
    patents="path/to/patents.jsonl"
)
```

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

### LLM Integration Pattern
- Uses OpenAI API (configurable via `config.yaml`)
- Temperature set to 0.0 for deterministic outputs
- JSON response format enforced
- Max tokens limited to 320 for efficiency
- Batch processing with configurable workers (default: 5)

## Input/Output Specifications

### Input
- **Invention idea**: Text or JSON with optional keys: `title, problem, solution, effects, key_elements[], constraints`
- **Patent JSONs**: Required fields: `publication_number, title, assignee, pub_date, claims[], abstract`
  - Claims format: `{ no: <int>, text: <string>, is_independent: <bool> }`
  - Optional fields: `description[], cpc, ipc, legal_status, citations_forward, url_hint`

### Output
- **CSV**: `rank, pub_number, title, assignee, pub_date, decision, LLM_confidence, hit_reason_1, hit_src_1, hit_reason_2, hit_src_2, url_hint`
- **JSONL**: Detailed JSON per patent with decision, confidence, reasons, and flags

## Configuration

Primary configuration in `config.yaml`:
- **LLM settings**: Model selection, temperature, response format
- **Processing**: Batch size, worker count
- **Output paths**: CSV and JSONL destinations
- **Debug options**: Logging levels, intermediate result saving

## MVP Configuration
```yaml
run:
  use_topk: false
  use_retrieval_score: false
  verify_quotes: false
llm:
  model: "gpt-4o-mini"
  temperature: 0.0
  response_format: json
  max_tokens: 320
ranking:
  method: "llm_only"  # final = LLM_confidence
```

## Development Process (Updated 2025-08-19)

### Improved TDD + Code Review Flow

Each module follows a two-phase development cycle:

**Phase A: Planning & Review-Driven Design**
1. Test case design
2. Code-reviewer design analysis for implementation guidance
3. Test improvement based on review feedback
4. Initial pytest execution (Red - tests should fail)

**Phase B: Implementation & Quality Assurance**
5. Implementation guided by Phase A insights
6. pytest execution (Green - tests should pass)
7. Code-reviewer implementation analysis for security/performance
8. Refactoring based on review recommendations
9. Final pytest execution (all tests pass)

### Code Review Integration
```bash
# Design review before implementation
claude-code "Review test cases in tests/unit/test_[module].py for design guidance"

# Implementation review after coding
claude-code "Review src/[module].py for security, performance, and quality improvements"
```

### Quality Standards Applied
- **Security**: Input validation, DoS protection, path traversal prevention
- **Performance**: Caching, pre-compiled regex, memory efficiency
- **Maintainability**: Constants classes, function decomposition, comprehensive error handling

## Important Notes
- MVP uses LLM_confidence only for scoring (no TopK, no retrieval score, no evidence verification)
- System outputs binary hit/miss decisions (borderline cases marked for human review)
- Stability requirement: Consistent sort order on re-runs (tiebreaker by pub_number)
- All processing is semantic-based, avoiding keyword-heavy approaches like BM25
- Logging via loguru to `archive/logs/`
- Outputs saved to `archive/outputs/`