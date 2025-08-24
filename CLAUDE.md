# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is Patent-Radar (注目特許仕分けくん), a patent screening system that performs binary classification (hit/miss) of prior patents against invention ideas, specifically for liquid separation equipment patents. The system uses LLM-based semantic analysis to identify potential patent conflicts and provides ranked results with confidence scores and evidence.

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
pytest testing/unit/test_extraction.py

# Run specific test
pytest testing/unit/test_extraction.py::test_extract_claim_1
```

### Running the System
```python
# Example usage
from src.core.screener import PatentScreener

screener = PatentScreener()
results = screener.analyze(
    invention="testing/data/invention_sample.json",
    patents="testing/data/patents.jsonl"
)
```

### Performance Evaluation
```bash
# Generate performance evaluation report
python simple_report_generator.py

# Run comprehensive analysis
python analysis/unified.py

# Test data generation
python testing/data_generation/generate_test_data.py
```

## Core Architecture

### Processing Pipeline
The system follows a 5-stage pipeline:
1. **Normalization** (`src/core/normalize.py`): Load and normalize JSON patent data (encoding, units, case)
2. **Extraction** (`src/core/extract.py`): Extract claim 1 (independent) + abstract from each patent
3. **LLM Processing** (`src/llm/pipeline.py`): 
   - Summarize invention idea (once)
   - Summarize each patent (claims + abstract)
   - Binary classification with confidence score and hit reasons
4. **Ranking** (`src/ranking/sorter.py`): Sort by LLM_confidence descending
5. **Export** (`src/core/export.py`): Generate CSV and JSONL outputs

### Key Components

**Main Controller**: `src/core/screener.py` - Orchestrates the entire workflow, integrating all processing stages. Uses thread-safe design with comprehensive error handling.

**LLM Integration**: 
- `src/llm/client.py` - OpenAI API client with batch processing and error handling
- `src/llm/pipeline.py` - High-level LLM processing pipeline with concurrent execution
- `src/llm/prompts.py` - Structured prompt templates for different analysis stages

**Data Processing**:
- Input validation and normalization with security measures (ReDoS protection, path traversal prevention)
- JSON structure validation with fallback handling
- Claims parsing with independent/dependent classification

**Configuration System**: Uses `config.yaml` for all system parameters:
- LLM settings (model: gpt-4o-mini, temperature: 0.0, max_tokens: 320)
- Processing parameters (batch_size: 10, max_workers: 5)
- MVP configuration (no TopK, no retrieval scoring, no evidence verification)

## Input/Output Specifications

### Input
- **Invention idea**: JSON with optional keys: `title, problem, solution, effects, key_elements[], constraints`
- **Patent JSONs**: Required fields: `publication_number, title, assignee, pub_date, claims[], abstract`
  - Claims format: `{ no: <int>, text: <string>, is_independent: <bool> }`
  - Optional fields: `description[], cpc, ipc, legal_status, citations_forward, url_hint`

### Output
- **CSV**: `rank, pub_number, title, assignee, pub_date, decision, LLM_confidence, hit_reason_1, hit_src_1, hit_reason_2, hit_src_2, url_hint`
- **JSONL**: Detailed JSON per patent with decision, confidence, reasons, and flags

## Current System Status (MVP Configuration)

### Core Features Implemented
- Binary hit/miss classification with 81.5% accuracy achieved
- LLM_confidence-only ranking (no TopK, no retrieval score, no evidence verification)
- Batch processing with configurable workers (default: 5)
- Temperature set to 0.0 for deterministic outputs
- JSON response format enforced with 320 token limit

### Known Performance Characteristics
- **Accuracy**: 81.5% overall (meets 80% target)
- **HIT Detection**: 80% precision, 80% recall, 80% F1-score
- **Issue**: 20% False Negative rate (5 missed HITs out of 25) - addressed in v2 requirements
- **Processing**: ~22 seconds per patent, cost-efficient at 2 yen per 50 patents

### Testing Infrastructure
- Comprehensive test suite in `testing/` directory (moved from `tests/`)
- Gold standard dataset: 59 patents with expert annotations
- Performance evaluation framework with HTML/Markdown reporting
- Test data generation capabilities for various patent categories

## V2 Development Roadmap

### Priority Improvements (from 要件定義書_v2.md)
1. **HIT Miss Prevention**: Target False Negative reduction from 20% to <8%
   - Multi-stage judgment system (primary → re-evaluation → conservative)
   - Confidence threshold optimization (0.5 → 0.3 for HIT detection)
   - Dual judgment with ensemble voting

2. **Excel-to-JSONL Pipeline**: Planned `src/data_preparation/` module
   - Excel parser with data validation
   - Quality assurance and error reporting
   - Automated claims structuring

### Architecture Evolution
- Current MVP uses semantic-only approach (no keyword-heavy methods like BM25)
- Future expansion planned for TopK filtering, retrieval scoring, and evidence verification
- All processing designed to maintain stability (consistent sort order on re-runs)

## Development Workflow

### TDD + Code Review Process
Each module follows a two-phase development:
- **Phase A**: Test design → code-reviewer design analysis → test implementation → pytest (Red)
- **Phase B**: Implementation → pytest (Green) → code-reviewer analysis → refactoring → final pytest

### Quality Standards Applied
- **Security**: Input validation, DoS protection, path traversal prevention
- **Performance**: Caching, pre-compiled regex, memory efficiency
- **Maintainability**: Constants classes, function decomposition, comprehensive error handling

### Troubleshooting Notes
- **OpenAI API Tests**: Use proper mock with httpx.Request for error handling tests
- **Interface Integration**: PatentScreener expects individual methods, LLMPipeline provides batch processing
- **Encoding**: Unicode issues on Windows - use UTF-8 encoding explicitly

## Important Notes
- System outputs binary hit/miss decisions only (borderline cases marked for human review)
- All logging via loguru to `archive/logs/`
- Outputs saved to `archive/outputs/`
- Work logs maintained in `archive/history/`
- MVP focus: High recall for HIT detection to minimize opportunity loss from missed patents
- Stability requirement: Deterministic behavior with tiebreaker by pub_number for consistent ranking