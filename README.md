# Patent-Radar (注目特許仕分けくん)

A patent screening system that performs binary classification (hit/miss) of prior patents against invention ideas, specifically designed for liquid separation equipment patents.

## Overview

Patent-Radar analyzes prior art patents and identifies potential conflicts with new invention ideas using LLM-based semantic analysis. The system outputs a ranked list of patents with hit/miss decisions and brief reasoning.

## Features

- **Binary Classification**: Automated hit/miss determination for each patent
- **Semantic Analysis**: LLM-based understanding of patent claims and abstracts
- **Confidence Scoring**: Each decision includes a confidence score (0-1)
- **Evidence Extraction**: Brief quotations with sources for hit decisions
- **Batch Processing**: Efficient processing of multiple patents
- **Flexible Output**: Both CSV and JSONL formats supported

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Atsuki0620/Patent-Radar.git
cd Patent-Radar
```

2. Create a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Configuration

Edit `config.yaml` to customize:
- LLM settings (model, temperature, max tokens)
- Processing parameters (batch size, workers)
- Output paths and formats
- Debug options

## Usage

```python
# Example usage (to be implemented)
from src.core import PatentScreener

screener = PatentScreener()
results = screener.analyze(
    invention="path/to/invention.json",
    patents="path/to/patents.jsonl"
)
```

## Input Format

### Invention Idea
```json
{
  "title": "Novel Liquid Separation System",
  "problem": "...",
  "solution": "...",
  "effects": "...",
  "key_elements": ["element1", "element2"]
}
```

### Patent JSON
```json
{
  "publication_number": "JP2025-xxxxxxA",
  "title": "Patent Title",
  "assignee": "Company Name",
  "pub_date": "2025-01-01",
  "claims": [
    {"no": 1, "text": "...", "is_independent": true}
  ],
  "abstract": "..."
}
```

## Output Format

### CSV
```csv
rank,pub_number,title,assignee,pub_date,decision,LLM_confidence,hit_reason_1,hit_src_1,url_hint
```

### JSONL
```json
{
  "pub_number": "JP2025-xxxxxxA",
  "decision": "hit",
  "LLM_confidence": 0.82,
  "reasons": [
    {"quote": "...", "source": {"field": "claim", "locator": "claim 1"}}
  ]
}
```

## Project Structure

```
Patent-Radar/
├── src/
│   ├── core/        # Core processing modules
│   ├── llm/         # LLM integration
│   ├── ranking/     # Ranking logic
│   ├── retrieval/   # Future: semantic retrieval
│   └── verify/      # Future: evidence verification
├── tests/           # Test suite
├── archive/         # Logs and outputs
└── docs/           # Documentation
```

## License

MIT License

## Author

Atsuki0620