# Multi-Modal Damage Claim Verification System

A sophisticated system for verifying damage claims using images, claim conversations, user history, and evidence requirements. Powered by Google Gemini Vision API.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API Key
```bash
export GEMINI_API_KEY="your-api-key-from-google-ai-studio"
```

### 3. Run System
```bash
python code/main.py
```

Output files:
- `output.csv` - Predictions for all claims
- `log.txt` - Execution transcript
- `evaluation/sample_output.csv` - Sample predictions (if sample_claims.csv provided)

## Project Structure

```
.
├── code/                              # Source code
│   ├── main.py                       # Entry point (HackerRank evaluator)
│   ├── data_loader.py                # Data loading utilities
│   ├── claim_analyzer.py             # Vision API integration
│   ├── output_generator.py           # Output formatting and validation
│   ├── README.md                     # Detailed usage guide
│   └── evaluation/
│       ├── main.py                   # Offline evaluation module
│       └── evaluation_report.md      # Operational analysis
├── dataset/                          # Data files (provided)
│   ├── claims.csv                    # Test claims
│   ├── sample_claims.csv             # Sample claims with labels
│   ├── user_history.csv              # User context
│   ├── evidence_requirements.csv     # Evidence standards
│   └── images/                       # Image files
├── IMPLEMENTATION_GUIDE.md           # Complete implementation guide
├── requirements.txt                  # Python dependencies
├── output.csv                        # Generated predictions
└── log.txt                          # Execution log

```

## What It Does

For each damage claim, the system:

1. **Loads** claim text, images, and user context
2. **Analyzes** images using Gemini Vision API
3. **Decides** whether claim is:
   - `supported` - images clearly show claimed damage
   - `contradicted` - claimed damage not visible
   - `not_enough_information` - insufficient visual evidence
4. **Flags** risks (image quality, user history, manipulation)
5. **Outputs** structured CSV with justifications

## Key Features

- **Vision-based analysis**: Images are primary truth source
- **Multi-image support**: Analyze multiple images per claim
- **User history integration**: Risk context without overriding evidence
- **Adversarial protection**: Guards against prompt injection
- **Structured validation**: Pydantic-enforced output schema
- **Robust error handling**: Graceful degradation with fallbacks
- **Cost-effective**: ~$0.25 per 75 claims using Gemini Flash

## Taxonomy

**Object Types**: car, laptop, package

**Claim Status**: supported, contradicted, not_enough_information

**Issue Types**: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown

**Severity**: none, low, medium, high, unknown

See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for complete taxonomy and allowed values.

## Performance

- **Runtime**: 3-8 minutes for ~75 claims
- **Cost**: ~$0.25 estimated for full test set
- **Accuracy**: Evaluated against sample_claims.csv ground truth
- **Rate Limiting**: Conservative 2s between calls, exponential backoff

## Configuration

Edit `code/main.py` to customize:

```python
MODEL_ID = "gemini-1.5-flash"  # Model selection
SLEEP_BETWEEN_CALLS = 2.0      # Rate limiting
MAX_RETRIES = 3                # Retry attempts
```

## Input Format

**claims.csv**:
```
user_id,image_paths,user_claim,claim_object
user001,"images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg","There's a dent on my car door",car
```

**image_paths**: Semicolon-separated paths relative to dataset/

**image IDs**: Extracted from filename stems (e.g., `img_1` from `img_1.jpg`)

## Output Format

**output.csv** with 14 required columns:
- user_id
- image_paths
- user_claim
- claim_object
- evidence_standard_met (true/false)
- evidence_standard_met_reason
- risk_flags (semicolon-separated)
- issue_type
- object_part
- claim_status
- claim_status_justification
- supporting_image_ids
- valid_image (true/false)
- severity

## Troubleshooting

**Missing packages**: `pip install -r requirements.txt`

**API key not set**: `export GEMINI_API_KEY=your-key`

**Image not found**: Verify paths are relative to dataset/

**Rate limits**: Increase SLEEP_BETWEEN_CALLS in code/main.py

See [code/README.md](code/README.md) for detailed troubleshooting.

## Submission

For HackerRank, submit:

1. **code.zip** - Source code (exclude __pycache__, venv, dataset/)
2. **output.csv** - Predictions for dataset/claims.csv
3. **log.txt** - Execution transcript

See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for submission details.

## Documentation

- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - Complete guide
- [code/README.md](code/README.md) - Detailed usage guide
- [code/evaluation/evaluation_report.md](code/evaluation/evaluation_report.md) - Performance analysis
- [problem_statement.md](repo_reference/problem_statement.md) - Original requirements

## Getting Help

1. Check [code/README.md](code/README.md) for detailed help
2. Review log.txt for error messages
3. See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) FAQ section

## License

For HackerRank competition use only.

---

**Status**: Production Ready  
**Last Updated**: 2026-06-19  
**Version**: 1.0
