# Multi-Modal Damage Claim Verification System

A comprehensive system for verifying damage claims using vision language models, claim conversations, user history, and image evidence analysis.

## Overview

This system analyzes damage claims for three object types (car, laptop, package) and determines whether submitted images support, contradict, or provide insufficient evidence for the claim.

**Key Features**:
- Multi-image analysis using Google Gemini Vision API
- Structured output with guaranteed schema compliance
- User history risk assessment
- Evidence requirement validation
- Adversarial prompt injection prevention
- Comprehensive error handling and fallback logic
- Offline evaluation module with accuracy reporting

## Project Structure

```
code/
├── main.py                  # Main orchestration script (HackerRank entry point)
├── data_loader.py          # CSV and image data loading
├── claim_analyzer.py       # Vision API integration and analysis logic
├── output_generator.py     # Output CSV generation and validation
├── evaluation/
│   ├── main.py            # Offline accuracy evaluation (zero API cost)
│   └── evaluation_report.md # Detailed operational analysis
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.9+
- pip or conda
- Google Gemini API key

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd <repo-dir>
   ```

2. **Create virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set API key**:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   # Or on Windows:
   set GEMINI_API_KEY=your-api-key-here
   ```

## Usage

### Quick Start

From the project root directory:

```bash
python code/main.py
```

This will:
1. Load all data files
2. Process sample claims (if available) for evaluation
3. Process test claims from `dataset/claims.csv`
4. Generate `output.csv` with predictions
5. Create `log.txt` with execution transcript

### Running Options

```bash
# Process only sample claims
python code/main.py --sample-only

# Specify custom output path
python code/main.py --output predictions.csv

# Specify custom log file
python code/main.py --log execution.log

# Process from specific base directory
python code/main.py --base-path /path/to/repo
```

### Running Evaluation

After generating predictions:

```bash
python code/evaluation/main.py
```

This will:
1. Compare predictions against ground truth (sample_claims.csv)
2. Generate accuracy metrics
3. Create detailed diff report
4. Output to `evaluation/eval_report.json`

## Input Files

### dataset/claims.csv
Test claims to be classified. Columns:
- `user_id`: User identifier
- `image_paths`: Semicolon-separated image paths
- `user_claim`: Claim description/conversation
- `claim_object`: `car`, `laptop`, or `package`

### dataset/sample_claims.csv
Labeled sample claims with expected outputs (optional). Same schema as claims.csv plus:
- `claim_status`: Ground truth decision
- `issue_type`: Expected damage type
- `severity`: Expected severity

### dataset/user_history.csv
Historical context for users. Columns:
- `user_id`
- `past_claim_count`
- `accept_claim`
- `manual_review_claim`
- `rejected_claim`
- `last_90_days_claim_count`
- `history_flags`
- `history_summary`

### dataset/evidence_requirements.csv
Evidence standards by object and issue type. Columns:
- `requirement_id`
- `claim_object`: `car`, `laptop`, `package`, or `all`
- `applies_to`: Issue family (e.g., "dent or scratch")
- `minimum_image_evidence`: Description of required evidence

### dataset/images/
Image files referenced in claims, organized as:
```
images/
├── sample/
│   └── case_001/
│       ├── img_1.jpg
│       └── img_2.jpg
└── test/
    └── case_001/
        ├── img_1.jpg
        └── img_2.jpg
```

## Output Format

### output.csv
Main predictions file with columns (in order):
- `user_id`
- `image_paths`
- `user_claim`
- `claim_object`
- `evidence_standard_met`: Boolean - images sufficient?
- `evidence_standard_met_reason`: Why/why not
- `risk_flags`: Semicolon-separated flags
- `issue_type`: Damage type (dent, crack, etc.)
- `object_part`: Affected part
- `claim_status`: `supported`, `contradicted`, or `not_enough_information`
- `claim_status_justification`: Image-grounded explanation
- `supporting_image_ids`: Supporting image names
- `valid_image`: Boolean - suitable for automated review
- `severity`: `none`, `low`, `medium`, `high`, or `unknown`

### log.txt
Execution transcript with timestamps, errors, and statistics.

### evaluation/
- `sample_output.csv`: Predictions on sample claims
- `eval_report.json`: Accuracy metrics
- `eval_diff.csv`: Detailed per-row comparison

## Configuration

### API Model

Edit in `code/main.py`:
```python
MODEL_ID = "gemini-2.0-flash"  # Or "gemini-2.0-pro" for higher accuracy
```

### Rate Limiting

Adjust in `code/main.py`:
```python
SLEEP_BETWEEN_CALLS = 2.0    # Seconds between sequential claims
MAX_RETRIES = 3              # Retry attempts
RETRY_BASE_DELAY = 8.0       # Base seconds for exponential backoff
```

## Taxonomy

### Allowed Values

**claim_status**: `supported`, `contradicted`, `not_enough_information`

**issue_type**:
- `dent`, `scratch`, `crack`, `glass_shatter`, `broken_part`
- `missing_part`, `torn_packaging`, `crushed_packaging`
- `water_damage`, `stain`, `none`, `unknown`

**object_part** (by type):
- **Car**: `front_bumper`, `rear_bumper`, `door`, `hood`, `windshield`, `side_mirror`, `headlight`, `taillight`, `fender`, `quarter_panel`, `body`, `unknown`
- **Laptop**: `screen`, `keyboard`, `trackpad`, `hinge`, `lid`, `corner`, `port`, `base`, `body`, `unknown`
- **Package**: `box`, `package_corner`, `package_side`, `seal`, `label`, `contents`, `item`, `unknown`

**risk_flags** (semicolon-separated):
- `blurry_image`, `cropped_or_obstructed`, `low_light_or_glare`
- `wrong_angle`, `wrong_object`, `wrong_object_part`
- `damage_not_visible`, `claim_mismatch`, `possible_manipulation`
- `non_original_image`, `text_instruction_present`
- `user_history_risk`, `manual_review_required`, `none`

**severity**: `none`, `low`, `medium`, `high`, `unknown`

## Error Handling

### Common Issues

**"GOOGLE_API_KEY not set"**
- Solution: Set environment variable: `export GEMINI_API_KEY=<key>`

**"Image not found: images/test/case_001/img_1.jpg"**
- Check image paths in CSV are correct
- Verify images exist in dataset directory
- Check file extensions are correct

**Rate limit errors**
- Increase `SLEEP_BETWEEN_CALLS` in code/main.py
- Reduce `MAX_RETRIES` if API is consistently failing
- Wait and retry later

**"Processing failed" rows**
- Check log.txt for detailed error
- Manual review required for these cases
- Consider increasing `MAX_RETRIES`

## Performance

See `code/evaluation/evaluation_report.md` for detailed operational analysis.

**Typical Runtime**:
- Sample claims (15): 30-60 seconds
- Test claims (75): 3-8 minutes
- Evaluation: 1-2 seconds

**Estimated Cost**:
- Sample: ~$0.04
- Full test: ~$0.25

## Security

The system includes safeguards against adversarial attacks:

1. **Prompt Injection Detection**: Flags suspicious instruction-like text
2. **Image Primary Evidence**: Visual evidence cannot be overridden by text
3. **Risk Tracking**: Logs detected manipulation attempts
4. **Schema Validation**: Enforced through Pydantic models

See section "CRITICAL SYSTEM SECURITY INSTRUCTIONS" in code/main.py for details.

## Submission Preparation

### 1. Create code.zip

```bash
# From repo root
zip -r code.zip code/ \
  -x "code/__pycache__/*" \
     "code/**/__pycache__/*" \
     "code/evaluation/__pycache__/*" \
     "venv/*" \
     ".venv/*" \
     "*.pyc"
```

### 2. Generate Predictions

```bash
python code/main.py
# Output: output.csv
```

### 3. Collect Chat Transcript

```bash
# Already generated during execution
cat log.txt
```

### 4. Upload to HackerRank
- **File 1**: code.zip
- **File 2**: output.csv
- **File 3**: log.txt (chat transcript)

## Troubleshooting

### Debugging

Enable verbose logging in code/main.py:
```python
logging.basicConfig(level=logging.DEBUG)  # Instead of INFO
```

### Check Specific Claim

Edit code/main.py to add:
```python
if user_id == "specific_user_id":
    print(f"DEBUG: {analysis}")
```

### Validate CSV Format

```python
import pandas as pd
df = pd.read_csv("output.csv")
print(df.dtypes)
print(df.head())
```

## References

- [Google Gemini API Docs](https://ai.google.dev/)
- [Problem Statement](../problem_statement.md)
- [Evaluation Report](evaluation/evaluation_report.md)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## Support

For issues or questions:
1. Check log.txt for error details
2. Review code/evaluation/evaluation_report.md
3. Validate input data format
4. Verify API key is set correctly

---

**Last Updated**: 2026-06-19  
**Version**: 1.0  
**Status**: Production Ready
