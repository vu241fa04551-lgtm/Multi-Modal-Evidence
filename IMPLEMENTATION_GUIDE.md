# System Implementation Guide

## Overview

A production-ready multi-modal damage claim verification system that uses Google Gemini Vision API to analyze images and determine whether they support damage claims for three object types: cars, laptops, and packages.

## What Has Been Built

### Core Components

#### 1. **code/main.py** (Entry Point)
- HackerRank evaluator entry point
- Loads CSV data and images from dataset/
- Processes sample claims for evaluation
- Generates final predictions
- Handles API retries with exponential backoff
- Uses Pydantic for structured output validation

**Key Features**:
- Model: `gemini-1.5-flash` (fast, cost-effective)
- Structured JSON output with schema enforcement
- Automatic fallback rows for failed claims
- Rate limiting: 2 seconds between sequential requests
- 3 retries with exponential backoff (8s × attempt)

#### 2. **code/data_loader.py** (Data Management)
- CSV file loading (claims, user history, evidence requirements)
- Image loading and caching
- User history lookups for risk assessment
- Evidence requirement queries

#### 3. **code/claim_analyzer.py** (Vision Analysis)
- Gemini Vision API integration
- Multi-image analysis pipeline
- Claim extraction from user conversation
- Risk flag detection
- Token usage tracking

#### 4. **code/output_generator.py** (Output Formatting)
- Validates output against required schema
- Maps inferred damages to allowed taxonomy
- Normalizes risk flags and severity
- Handles edge cases and defaults

#### 5. **code/evaluation/main.py** (Offline Evaluation)
- Compares predictions against ground truth (zero API cost)
- Generates accuracy metrics
- Produces diff report for analysis

#### 6. **code/evaluation/evaluation_report.md** (Operational Analysis)
- API call estimates
- Token usage projections
- Cost analysis ($0.25 estimated for full test set)
- Latency estimates (3-8 minutes total)
- Rate limiting strategy documentation

### Supporting Files

- **code/README.md**: Complete user guide with installation, usage, troubleshooting
- **requirements.txt**: Updated dependencies (google-genai, pandas, pydantic, Pillow)

## How to Run

### 1. Set Up Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (get key from Google AI Studio)
export GEMINI_API_KEY="your-api-key-here"
```

### 2. Process Claims

```bash
# From project root
python code/main.py

# Or with custom output path
python code/main.py --output my_predictions.csv --log my_log.txt
```

### 3. Evaluate Results

```bash
# Run offline evaluation (compares against sample_claims.csv ground truth)
python code/evaluation/main.py
```

## Input Schema

**dataset/claims.csv** (test claims):
```
user_id | image_paths | user_claim | claim_object
--------|-------------|------------|-------------
user001 | images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg | "There's a dent on the door" | car
```

**dataset/sample_claims.csv** (optional, for evaluation):
Same as above, plus ground truth columns:
- `claim_status`
- `issue_type`
- `severity`

**dataset/user_history.csv**:
Historical context for each user (past_claim_count, rejection patterns, etc.)

**dataset/evidence_requirements.csv**:
Minimum image evidence standards by object and issue type

## Output Schema

**output.csv** (required submission file):
14 columns in this order:
1. `user_id`
2. `image_paths`
3. `user_claim`
4. `claim_object`
5. `evidence_standard_met` (true/false)
6. `evidence_standard_met_reason`
7. `risk_flags` (semicolon-separated)
8. `issue_type` (from allowed taxonomy)
9. `object_part` (object-specific)
10. `claim_status` (supported/contradicted/not_enough_information)
11. `claim_status_justification`
12. `supporting_image_ids` (image filenames)
13. `valid_image` (true/false)
14. `severity` (none/low/medium/high/unknown)

## Allowed Values (Taxonomy)

**claim_status**: `supported`, `contradicted`, `not_enough_information`

**issue_type**: `dent`, `scratch`, `crack`, `glass_shatter`, `broken_part`, `missing_part`, `torn_packaging`, `crushed_packaging`, `water_damage`, `stain`, `none`, `unknown`

**object_part** (by claim_object):
- **car**: front_bumper, rear_bumper, door, hood, windshield, side_mirror, headlight, taillight, fender, quarter_panel, body, unknown
- **laptop**: screen, keyboard, trackpad, hinge, lid, corner, port, base, body, unknown  
- **package**: box, package_corner, package_side, seal, label, contents, item, unknown

**risk_flags** (semicolon-separated, or "none"):
- Image quality: `blurry_image`, `cropped_or_obstructed`, `low_light_or_glare`, `wrong_angle`
- Object issues: `wrong_object`, `wrong_object_part`, `damage_not_visible`
- Content: `claim_mismatch`, `possible_manipulation`, `non_original_image`, `text_instruction_present`
- Context: `user_history_risk`, `manual_review_required`

**severity**: `none`, `low`, `medium`, `high`, `unknown`

## System Design

### Processing Pipeline

```
Load Data → Process Each Claim → Analyze Images → Generate Output → Save CSV
    ↓            ↓                    ↓                ↓
  CSV, Images   Extract Claim   Gemini Vision      Validate
  User History  User Claim      Check Evidence     Schema
  Evidence      Analyze Images  Compare Claim      Fallback
  Requirements  Quality Check   Decision
```

### Decision Logic

1. **Evidence Check**: Are images sufficient?
   - No images → not_enough_information
   - Critical quality issues → not_enough_information
   - Viable images → proceed

2. **Image Analysis**:
   - What damage is visible?
   - Which parts are visible?
   - Is image quality adequate?
   - Signs of manipulation?

3. **Claim Comparison**:
   - Clear match → supported
   - No visible damage despite claim → contradicted
   - Ambiguous → not_enough_information

4. **Risk Assessment**:
   - User history flags
   - Image quality issues
   - Possible manipulation
   - Manual review requirements

### Adversarial Protection

- **Images are primary truth**: Visual evidence cannot be overridden by user text
- **Prompt injection detection**: Flags suspicious instruction-like text in claims
- **Risk tracking**: Logs detected manipulation attempts
- **Schema enforcement**: Pydantic models guarantee valid output

## Cost & Performance

### Estimated Costs

| Metric | Sample (15 claims) | Test (75 claims) |
|--------|------------------|-----------------|
| API Calls | ~15-23 | ~75-150 |
| Input Tokens | ~110K | ~550K |
| Output Tokens | ~2K | ~10K |
| Total Cost | ~$0.041 | ~$0.206 |

*Pricing: Input $0.075/1M, Output $0.30/1M tokens (Gemini 2.0 Flash rates)*

### Performance

- Per claim: 4-8 seconds (API + overhead)
- Sample set: 30-60 seconds
- Full test set: 3-8 minutes
- Evaluation: 1-2 seconds

### Rate Limiting

- Sequential processing (no parallel API calls)
- 2-second sleep between claims
- 3 retries with exponential backoff
- Well below standard rate limits (~30 req/min)

## Submission Files

### 1. code.zip
```
code/
├── main.py
├── data_loader.py
├── claim_analyzer.py
├── output_generator.py
├── README.md
├── evaluation/
│   ├── main.py
│   └── evaluation_report.md
└── .../other supporting files
```

**Exclude**:
- `__pycache__/`, `*.pyc`
- `venv/`, `.venv/`
- `dataset/` (reference only, not needed)
- `repo_reference/` (reference only, not needed)

### 2. output.csv
Predictions for all rows in `dataset/claims.csv`

### 3. log.txt
Execution transcript with timestamps and statistics

## Error Handling

**Missing Images**: 
- Skip image, continue with others
- If no readable images, return "not_enough_information"

**API Failures**:
- Retry with exponential backoff
- Fallback after 3 attempts
- Log detailed error, mark for manual review

**Invalid Data**:
- Use sensible defaults
- Validate against schema
- Add risk flags as needed

**Output Validation**:
- Enforce allowed values
- Map inferred values to taxonomy
- Generate fallback rows for failures

## Customization

### Change Model
Edit code/main.py:
```python
MODEL_ID = "gemini-1.5-pro"  # Higher accuracy, higher cost
```

### Adjust Rate Limiting
Edit code/main.py:
```python
SLEEP_BETWEEN_CALLS = 5.0  # More conservative
MAX_RETRIES = 2             # Fewer retries
```

### Modify Prompts
Edit `build_prompt()` in code/main.py to adjust:
- System instructions
- Claim extraction logic
- Evidence assessment criteria
- Risk flag detection

## Troubleshooting

**"No module named google"**
- Run: `pip install -r requirements.txt`

**"GEMINI_API_KEY not set"**
- Run: `export GEMINI_API_KEY=your-key-here`

**Rate limit errors**
- Increase `SLEEP_BETWEEN_CALLS`
- Reduce `MAX_RETRIES`
- Run during off-peak hours

**Image not found**
- Verify image paths are correct relative to dataset/
- Check file extensions match

**Invalid output format**
- Check log.txt for errors
- Review output_generator.py for allowed values
- Ensure all required columns present

## Performance Optimization Notes

### Already Implemented
✓ Structured JSON output (vs. parsing text)
✓ Concise prompts (vs. verbose reasoning)
✓ Validation at output layer (vs. in prompt)
✓ Sequential processing (avoids rate limits)
✓ Error handling (graceful degradation)

### Optional Future Enhancements
- Parallel image loading (while keeping API sequential)
- Image embedding cache (avoid reanalysis)
- Multi-turn analysis for ambiguous cases
- Model ensemble (vote on uncertain results)
- Cross-claim pattern detection

## FAQ

**Q: Can I run this without an API key?**  
A: No, the system requires a valid Google Gemini API key.

**Q: What if an image is corrupted?**  
A: The system skips it and continues with other images. If no images are readable, returns "not_enough_information".

**Q: How long does processing take?**  
A: ~3-8 minutes for 75 test claims, depending on API responsiveness.

**Q: Can I parallelize API calls?**  
A: Not recommended - stays within rate limits by processing sequentially.

**Q: What about hallucinations?**  
A: Mitigated by structured output schema and explicit instructions to base decisions on visual evidence only.

## References

- [Google Gemini API](https://ai.google.dev/)
- [Pydantic Docs](https://docs.pydantic.dev/)
- [Problem Statement](problem_statement.md)
- [Evaluation Report](code/evaluation/evaluation_report.md)

---

## Summary

This system provides:
- **✓ Automated damage claim verification** using vision AI
- **✓ Multi-image analysis** for comprehensive assessment
- **✓ Risk-aware decision making** using user history
- **✓ Production-ready error handling** and fallbacks
- **✓ Comprehensive documentation** for deployment
- **✓ Cost-effective implementation** (~$0.25 for full test set)
- **✓ Fast execution** (3-8 minutes total runtime)

**Status**: Ready for HackerRank submission
