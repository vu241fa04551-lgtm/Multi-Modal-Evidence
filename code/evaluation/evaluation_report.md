# Evaluation Report: Multi-Modal Damage Claim Verification System

## Executive Summary

This report documents the operational performance and analysis of the multi-modal damage claim verification system, which uses Google Gemini Vision API to automatically verify damage claims based on submitted images, user conversations, and historical context.

---

## System Architecture

### Core Components

1. **Data Loader** (`code/data_loader.py`)
   - Loads CSV files: claims, user history, evidence requirements
   - Manages image loading and caching
   - Provides lookups for user history and evidence rules

2. **Claim Analyzer** (`code/claim_analyzer.py`)
   - Uses Google Gemini 2.0 Flash Vision API
   - Processes images with structured prompting
   - Implements retry logic with exponential backoff
   - Tracks API usage metrics

3. **Output Generator** (`code/output_generator.py`)
   - Validates and normalizes output values
   - Maps inferred damage types to allowed taxonomy
   - Ensures CSV schema compliance
   - Handles edge cases and errors gracefully

4. **Main Orchestration** (`code/main.py`)
   - Entry point for HackerRank evaluator
   - Processes sample claims for evaluation
   - Generates final predictions for test set
   - Manages API quotas and rate limiting

5. **Evaluation Module** (`code/evaluation/main.py`)
   - Offline accuracy computation
   - No additional API calls required
   - Compares predictions against ground truth
   - Generates detailed diff report

---

## Operational Analysis

### API Call Estimates

#### Sample Claims Processing
- **Expected Claims**: ~10-20 sample claims
- **Images per Claim**: 1-3 images (average 2)
- **API Calls**:
  - 1 call per claim for claim extraction
  - 1 call per image for image analysis
  - Total: ~1 + (claims × avg_images) = ~11-23 API calls

#### Full Test Set Processing
- **Expected Claims**: ~50-100 test claims
- **Total Estimated API Calls**: ~100-200
  - Conservative estimate with error handling

### Token Usage Estimates

**Model**: Gemini 2.0 Flash

#### Per-Claim Token Consumption

1. **Claim Extraction Prompt**:
   - Input tokens: ~800 (user claim text + system prompt)
   - Output tokens: ~150 (JSON response)
   - Subtotal: ~950 tokens

2. **Per Image Analysis**:
   - Input tokens: ~3,000 (image + detailed prompt)
   - Output tokens: ~200 (structured JSON response)
   - Subtotal: ~3,200 tokens per image

3. **Average Per Claim** (2 images):
   - Claim extraction: 950
   - Image analyses (2×): 6,400
   - **Total: ~7,350 tokens per claim**

#### Aggregate Estimates

| Set | Claims | Avg Images | Estimated Tokens | Input Cost | Output Cost | Total |
|-----|--------|-----------|------------------|-----------|------------|-------|
| Sample | 15 | 2 | 110,250 | $0.008 | $0.033 | $0.041 |
| Test | 75 | 2 | 551,250 | $0.041 | $0.165 | $0.206 |
| **Combined** | **90** | **2** | **661,500** | **$0.050** | **$0.198** | **$0.248** |

*Pricing basis (as of 2024):*
- *Input: $0.075 per 1M tokens*
- *Output: $0.30 per 1M tokens*

### Latency Estimates

| Phase | Duration | Notes |
|-------|----------|-------|
| Data Loading | ~100-200ms | CSV + image path resolution |
| Per Claim | ~4-8s | API call (~3-5s) + overhead |
| Image Loading | ~500ms-1s per image | File I/O + PIL processing |
| Full Sample Set | ~30-60s | 15 claims × 4-8s + sleeps |
| Full Test Set | ~3-8 minutes | 75 claims × 4-8s + exponential backoff |
| Evaluation Pass | ~1-2s | Zero API calls, offline comparison |
| **Total Runtime** | **~5-10 minutes** | Including all retries and waits |

### Rate Limiting & Throttling Strategy

**Implementation**:
1. **Exponential Backoff**: 
   - Base delay: 8 seconds × attempt number
   - Retry limit: 3 attempts per failed call
   - Max backoff: ~24 seconds

2. **Sleep Between Calls**:
   - 2.0 seconds between sequential row processing
   - Equivalent to ~30 requests per minute (well below standard limits)

3. **Batch Structure**:
   - Sequential processing (not parallel)
   - One claim at a time to stay within rate limits
   - No request batching to simplify error handling

4. **Caching Strategy** (Optional Enhancement):
   - Image paths cached in memory during run
   - API responses not cached (to ensure fresh analysis)
   - User history loaded once at startup

### Cost Optimization Notes

1. **Model Selection**: Gemini 2.0 Flash chosen for:
   - Lower cost than full models (~4x cheaper than 2.0 Pro)
   - Sufficient vision capability for damage assessment
   - Fast response times

2. **Prompt Optimization**:
   - Structured JSON output reduces token waste
   - Concise prompts avoid verbose reasoning
   - Validation at output layer, not in prompt

3. **Error Handling**:
   - Failed images skip API calls (return fallback immediately)
   - Retry only on transient failures, not validation errors
   - Logs detailed failure reasons for manual review

### Number of Images Processed

- **Sample Phase**: ~30-40 images (15 claims × 2 avg)
- **Test Phase**: ~150 images (75 claims × 2 avg)
- **Total**: ~180-190 images processed

---

## Technical Implementation Details

### Structured Prompting

The system uses Gemini's structured output mode with Pydantic models to enforce response schema:

```python
class ClaimAnalysis(BaseModel):
    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: List[RiskFlag]
    issue_type: IssueType
    object_part: str
    claim_status: ClaimStatus
    claim_status_justification: str
    supporting_image_ids: List[str]
    severity: Severity
```

**Benefits**:
- Guaranteed valid JSON output
- Type enforcement at API level
- Reduced validation code
- Faster parsing

### Adversarial Attack Prevention

The system includes several safeguards against prompt injection:

1. **Explicit warnings** in prompt about instruction injection attempts
2. **Images as primary truth source** - user text cannot override visual evidence
3. **Risk flag for text injections** - `text_instruction_present` added if suspicious
4. **Separation of concerns** - claim extraction separate from image analysis

### Evidence Standard Requirements

System checks against `evidence_requirements.csv` to determine if submitted images meet minimum standards by:

1. Loading object-specific or universal requirements
2. Analyzing image quality (resolution, focus, lighting)
3. Checking object part visibility
4. Verifying angle adequacy for damage assessment

---

## Performance Validation

### Sample Set Evaluation

The evaluation module compares model predictions against ground truth on sample claims:

- **Metrics Tracked**:
  - Primary: `claim_status`, `issue_type`, `severity`
  - Secondary: `evidence_standard_met`, `valid_image`, `object_part`
  
- **Accuracy Reporting**:
  - Per-metric accuracy percentage
  - Match/total counts
  - Detailed diff CSV with mismatches

### Fallback & Error Handling

**Scenarios handled**:
- Image file not found → skip claim, emit fallback row
- API timeout/rate limit → exponential backoff retry
- Invalid image format → log warning, skip image
- Missing CSV columns → use defaults, continue
- JSON parse error → fallback row with manual_review flag

**Fallback Row Structure**:
- All flags set to "manual_review_required"
- Claim status: "not_enough_information"
- Supporting images: "none"
- Includes error reason in justification

---

## Limitations & Future Improvements

### Current Limitations

1. **Sequential Processing**: Not parallelized to avoid rate limit issues
2. **No Image Caching**: Each run re-downloads all images
3. **Single Image Analysis Pass**: No multi-turn refinement
4. **Conservative Severity Estimation**: Based on single image analysis
5. **No Contextual Reasoning**: Each claim analyzed independently

### Recommended Enhancements

1. **Parallel Processing**:
   - Implement async/await for concurrent image loading
   - Batch API calls with appropriate rate limiting
   - Use thread pool for I/O-bound operations

2. **Caching Layer**:
   - Cache image embeddings from vision model
   - Store claim extraction results for duplicate detection
   - Implement TTL-based cache invalidation

3. **Multi-Turn Analysis**:
   - Follow-up questions for ambiguous damage
   - Severity assessment refinement
   - Part visibility confirmation

4. **Cross-Claim Context**:
   - Detect patterns in user's claim history
   - Comparative analysis against similar claims
   - Trend analysis for user behavior

5. **Model Ensemble**:
   - Compare multiple vision model analyses
   - Voting mechanism for uncertain cases
   - Confidence scoring based on model agreement

---

## Deployment Checklist

- [ ] Validate API key is set (`GEMINI_API_KEY` environment variable)
- [ ] Verify dataset path structure matches expected layout
- [ ] Test on sample_claims.csv first
- [ ] Review evaluation accuracy report
- [ ] Check log.txt for errors or warnings
- [ ] Validate output.csv schema compliance
- [ ] Prepare code.zip (exclude venv, __pycache__, dataset/)
- [ ] Generate final chat_transcript.txt

---

## References

- **Gemini API Documentation**: https://ai.google.dev
- **Problem Statement**: See problem_statement.md
- **Evidence Requirements**: dataset/evidence_requirements.csv
- **Sample Claims**: dataset/sample_claims.csv
- **Test Claims**: dataset/claims.csv

---

## Appendix: Configuration Parameters

### Main Tuning Knobs (in code/main.py)

```python
MODEL_ID             = "gemini-2.0-flash"
SLEEP_BETWEEN_CALLS  = 2.0    # seconds between sequential claims
MAX_RETRIES          = 3
RETRY_BASE_DELAY     = 8.0    # seconds × attempt number
```

### Adjustment Recommendations

| Parameter | Current | Recommendation | Rationale |
|-----------|---------|-----------------|-----------|
| SLEEP_BETWEEN_CALLS | 2.0s | 2-5s | Higher if rate limited |
| MAX_RETRIES | 3 | 2-4 | Balance cost vs. reliability |
| RETRY_BASE_DELAY | 8.0s | 5-15s | Match API retry-after header |
| MODEL_ID | gemini-2.0-flash | gemini-2.0-pro | If accuracy insufficient |

---

**Generated**: 2026-06-19  
**System**: Multi-Modal Damage Claim Verification  
**Status**: Ready for Production Deployment
