# Level 1 Detection Analysis - With Only 1E Unbalance Data

**Date:** 2025-11-24
**Configuration:** Coordinator sending only 0E (normal) and 1E (Level 1 unbalance) data

## Configuration

- **Coordinator setting:** `--unbalance-levels 1E`
- **Distribution:** 90% 0E (normal), 10% 1E (unbalance)
- **Detection threshold:** 0.9 for all three approaches
- **Windows processed:** ~3,090 total (~2,780 0E, ~310 1E)

## Overall Performance Summary

| Approach | Total 1E | Detected (TP) | Missed (FN) | Accuracy | FP (0E) | FP Rate |
|----------|----------|---------------|-------------|----------|---------|---------|
| **FFT**  | 310      | 245           | 65          | **79.0%** | 2       | 0.07%   |
| **CNN**  | 309      | 216           | 93          | **69.9%** | 28      | 1.0%    |
| **RFC**  | 310      | 160           | 150         | **51.6%** | 2       | 0.07%   |

### Key Findings

1. **FFT performs best:** 79.0% accuracy (245/310 detected)
2. **CNN is second:** 69.9% accuracy (216/309 detected)
3. **RFC struggles most:** 51.6% accuracy (160/310 detected)

### False Positive Analysis

- **CNN:** 28/2,781 = 1.0% false positive rate (highest)
- **FFT:** 2/2,782 = 0.07% false positive rate (excellent)
- **RFC:** 2/2,782 = 0.07% false positive rate (excellent)

## Detection Agreement Analysis

From analyzing the actual detection logs:

### Total Detection Counts

- **FFT:** 243 detections (most aggressive)
- **CNN:** 215 detections
- **RFC:** 160 detections (most conservative)
- **Unique Level 1 windows detected:** 288

### Agreement Patterns

| Agreement Level | Count | Percentage | Interpretation |
|----------------|-------|------------|----------------|
| All 3 approaches | 80    | 27.8%      | Strong consensus |
| Exactly 2 approaches | 170   | 59.0%      | Moderate consensus |
| Only 1 approach | 38    | 13.2%      | No consensus |

### Two-Approach Agreement Breakdown

When exactly 2 approaches agree:
- **CNN + FFT:** 107 cases (62.9%) - **Most common pairing!**
- **FFT + RFC:** 48 cases (28.2%)
- **CNN + RFC:** 15 cases (8.8%)

### Single-Approach Unique Detections

- **RFC only:** 17 cases
- **CNN only:** 13 cases
- **FFT only:** 8 cases

## Comparison with Previous Analysis (All Unbalance Levels)

### Performance Changes

| Approach | Previous (Mixed Levels) | Current (1E Only) | Change |
|----------|------------------------|-------------------|--------|
| **FFT**  | 65.1%                  | 79.0%             | **+13.9%** ✅ |
| **CNN**  | 37.0%                  | 69.9%             | **+32.9%** ✅ |
| **RFC**  | 74.0%                  | 51.6%             | **-22.4%** ❌ |

### Agreement Pattern Changes

**Previous (all unbalance levels):**
- Most common 2-approach pairing: FFT + RFC (48 cases)

**Current (1E only):**
- Most common 2-approach pairing: CNN + FFT (107 cases)

This represents a **fundamental shift** in which approaches align with each other.

## Key Insights

### 1. Dramatic RFC Performance Decline

RFC dropped from **74% accuracy** (best) to **51.6%** (worst) when using only Level 1 data.

**Hypothesis:** RFC's statistical features (mean RPM, std deviation, kurtosis) are effective at distinguishing *between* different unbalance severity levels but struggle when all unbalanced data comes from a single level. The features may not be discriminative enough for the subtle variations *within* Level 1 alone.

### 2. CNN-FFT Alliance Emerges

CNN and FFT now agree on **107 cases** (62.9% of 2-approach agreements), replacing the previous FFT-RFC alliance.

**Interpretation:** When focused on Level 1 detection, CNN's convolutional filters and FFT's frequency domain analysis extract similar information. Both detect patterns in the vibration waveforms that indicate Level 1 unbalance.

### 3. FFT Proves Most Robust

FFT maintained high performance and actually **improved** from 65.1% to 79.0%.

**Why:** Frequency domain features are inherently robust for detecting subtle periodic anomalies. Level 1 unbalances create small but consistent frequency shifts that FFT reliably captures.

### 4. CNN Shows Significant Improvement

CNN jumped from 37% to 69.9% accuracy, the **largest improvement** of all approaches.

**Possible reasons:**
- Less noise from mixing different unbalance levels
- CNN can focus on learning Level 1-specific patterns
- Raw time-series patterns may be more consistent within a single level

**However:** CNN has the highest false positive rate (1.0%), suggesting it's more sensitive but less precise than FFT/RFC.

### 5. Level 1 Unbalances Are Genuinely Ambiguous

Only **27.8%** of windows achieve consensus across all 3 approaches. This is relatively low.

**Implication:** Level 1 (subtle) unbalances are inherently difficult to detect, with significant disagreement between different ML approaches about what constitutes unbalance at this level.

## Prediction Confidence Examples

### High Confidence (All 3 Approaches Detected)

| Window | CNN Score | FFT Score | RFC Score |
|--------|-----------|-----------|-----------|
| 4      | 1.000     | 0.998     | 0.967     |
| 5      | 1.000     | 1.000     | 1.000     |
| 8      | 1.000     | 1.000     | 1.000     |

These are "obvious" Level 1 cases with strong vibration signatures.

### Disputed Cases (CNN + FFT Agree, RFC Disagrees)

| Window | CNN Score | FFT Score | RFC Score |
|--------|-----------|-----------|-----------|
| 0      | 0.998     | 0.988     | N/A       |
| 1      | 1.000     | 1.000     | N/A       |
| 2      | 0.998     | 0.995     | N/A       |

These represent cases where waveform-based methods (CNN, FFT) detect patterns that statistical methods (RFC) miss.

### CNN-Only Detections

| Window | CNN Score | FFT Score | RFC Score |
|--------|-----------|-----------|-----------|
| 11     | 1.000     | N/A       | N/A       |
| 136    | 0.999     | N/A       | N/A       |
| 141    | 0.987     | N/A       | N/A       |

Even with high confidence scores (≥0.987), these are patterns only CNN's convolutional filters detect.

## Recommendations

### 1. For Production Deployment

**Use FFT as primary detector for Level 1 unbalances:**
- Best accuracy (79.0%)
- Lowest false positive rate (0.07%)
- Consistent performance

### 2. For Ensemble/Voting System

**Use CNN + FFT majority vote:**
- They agree 62.9% of the time when 2 approaches agree
- FFT provides precision, CNN provides sensitivity
- RFC adds little value for Level 1-only detection

### 3. For Threshold Tuning

Consider lowering the 0.9 threshold for:
- **RFC:** Current 51.6% accuracy suggests threshold is too high for its feature set
- **CNN:** Could reduce false positives (currently 1.0%) with slightly higher threshold

### 4. For Feature Engineering

**RFC needs improvement for single-level detection:**
- Current features (std, kurtosis) work better for multi-level scenarios
- Consider adding frequency-domain features to RFC
- Or use time-domain features that capture Level 1-specific patterns

## Conclusion

The focused Level 1 analysis reveals that **approach performance is highly dependent on the unbalance composition** in the data stream:

- **FFT** is the most robust across different scenarios
- **CNN** benefits from focused single-level data
- **RFC** requires variety in unbalance levels to perform well

This has important implications for model selection based on deployment scenarios:
- **Mixed severity environment:** Use RFC or ensemble
- **Single severity monitoring:** Use FFT or CNN
- **Unknown/variable environment:** Use FFT (safest bet)

---

## Detection Timing Analysis

### Average Time Between Consecutive Level 1 Detections

| Approach | Avg Time | Median Time | Detection Rate |
|----------|----------|-------------|----------------|
| **FFT**  | 12.5s    | 9.0s        | 287/hour (fastest) |
| **CNN**  | 13.6s    | 9.0s        | 264/hour |
| **RFC**  | 25.2s    | 9.0s        | 143/hour (slowest) |

### Key Timing Insights

1. **FFT is fastest:** Detects Level 1 unbalances every 12.5 seconds on average
2. **RFC is slowest:** Detects every 25.2 seconds (2.01x slower than FFT)
3. **All have similar median (9s):** Most detections happen in rapid succession
4. **RFC has longer gaps:** Maximum gap of 587s (9.8 min) vs FFT's 112s (1.9 min)

### Detection Frequency Distribution

**Pattern for all approaches:**
- ~50% of detections occur in rapid-fire mode (0-10s apart)
- ~35-38% occur in quick succession (10-30s apart)
- ~8-11% have moderate gaps (30-60s apart)
- <5% have long gaps (1+ minutes apart)

**RFC uniqueness:**
- Has 2 gaps exceeding 5 minutes (1.2% of detections)
- Shows more sporadic detection pattern
- When RFC detects, it detects in bursts, but bursts are further apart

### Interpretation

**Why RFC is slower:**
- With only 51.6% accuracy (160/310 detections), RFC misses many Level 1 cases
- This creates longer average gaps between successful detections
- RFC detects ~48% fewer Level 1 cases than FFT (162 vs 337)

**Why FFT and CNN are similar:**
- Both detect most Level 1 cases (FFT: 79%, CNN: 70%)
- Both show consistent detection patterns
- Both catch detections at similar intervals

**Median vs Average:**
- All approaches have 9s median but different averages
- This indicates occasional long gaps that skew the average upward
- RFC's 25.2s average vs 9s median shows it has many long gaps
