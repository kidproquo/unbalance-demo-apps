# Performance Comparison: All Unbalance Levels vs Level 1 Only

**Date:** 2025-11-26
**Dataset Configurations:**
- **All Levels:** 0E, 1E, 2E, 3E, 4E (90% normal, 10% unbalanced distributed across 1E-4E)
- **Level 1 Only:** 0E, 1E (90% normal, 10% Level 1 only)

## Performance Summary Tables

### All Unbalance Levels (1E-4E Mixed)

**Total Windows Processed:** ~77,170

| Approach | 0E (Normal) | 1E    | 2E    | 3E    | 4E    | Overall Accuracy | Overall Recall | FP Rate |
|----------|-------------|-------|-------|-------|-------|------------------|----------------|---------|
| **CNN**  | 99.0%       | 89.8% | 93.2% | 98.9% | 100%  | 98.7%            | 95.5%          | 1.0%    |
| **FFT**  | 99.9%       | 90.1% | 98.9% | 90.4% | 99.9% | 99.4%            | 94.8%          | 0.06%   |
| **RFC**  | 99.9%       | 27.2% | 28.1% | 68.6% | 95.7% | 95.5%            | 55.1%          | 0.06%   |

### Level 1 Only

**Total Windows Processed:** ~3,090

| Approach | 0E (Normal) | 1E    | Overall Accuracy | Overall Recall | FP Rate |
|----------|-------------|-------|------------------|----------------|---------|
| **CNN**  | 99.0%       | 69.9% | 96.1%            | 69.9%          | 1.0%    |
| **FFT**  | 99.9%       | 79.0% | 97.8%            | 79.0%          | 0.07%   |
| **RFC**  | 99.9%       | 51.6% | 95.1%            | 51.6%          | 0.07%   |

## Detailed Breakdown by Unbalance Level

### Level 1 (1E) - Subtle Unbalance

| Dataset | CNN Accuracy | FFT Accuracy | RFC Accuracy | Winner |
|---------|--------------|--------------|--------------|--------|
| **All Levels (1936 samples)** | 89.8% | 90.1% | 27.2% | FFT ✅ |
| **Level 1 Only (310 samples)** | 69.9% | 79.0% | 51.6% | FFT ✅ |
| **Change** | -19.9% ❌ | -11.1% ❌ | +24.4% ✅ | |

**Observations:**
- **FFT** is most consistent: Best performer in both scenarios
- **CNN** dropped significantly: 89.8% → 69.9% (-19.9%)
- **RFC** actually improved: 27.2% → 51.6% (+24.4%) but still worst overall

### Level 2 (2E) - Moderate Unbalance

| Approach | Total Samples | TP | FN | Accuracy | Recall |
|----------|---------------|----|----|----------|--------|
| **CNN**  | 1848          | 1722 | 126 | 93.2%  | 93.2% |
| **FFT**  | 1848          | 1827 | 21  | 98.9%  | 98.9% |
| **RFC**  | 1848          | 519  | 1329| 28.1%  | 28.1% |

**Winner:** FFT (98.9% accuracy)

### Level 3 (3E) - Strong Unbalance

| Approach | Total Samples | TP | FN | Accuracy | Recall |
|----------|---------------|----|----|----------|--------|
| **CNN**  | 1940          | 1918 | 22  | 98.9%  | 98.9% |
| **FFT**  | 1848          | 1753 | 187 | 90.4%  | 90.4% |
| **RFC**  | 1940          | 1331 | 609 | 68.6%  | 68.6% |

**Winner:** CNN (98.9% accuracy)

### Level 4 (4E) - Severe Unbalance

| Approach | Total Samples | TP | FN | Accuracy | Recall |
|----------|---------------|----|----|----------|--------|
| **CNN**  | 1911          | 1911 | 0   | 100%   | 100%  |
| **FFT**  | 1911          | 1910 | 1   | 99.9%  | 99.9% |
| **RFC**  | 1911          | 1828 | 83  | 95.7%  | 95.7% |

**Winner:** CNN (100% accuracy - perfect detection)

## Key Findings

### 1. Dataset Composition Dramatically Affects Performance

**CNN Performance:**
- **All Levels:** 89.8% on 1E, 98.7% overall
- **1E Only:** 69.9% on 1E, 96.1% overall
- **Impact:** CNN performs 19.9% worse on Level 1 when it's isolated
- **Hypothesis:** CNN benefits from learning patterns across multiple unbalance severities

**FFT Performance:**
- **All Levels:** 90.1% on 1E, 99.4% overall
- **1E Only:** 79.0% on 1E, 97.8% overall
- **Impact:** Most consistent, only 11.1% drop
- **Strength:** Frequency domain features are robust regardless of dataset composition

**RFC Performance:**
- **All Levels:** 27.2% on 1E, 95.5% overall
- **1E Only:** 51.6% on 1E, 95.1% overall
- **Impact:** Improved 24.4% on Level 1 when isolated, but still worst performer
- **Paradox:** Better at Level 1 in isolation but terrible in mixed dataset

### 2. RFC Shows Severity-Dependent Performance

| Level | Severity | RFC Accuracy | Performance |
|-------|----------|--------------|-------------|
| 1E    | Subtle   | 27.2%        | Poor ❌      |
| 2E    | Moderate | 28.1%        | Poor ❌      |
| 3E    | Strong   | 68.6%        | Moderate ⚠️  |
| 4E    | Severe   | 95.7%        | Excellent ✅ |

**Pattern:** RFC accuracy increases linearly with unbalance severity
- Statistical features (std, kurtosis) are more discriminative for severe unbalances
- Subtle vibrations (1E, 2E) don't produce large enough statistical differences

### 3. CNN Excels at Severe Unbalances

**CNN's sweet spot:** Level 3 and Level 4
- **Level 3:** 98.9% accuracy (22 misses out of 1,940)
- **Level 4:** 100% accuracy (ZERO misses out of 1,911)

**CNN's weakness:** Level 1
- **All levels:** 89.8% (still decent)
- **Level 1 only:** 69.9% (struggles without context)

### 4. FFT is Most Balanced and Reliable

**Consistent high performance across all levels:**
- Level 1: 90.1% (all levels) / 79.0% (isolated)
- Level 2: 98.9%
- Level 3: 90.4%
- Level 4: 99.9%

**Trade-off:** FFT doesn't achieve 100% on any level but maintains 90%+ on most

### 5. False Positive Rates

| Approach | All Levels FP | Level 1 Only FP | FP Rate |
|----------|---------------|-----------------|---------|
| **CNN**  | 668/69,535    | 28/2,781        | ~1.0%   |
| **FFT**  | 42/69,540     | 2/2,782         | ~0.06%  |
| **RFC**  | 42/69,540     | 2/2,782         | ~0.06%  |

**Observation:** False positive rates remain consistent regardless of dataset composition
- CNN is consistently more aggressive (1% FP rate)
- FFT and RFC are very conservative (0.06% FP rate)

## Detection Count Analysis (All Levels)

From the performance reports:

### True Positives by Approach

| Approach | 1E TP | 2E TP | 3E TP | 4E TP | Total TP | Total Possible |
|----------|-------|-------|-------|-------|----------|----------------|
| **CNN**  | 1738  | 1722  | 1918  | 1911  | 7,289    | 7,635          |
| **FFT**  | 1745  | 1827  | 1753  | 1910  | 7,235    | 7,635          |
| **RFC**  | 527   | 519   | 1331  | 1828  | 4,205    | 7,635          |

**Total Missed Detections:**
- **CNN:** 346 misses (95.5% recall)
- **FFT:** 400 misses (94.8% recall)
- **RFC:** 3,430 misses (55.1% recall)

### False Negatives by Level

| Approach | 1E FN | 2E FN | 3E FN | 4E FN | Pattern |
|----------|-------|-------|-------|-------|---------|
| **CNN**  | 198   | 126   | 22    | 0     | Decreasing with severity ✅ |
| **FFT**  | 191   | 21    | 187   | 1     | Inconsistent ⚠️ |
| **RFC**  | 1409  | 1329  | 609   | 83    | Decreasing with severity ✅ |

**CNN and RFC:** Both miss fewer cases as severity increases (expected)
**FFT:** Anomalous spike in Level 3 misses (187) compared to Level 2 (21)

## Practical Implications

### For Production Deployment

**Scenario 1: Mixed severity environment (typical factory)**
- **Best choice:** FFT
- **Rationale:** 99.4% overall accuracy, 94.8% recall, only 0.06% FP rate
- **Trade-off:** May miss some Level 3 cases (90.4%)

**Scenario 2: Known Level 1 monitoring (early detection focus)**
- **Best choice:** FFT
- **Rationale:** 79% accuracy on Level 1 (both scenarios)
- **Alternative:** CNN if you can tolerate 1% false positive rate

**Scenario 3: Severe unbalance detection (safety critical)**
- **Best choice:** CNN
- **Rationale:** 100% on Level 4, 98.9% on Level 3
- **Benefit:** Zero misses on most dangerous conditions

**Scenario 4: Budget/resource constrained**
- **Best choice:** RFC
- **Rationale:** Smallest model, fastest inference
- **Caveat:** Only use if you care primarily about severe unbalances (3E, 4E)

### Ensemble Recommendations

**Voting system for all scenarios:**
1. **2-of-3 majority vote (CNN + FFT + RFC)**
   - Pros: High confidence detections
   - Cons: May miss subtle cases where only 1 detects

2. **Any-of-2 vote (FFT + CNN)**
   - Pros: Better recall, catches more cases
   - Cons: Higher false positive rate
   - **Recommended** for most applications

3. **Severity-based routing:**
   - Use RFC for 4E detection (95.7% accuracy, very fast)
   - Use FFT for 1E-3E detection (90%+ accuracy)
   - Use CNN as backup/validator

## Statistical Summary

### Overall Performance Ranking (All Levels)

**By Overall Accuracy:**
1. FFT: 99.4%
2. CNN: 98.7%
3. RFC: 95.5%

**By Overall Recall:**
1. CNN: 95.5%
2. FFT: 94.8%
3. RFC: 55.1%

**By False Positive Rate (lower is better):**
1. FFT: 0.06%
2. RFC: 0.06%
3. CNN: 1.0%

**By Level 1 Performance (All Levels dataset):**
1. FFT: 90.1%
2. CNN: 89.8%
3. RFC: 27.2%

### Processing Statistics

**Windows Processed:** 77,170
**Time Period:** ~24.4 hours (87,808 seconds)
**Processing Rate:** ~3,160 windows/hour (~0.88 windows/second)

**Distribution:**
- 0E (Normal): ~69,540 (90.1%)
- 1E-4E (Unbalanced): ~7,635 (9.9%)
  - 1E: ~1,936 (25.3% of unbalanced)
  - 2E: ~1,848 (24.2% of unbalanced)
  - 3E: ~1,940 (25.4% of unbalanced)
  - 4E: ~1,911 (25.0% of unbalanced)

## Conclusion

**Key Takeaway:** Model performance is highly dependent on whether unbalance levels are mixed or isolated.

- **FFT** is the most robust and production-ready across all scenarios
- **CNN** excels at severe unbalances but struggles with isolated Level 1
- **RFC** is severity-dependent and only viable for moderate-to-severe detection

**For research:** This demonstrates that model evaluation must consider real-world deployment scenarios. A model that performs well on mixed data may fail when deployed to monitor a single failure mode.
