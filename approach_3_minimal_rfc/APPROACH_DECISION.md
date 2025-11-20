# Approach 3 Decision: Minimal Features vs. TSFRESH

## Decision Summary

**CHOSEN**: Minimal Features Random Forest Classifier (7 features)
**REJECTED**: TSFRESH Random Forest Classifier (748 features)

## Background

The original Notebook 4 (TSFRESH RFC) presents two variants for Approach 3:

1. **TSFRESH Variant**: Uses 748 automatically-extracted time series features
2. **Minimal Features Variant**: Uses only 7 hand-crafted statistical features

## Critical Issues with TSFRESH Variant

### 1. Missing Pre-trained Model ❌
- No `tsfresh_rfc.joblib` file exists in `models/reference/`
- Would require training from scratch

### 2. Unknown Feature Extraction Process ❌
- The notebook uses pre-computed TSFRESH features stored in `.npy` files
- **No code or configuration found** for how these features were generated
- Cannot process new or real-time data without knowing:
  - Which 748 TSFRESH features were selected
  - TSFRESH configuration parameters
  - Feature extraction settings

### 3. Performance Concerns ❌
- TSFRESH extracts hundreds of statistical features per window
- Feature extraction can be **extremely slow** (seconds per window)
- May not achieve real-time performance for 1-second windows
- Requires heavy `tsfresh` library with many dependencies

### 4. Complexity vs. Benefit ⚠️
- Accuracy: 94.2% (mean across datasets)
- Significantly lower than:
  - Approach 1 (CNN): **99.6%**
  - Approach 2 (FFT FCN): **99.7%**
- Not worth the complexity given lower accuracy

## Advantages of Minimal Features Variant

### 1. Complete Self-Contained Implementation ✅
- Model can be trained from scratch using only the data ZIP
- All 7 features are simple to compute:
  ```python
  features = [
      np.mean(rpm_data),              # RPM mean
      np.std(sensor1_data),           # Sensor 1 std
      np.std(sensor2_data),           # Sensor 2 std
      np.std(sensor3_data),           # Sensor 3 std
      scipy.stats.kurtosis(sensor1),  # Sensor 1 kurtosis
      scipy.stats.kurtosis(sensor2),  # Sensor 2 kurtosis
      scipy.stats.kurtosis(sensor3)   # Sensor 3 kurtosis
  ]
  ```

### 2. Fast Feature Extraction ✅
- Only 7 features to compute (vs. 748)
- Feature extraction takes **milliseconds** per window
- Suitable for real-time deployment
- No heavy TSFRESH dependency

### 3. Interpretability ✅
- Features have clear physical meaning:
  - RPM: Rotation speed
  - Std: Vibration amplitude
  - Kurtosis: Distribution "tailedness" (spikiness of vibration)
- Can understand why model makes predictions
- Feature importance is meaningful

### 4. Lightweight Dependencies ✅
- Only requires: `numpy`, `pandas`, `scipy`, `scikit-learn`
- No TensorFlow (unlike Approaches 1 & 2)
- No TSFRESH (heavyweight library)
- Smaller memory footprint

### 5. Proven Performance ✅
- Per-dataset accuracy on evaluation set:
  - 0E (no unbalance): **93.6%**
  - 1E (level 1): 45.0% ⚠️ (poor)
  - 2E (level 2): 72.3%
  - 3E (level 3): 99.4%
  - 4E (level 4): **100.0%**
  - **Mean: 82.2%**

## Trade-offs Analysis

| Criterion | Minimal Features | TSFRESH | Winner |
|-----------|------------------|---------|--------|
| **Feasibility** | Can implement now | Missing model & config | **Minimal** |
| **Accuracy** | 82.2% mean | 94.2% mean | TSFRESH |
| **Real-time Speed** | Milliseconds | Seconds (estimated) | **Minimal** |
| **Interpretability** | High (7 features) | Low (748 features) | **Minimal** |
| **Dependencies** | Lightweight | Heavy (TSFRESH) | **Minimal** |
| **Complexity** | Simple | Complex | **Minimal** |
| **Training Time** | Minutes | Minutes-Hours | **Minimal** |

## Accuracy Comparison Across All Approaches

| Approach | Type | Features | Accuracy | Speed |
|----------|------|----------|----------|-------|
| **1: CNN** | Deep Learning | Raw signal (4096) | **99.6%** | Fast |
| **2: FFT FCN** | Deep Learning | FFT spectrum (2048) | **99.7%** | Fast |
| **3a: TSFRESH RFC** | Classical ML | TSFRESH (748) | 94.2% | Slow ⚠️ |
| **3b: Minimal RFC** ✅ | Classical ML | Stats (7) | 82.2% | **Very Fast** |

## Known Limitations

### 1. Lower Accuracy
- 82.2% mean is significantly lower than Approaches 1 & 2 (99%+)
- Particularly poor on Level 1 unbalance (45%)
- May not meet production requirements for critical applications

### 2. Performance by Unbalance Level
The model struggles with low unbalance levels:
- **Strong unbalances** (3E, 4E): Excellent (99-100%)
- **Medium unbalances** (2E): Fair (72%)
- **Weak unbalances** (1E): Poor (45%)
- **No unbalance** (0E): Good (94%)

This suggests the minimal features may not capture subtle unbalance signatures.

## Use Cases

### When to Use Minimal RFC (Approach 3):

✅ **Educational/Research Purposes**
- Demonstrating classical ML approaches
- Comparing deep learning vs. classical ML
- Understanding feature engineering

✅ **Resource-Constrained Environments**
- Limited computational power
- Cannot use TensorFlow/GPU
- Need fast inference (<1ms)

✅ **Interpretability Required**
- Need to explain predictions
- Regulatory or safety-critical contexts
- Feature importance analysis needed

✅ **Strong Unbalances Only**
- If only detecting severe unbalance (levels 3-4)
- 99-100% accuracy on these levels

### When to Use Approaches 1 or 2 Instead:

⚠️ **Production/Critical Applications**
- Need highest accuracy (99%+)
- Must detect all unbalance levels reliably
- TensorFlow/computational resources available

⚠️ **Weak Unbalance Detection**
- Need to detect level 1 unbalances
- Require consistent performance across all levels

## Conclusion

We chose the **Minimal Features variant** because it is:
1. **Implementable**: All code and data available
2. **Practical**: Fast enough for real-time use
3. **Simple**: Easy to understand and maintain
4. **Sufficient**: Adequate for demonstration purposes

However, users should be aware that **Approaches 1 and 2 provide significantly better accuracy** (99% vs. 82%) and should be preferred for production use.

The minimal features approach serves as a useful comparison point showing:
- Classical ML can work for unbalance detection
- Simple features capture major unbalance patterns
- But deep learning provides superior performance
- Trade-off: Speed/simplicity vs. accuracy

---

**Date**: November 20, 2025
**Decision**: Implement Approach 3 with Minimal Features (7 features)
**Status**: Approved for implementation
