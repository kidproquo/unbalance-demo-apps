# Performance Analysis - Unbalance Detection System

**Date:** 2025-11-24
**System Configuration:** Docker Compose with 3 ML approaches + Redis coordinator

## Processing Time Breakdown

### CNN Approach
```
Sample timing logs (Window 123-133):
[CNN] Window 123: read=443.8ms, parse=0.0ms, predict=68.9ms, ack=0.3ms, total=513.0ms
[CNN] Window 124: read=1075.9ms, parse=0.0ms, predict=63.5ms, ack=0.5ms, total=1139.9ms
[CNN] Window 125: read=1013.3ms, parse=0.0ms, predict=69.0ms, ack=0.3ms, total=1082.7ms
[CNN] Window 126: read=1080.7ms, parse=0.0ms, predict=63.9ms, ack=0.3ms, total=1144.9ms
[CNN] Window 127: read=994.3ms, parse=0.0ms, predict=64.6ms, ack=0.3ms, total=1059.3ms
[CNN] Window 128: read=1050.3ms, parse=0.0ms, predict=64.2ms, ack=0.4ms, total=1114.9ms
[CNN] Window 129: read=993.1ms, parse=0.0ms, predict=66.2ms, ack=0.3ms, total=1059.6ms
[CNN] Window 130: read=1051.7ms, parse=0.0ms, predict=57.4ms, ack=0.3ms, total=1109.4ms
[CNN] Window 131: read=1033.8ms, parse=0.0ms, predict=56.6ms, ack=0.3ms, total=1090.7ms
[CNN] Window 132: read=1067.5ms, parse=0.0ms, predict=117.5ms, ack=0.5ms, total=1185.6ms
[CNN] Window 133: read=947.4ms, parse=0.0ms, predict=55.1ms, ack=0.4ms, total=1002.9ms
```

**Average timings:**
- Read (Redis): ~1000ms (waiting for messages)
- Parse: ~0.0ms (negligible - simple array indexing)
- Predict: ~55-70ms (3-layer CNN inference on 4096 samples)
- Acknowledge: ~0.3-0.5ms (negligible)
- **Total: ~1060-1140ms per window**

### FFT Approach
```
Sample timing logs (Window 123-129):
[FFT] Window 3: read=1042.2ms, parse=0.4ms, predict=123.3ms, ack=0.2ms, total=1733.9ms
[FFT] Window 123: read=375.4ms, parse=0.4ms, predict=56.1ms, ack=0.3ms, total=432.2ms
[FFT] Window 124: read=1088.5ms, parse=0.5ms, predict=67.7ms, ack=0.3ms, total=1157.1ms
[FFT] Window 125: read=1008.9ms, parse=0.5ms, predict=70.4ms, ack=0.2ms, total=1080.0ms
[FFT] Window 126: read=1078.5ms, parse=0.5ms, predict=61.6ms, ack=0.3ms, total=1140.9ms
[FFT] Window 127: read=997.2ms, parse=0.4ms, predict=67.4ms, ack=0.2ms, total=1065.4ms
[FFT] Window 128: read=1047.4ms, parse=0.4ms, predict=66.0ms, ack=0.2ms, total=1114.2ms
[FFT] Window 129: read=990.6ms, parse=0.4ms, predict=63.8ms, ack=0.3ms, total=1055.2ms
```

**Average timings:**
- Read (Redis): ~1000ms (waiting for messages)
- Parse: ~0.4-0.5ms (FFT transformation + RobustScaler)
- Predict: ~60-70ms (4-layer FCN inference on 2048 FFT features)
- Acknowledge: ~0.2-0.3ms (negligible)
- **Total: ~1060-1150ms per window**

### RFC (Random Forest) Approach
```
Sample timing logs (Window 120-129):
[RFC] Window 120: read=1057.7ms, parse=3.9ms, predict=3.5ms, ack=0.3ms, total=1065.5ms
[RFC] Window 121: read=1162.9ms, parse=1.9ms, predict=1.6ms, ack=0.2ms, total=1166.6ms
[RFC] Window 122: read=1081.1ms, parse=1.7ms, predict=1.5ms, ack=0.2ms, total=1084.5ms
[RFC] Window 3: read=1107.4ms, parse=2.5ms, predict=2.0ms, ack=0.2ms, total=1700.5ms
[RFC] Window 123: read=473.7ms, parse=2.2ms, predict=1.9ms, ack=0.2ms, total=478.0ms
[RFC] Window 124: read=1141.0ms, parse=2.5ms, predict=2.4ms, ack=0.2ms, total=1146.2ms
[RFC] Window 125: read=1072.1ms, parse=1.8ms, predict=1.7ms, ack=0.2ms, total=1075.8ms
[RFC] Window 126: read=1146.2ms, parse=2.3ms, predict=2.2ms, ack=0.3ms, total=1151.1ms
[RFC] Window 127: read=1054.6ms, parse=3.1ms, predict=3.3ms, ack=0.3ms, total=1061.4ms
[RFC] Window 128: read=1108.5ms, parse=1.8ms, predict=1.6ms, ack=0.2ms, total=1112.1ms
[RFC] Window 129: read=1054.2ms, parse=2.3ms, predict=2.1ms, ack=0.2ms, total=1058.8ms
```

**Average timings:**
- Read (Redis): ~1000ms (waiting for messages)
- Parse: ~2-4ms (statistical feature extraction: mean, std, kurtosis)
- Predict: ~1.5-3.5ms (Random Forest inference - very fast!)
- Acknowledge: ~0.2-0.3ms (negligible)
- **Total: ~1060-1150ms per window**

## Key Performance Insights

### 1. Bottleneck Analysis
- **Read time dominates** (~1000ms out of ~1100ms total) - all approaches spend ~90% of time waiting for Redis messages
- **Actual ML inference is fast**: CNN/FFT ~60-70ms, RFC ~2-3ms
- **The bottleneck is the data pipeline, not the models**

### 2. Model Comparison
| Approach | Parse Time | Predict Time | Total Processing | Relative Speed |
|----------|------------|--------------|------------------|----------------|
| CNN      | ~0.0ms     | ~60-70ms     | ~60-70ms         | 1x (baseline)  |
| FFT      | ~0.5ms     | ~60-70ms     | ~60-70ms         | ~1x            |
| RFC      | ~2-3ms     | ~2-3ms       | ~4-6ms           | **10-15x faster** |

**Winner: RFC** is 10-15x faster at actual processing, though all approaches are bottlenecked by the 1s message rate.

### 3. Coordinator Publishing Rate
```
Published 10 windows (Rate: 1.0 windows/s, Queue: 1000)
Published 20 windows (Rate: 1.0 windows/s, Queue: 1000)
...
Published 300 windows (Rate: 0.9 windows/s, Queue: 1000)
```

- Coordinator publishes at **~0.9-1.0 windows/second** (real-time speed)
- Queue stays constant at **1000 messages** (configured maxlen)
- All consumers keep up with this rate

### 4. Redis Consumer Group Status
```
name: cnn-group
  consumers: 1
  pending: 0
  lag: 0
  entries-read: 6219

name: fft-group
  consumers: 1
  pending: 0
  lag: 0
  entries-read: 6219

name: rfc-group
  consumers: 1
  pending: 0
  lag: 0
  entries-read: 6219
```

**Health indicators:**
- ✅ **pending: 0** - All messages are being acknowledged
- ✅ **lag: 0** - All consumers are caught up
- ✅ **Same entries-read** - All approaches processing at same rate

## Memory Management

### Problem Identified
Original configuration led to OOM (Out of Memory) kills:
- Redis stream with `maxlen=10000`
- Each message: ~131KB (4096 samples × 4 columns × 8 bytes)
- Total queue: 10,000 × 131KB = **~1.3GB in Redis alone**
- Coordinator: 4G limit with full dataset in memory
- Combined usage exceeded available memory

### Solution Implemented
1. **Reduced stream maxlen**: 10,000 → 1,000 with `approximate=True`
   - New queue size: 1,000 × 131KB = **~130MB** (10x reduction)
   - `approximate=True` allows Redis to trim in batches (more efficient)

2. **Added Redis memory limit**: 1G hard limit
   - Prevents unbounded growth
   - Forces eviction if limit reached

3. **Result**: Queue stays at 1000 messages (expected behavior)
   - This is a **buffer of recent history**, not a backlog
   - All consumers keep up (lag=0, pending=0)
   - System is stable

## System Throughput

### Current Performance
- **Processing rate**: ~1 window/second per approach
- **Synchronized processing**: All 3 approaches process the same windows
- **Total system throughput**: 3 approaches × 1 window/s = **3 detections/second**
- **Latency**: ~60-70ms (CNN/FFT) or ~4-6ms (RFC) from message receipt to acknowledgment

### Theoretical Maximum
If not throttled by coordinator's 1s/window rate:
- **CNN**: ~14-17 windows/second (1000ms / 70ms)
- **FFT**: ~14-17 windows/second (1000ms / 70ms)
- **RFC**: ~170-250 windows/second (1000ms / 4ms)

**Conclusion**: System is currently **I/O bound** (waiting for data), not **compute bound**. Models could process 14-250x faster if data was available.

## Recommendations

### For Higher Throughput
1. **Increase coordinator speed**: Use `--speed 10.0` or `--speed 0` (max)
2. **Parallel processing**: Each approach could handle multiple workers
3. **Batch processing**: Send multiple windows per message

### For Lower Latency
1. **RFC is optimal**: 4-6ms total processing time
2. **Remove Redis**: Direct streaming would save ~1000ms read time
3. **Optimize FFT/CNN**: Model quantization or ONNX runtime

### For Resource Efficiency
1. **Current config is good**: 1000 message buffer is appropriate
2. **RFC uses least resources**: Fastest inference, smallest model
3. **Consider disabling slower approaches** if only speed matters

## Detection Accuracy (with 0.9 threshold)

From logs at 100-110 windows processed:

### FFT (Best Overall)
- **0E accuracy**: 99.0% (99 TN, 1 FP out of 100)
- **1E-4E accuracy**: 100% (all detected)
- **Overall accuracy**: 99.1%

### CNN (Good Balance)
- **0E accuracy**: 84.9% (79 TN, 14 FP out of 93)
- **1E-4E accuracy**: 100% (all detected)
- **Overall accuracy**: 86.0%

### RFC (High Precision, Low Recall)
- **0E accuracy**: 100% (100 TN, 0 FP out of 100)
- **1E-4E accuracy**: 20% (2 TP, 8 FN out of 10)
- **Overall accuracy**: 92.7%

**Trade-offs:**
- **FFT**: Best all-around performer
- **CNN**: Moderate false positives, no false negatives
- **RFC**: No false positives but misses subtle unbalances (0.9 threshold too high for RFC)
