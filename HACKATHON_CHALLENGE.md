# Hackathon Challenge: Multi-Model Consensus Alert System

## Challenge Overview

Build an intelligent agent that analyzes detections from three different machine learning models (CNN, FFT, RFC) and raises alerts when there's consensus on unbalance detection. Your agent must handle disagreement, temporal correlation, and minimize false alarms.

## The Problem

You have access to **three ML models** running in real-time, each detecting unbalances in rotating machinery:

1. **CNN (Convolutional Neural Network)** - Analyzes raw vibration waveforms
2. **FFT (Fast Fourier Transform + FCN)** - Analyzes frequency domain patterns
3. **RFC (Random Forest Classifier)** - Analyzes statistical features

**The catch:** These models don't always agree! They have different strengths and weaknesses:
- Different detection rates
- Different confidence thresholds
- Different response times
- Different accuracy on subtle vs severe unbalances

## Your Mission

Create an agent that:

1. **Subscribes** to detection streams from all 3 models via MCP (Model Context Protocol)
2. **Correlates** detections across models by timestamp/window
3. **Applies intelligent voting logic** to determine when to raise an alert
4. **Minimizes false positives** while maintaining high recall

## What You Have Access To

### Detection Data (via MCP)

Each model exposes an MCP server with detection logs. Here's what you get for **each detection**:

```json
{
  "timestamp": "2025-11-24 20:45:01",
  "window_idx": 123,
  "start_idx": 503808,
  "end_idx": 507904,
  "detection_ratio": 0.95,
  "mean_prediction": 0.94,
  "max_prediction": 0.98,
  "confidence": 0.98,
  "figure_file": "unbalance_detection_1E_20251124_204501_row503808.png",
  "approach": "cnn"
}
```

**What you DON'T get:** Ground truth labels! You won't know if it's really an unbalance (0E, 1E, 2E, etc.) - that's what makes it realistic!

### MCP Server Endpoints

Connect to these endpoints to fetch detections:

| Model | Port  | MCP Tools Available |
|-------|-------|---------------------|
| CNN   | 12001 | `cnn_get_recent_detections(limit=100)` |
| FFT   | 12002 | `fft_get_recent_detections(limit=100)` |
| RFC   | 12003 | `rfc_get_recent_detections(limit=100)` |

Example using MCP:
```python
from mcp import Client

# Connect to CNN MCP server
cnn_client = Client("http://localhost:12001")
detections = cnn_client.call_tool("cnn_get_recent_detections", {"limit": 50})
```

## Evaluation Criteria

After the challenge, we'll reveal ground truth and score your agent on:

### 1. Precision (40 points)
```
Precision = True Alerts / (True Alerts + False Alerts)
```
How many of your alerts were real unbalances?

### 2. Recall (30 points)
```
Recall = True Alerts / Total Real Unbalances
```
How many real unbalances did you catch?

### 3. Latency (20 points)
```
Latency = Time from first model detection to your alert
```
How quickly did you raise alerts after the first model detected something?

### 4. False Alarm Rate (10 points)
```
False Alarm Rate = False Alerts / Total Windows
```
How often did you cry wolf on normal data?

## Suggested Approaches

### Approach 1: Simple 2-of-3 Vote
```python
def should_alert(cnn_detected, fft_detected, rfc_detected):
    votes = sum([cnn_detected, fft_detected, rfc_detected])
    return votes >= 2
```

**Pros:** Simple, good precision
**Cons:** May miss cases where only 1 model detects (low recall)

### Approach 2: Weighted Vote by Confidence
```python
def should_alert(detections):
    # Weight each model's vote by its confidence score
    weighted_sum = (
        detections['cnn']['confidence'] * 0.3 +
        detections['fft']['confidence'] * 0.5 +
        detections['rfc']['confidence'] * 0.2
    )
    return weighted_sum > 0.7
```

**Pros:** Considers confidence, can tune weights
**Cons:** Need to learn good weights

### Approach 3: Temporal Correlation
```python
def should_alert(detection_history, time_window=5):
    # Check if 2+ models detected within 5 seconds
    recent = [d for d in detection_history
              if d['timestamp'] > now - time_window]

    models_detected = set([d['approach'] for d in recent])
    return len(models_detected) >= 2
```

**Pros:** Handles timing differences
**Cons:** More complex, requires buffering

### Approach 4: Model-Specific Reliability
```python
# Trust FFT and CNN more than RFC for subtle unbalances
def should_alert(detections):
    if detections['fft']['detected'] and detections['cnn']['detected']:
        return True  # High confidence
    if all([d['detected'] for d in detections.values()]):
        return True  # All 3 agree
    return False  # Otherwise wait
```

**Pros:** Uses domain knowledge
**Cons:** Requires understanding model strengths

## Challenge Modes

### Mode 1: Easy (Warm-up) - 50 points
- **Data:** Only severe unbalances (3E, 4E)
- **Agreement:** Models agree ~95% of the time
- **Goal:** Get 90%+ precision and recall

### Mode 2: Medium (Main Challenge) - 100 points
- **Data:** Mix of Level 1 and Level 2 unbalances
- **Agreement:** Models agree ~60% of the time
- **Goal:** Balance precision vs recall

### Mode 3: Hard (Advanced) - 150 points
- **Data:** All levels (1E-4E) with different thresholds per model
- **Agreement:** Models agree ~40-50% of the time
- **Goal:** Achieve >80% precision AND >80% recall

## Getting Started

### 1. Spin Up the System

```bash
cd unbalance-demo-apps
docker compose up
```

Wait for all services to be healthy (~1-2 minutes).

### 2. Verify MCP Servers

```bash
# Test CNN endpoint
curl http://localhost:12001/tools

# You should see: cnn_get_recent_detections, cnn_get_performance_metrics, etc.
```

### 3. Start Building Your Agent

Create `my_agent.py`:

```python
import time
import requests
from datetime import datetime

def fetch_detections(approach, port, limit=20):
    """Fetch recent detections from an MCP server."""
    url = f"http://localhost:{port}/call_tool"
    response = requests.post(url, json={
        "name": f"{approach}_get_recent_detections",
        "arguments": {"limit": limit}
    })
    return response.json()

def main():
    print("Starting alert agent...")

    while True:
        # Fetch from all 3 models
        cnn_detections = fetch_detections("cnn", 12001)
        fft_detections = fetch_detections("fft", 12002)
        rfc_detections = fetch_detections("rfc", 12003)

        # YOUR LOGIC HERE
        # Analyze detections and decide when to alert

        time.sleep(1)  # Poll every second

if __name__ == "__main__":
    main()
```

### 4. Test Your Agent

Run your agent and watch it make decisions:
```bash
python my_agent.py
```

## Tips & Tricks

1. **Start simple:** Get a basic 2-of-3 vote working first
2. **Log everything:** You'll want to analyze your agent's decisions later
3. **Handle edge cases:** What if a model returns no detections?
4. **Timestamps matter:** Models may detect at slightly different times
5. **Confidence scores:** Higher confidence = more reliable detection
6. **Consider latency:** The faster you alert, the better (but don't sacrifice accuracy)

## Common Pitfalls

❌ **Assuming perfect synchronization** - Models detect at different times
❌ **Ignoring confidence scores** - A 0.91 detection is different from 0.99
❌ **No deduplication** - Don't alert multiple times for the same unbalance
❌ **Hardcoded logic** - What works for Mode 1 won't work for Mode 3
❌ **Not handling missing data** - What if one model is slow?

## Bonus Points

- **Adaptive thresholds:** Adjust voting logic based on detection patterns (+10 pts)
- **Explainability:** Log WHY you raised each alert (+10 pts)
- **Real-time visualization:** Dashboard showing model agreement (+15 pts)
- **Multi-window correlation:** Look at sequences of windows (+15 pts)

## Submission

Submit:
1. Your agent code (`my_agent.py`)
2. Alert log (JSONL format with timestamp, reason, models_agreed)
3. Brief explanation of your approach (max 500 words)

## Example Alert Log Format

```jsonl
{"timestamp": "2025-11-24 20:45:03", "alert": true, "reason": "CNN+FFT agreement", "confidence": 0.95, "models": ["cnn", "fft"]}
{"timestamp": "2025-11-24 20:45:15", "alert": true, "reason": "All 3 models agree", "confidence": 0.98, "models": ["cnn", "fft", "rfc"]}
{"timestamp": "2025-11-24 20:45:28", "alert": false, "reason": "Only RFC detected, low confidence", "confidence": 0.52, "models": ["rfc"]}
```

## Questions?

- **Q: Can I use the MCP performance_metrics endpoint?**
  A: No! That contains ground truth. Detection logs only.

- **Q: Can I train my own model?**
  A: No, but you can learn weights/thresholds for voting logic.

- **Q: What if models disagree forever?**
  A: That's the challenge! Design logic to handle it.

- **Q: Can I use external data?**
  A: No, only detection logs from the 3 MCP servers.

## Good Luck! 🚀

May your precision be high and your false alarms be low!
