# Hackathon Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- Python 3.8+ (for your agent)
- 8GB RAM minimum

## Step 1: Start the System

```bash
cd unbalance-demo-apps

# Pull the latest images
docker compose -f docker-compose.hackathon.yml pull

# Start all services
docker compose -f docker-compose.hackathon.yml up
```

**Wait for this message:**
```
hackathon-coordinator | Starting continuous publishing (Ctrl+C to stop)...
```

This means all models are running and receiving data!

## Step 2: Verify MCP Servers

Open a new terminal and test the endpoints:

```bash
# Test CNN
curl http://localhost:12001/tools

# Test FFT
curl http://localhost:12002/tools

# Test RFC
curl http://localhost:12003/tools
```

You should see JSON responses listing available tools.

## Step 3: Fetch Your First Detections

Create `test_fetch.py`:

```python
import requests
import json

def fetch_detections(approach, port):
    """Fetch recent detections from an MCP server."""
    url = f"http://localhost:{port}/call_tool"
    payload = {
        "name": f"{approach}_get_recent_detections",
        "arguments": {"limit": 5}
    }
    response = requests.post(url, json=payload)
    return response.json()

# Fetch from all three models
print("=== CNN Detections ===")
cnn = fetch_detections("cnn", 12001)
print(json.dumps(cnn, indent=2))

print("\n=== FFT Detections ===")
fft = fetch_detections("fft", 12002)
print(json.dumps(fft, indent=2))

print("\n=== RFC Detections ===")
rfc = fetch_detections("rfc", 12003)
print(json.dumps(rfc, indent=2))
```

Run it:
```bash
python test_fetch.py
```

## Step 4: Build Your Agent

Create `my_agent.py`:

```python
#!/usr/bin/env python3
"""
My Hackathon Agent - Multi-Model Consensus Alert System
"""

import time
import requests
import json
from datetime import datetime
from collections import defaultdict

class AlertAgent:
    def __init__(self):
        self.servers = {
            'cnn': 12001,
            'fft': 12002,
            'rfc': 12003
        }
        self.detection_history = []
        self.alerts_raised = []

    def fetch_detections(self, approach, limit=20):
        """Fetch recent detections from an MCP server."""
        try:
            url = f"http://localhost:{self.servers[approach]}/call_tool"
            payload = {
                "name": f"{approach}_get_recent_detections",
                "arguments": {"limit": limit}
            }
            response = requests.post(url, json=payload, timeout=5)
            data = response.json()

            # Parse the response
            if 'content' in data:
                content = json.loads(data['content'][0]['text'])
                if content.get('status') == 'success':
                    return content.get('detections', [])
            return []
        except Exception as e:
            print(f"Error fetching {approach}: {e}")
            return []

    def should_alert(self, detections):
        """
        YOUR LOGIC HERE!

        Decide whether to raise an alert based on detections from all models.

        Args:
            detections: dict with keys 'cnn', 'fft', 'rfc', each containing
                       a list of recent detection events

        Returns:
            tuple: (should_alert: bool, reason: str, confidence: float)
        """
        # Example: Simple 2-of-3 vote
        models_with_detections = []
        max_confidence = 0

        for approach, det_list in detections.items():
            if det_list:  # Has detections
                models_with_detections.append(approach)
                # Get max confidence from this model's detections
                for det in det_list:
                    max_confidence = max(max_confidence, det.get('confidence', 0))

        num_models = len(models_with_detections)

        if num_models >= 2:
            reason = f"{'+'.join(models_with_detections)} agree"
            return True, reason, max_confidence

        return False, "Not enough agreement", max_confidence

    def raise_alert(self, reason, confidence, models):
        """Log an alert."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert': True,
            'reason': reason,
            'confidence': confidence,
            'models': models
        }
        self.alerts_raised.append(alert)

        # Write to log file
        with open('alerts.jsonl', 'a') as f:
            f.write(json.dumps(alert) + '\n')

        print(f"🚨 ALERT: {reason} (confidence: {confidence:.2f})")

    def run(self, poll_interval=1):
        """Main loop - poll servers and make decisions."""
        print("Starting Alert Agent...")
        print(f"Polling every {poll_interval} second(s)")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                # Fetch from all models
                detections = {
                    'cnn': self.fetch_detections('cnn'),
                    'fft': self.fetch_detections('fft'),
                    'rfc': self.fetch_detections('rfc')
                }

                # Count detections
                counts = {k: len(v) for k, v in detections.items()}
                print(f"Detections: CNN={counts['cnn']}, FFT={counts['fft']}, RFC={counts['rfc']}", end=' ')

                # Decide whether to alert
                should_alert, reason, confidence = self.should_alert(detections)

                if should_alert:
                    models_detecting = [k for k, v in detections.items() if v]
                    self.raise_alert(reason, confidence, models_detecting)
                else:
                    print(f"[No alert: {reason}]")

                time.sleep(poll_interval)

        except KeyboardInterrupt:
            print(f"\n\nStopping agent. Raised {len(self.alerts_raised)} alerts.")
            print(f"Alerts saved to: alerts.jsonl")

if __name__ == "__main__":
    agent = AlertAgent()
    agent.run()
```

## Step 5: Run Your Agent

```bash
python my_agent.py
```

You should see:
```
Starting Alert Agent...
Polling every 1 second(s)
Press Ctrl+C to stop

Detections: CNN=5, FFT=3, RFC=2 🚨 ALERT: cnn+fft agree (confidence: 0.95)
Detections: CNN=6, FFT=4, RFC=2 🚨 ALERT: cnn+fft agree (confidence: 0.97)
...
```

## Step 6: Analyze Your Results

Check your alert log:
```bash
cat alerts.jsonl | jq
```

Compare with ground truth (after challenge):
```bash
# Organizers will provide this script
python evaluate_agent.py alerts.jsonl
```

## Current Configuration

**System Setup:**
- **Data:** 85% normal (0E), 15% unbalanced (7.5% Level 1, 7.5% Level 2)
- **Rate:** 1.5 windows/second
- **Thresholds:** CNN=0.75, FFT=0.90, RFC=0.85

**Expected Behavior:**
- Models agree ~50-60% of the time
- CNN detects more cases (lower threshold)
- FFT is most balanced
- RFC is more conservative

## Useful Commands

```bash
# View coordinator logs (see data distribution)
docker compose -f docker-compose.hackathon.yml logs -f coordinator

# View CNN model logs
docker compose -f docker-compose.hackathon.yml logs -f approach-cnn

# Stop everything
docker compose -f docker-compose.hackathon.yml down

# Restart fresh (clears all outputs)
docker compose -f docker-compose.hackathon.yml down
docker compose -f docker-compose.hackathon.yml up
```

## Tips for Your Agent

### Improve Beyond 2-of-3 Vote:

**1. Weight by confidence:**
```python
def weighted_vote(detections):
    score = (
        0.3 * max_confidence(detections['cnn']) +
        0.5 * max_confidence(detections['fft']) +
        0.2 * max_confidence(detections['rfc'])
    )
    return score > 0.7
```

**2. Temporal correlation:**
```python
def check_temporal_agreement(detections, time_window=5):
    # Check if 2+ models detected within 5 seconds
    recent_timestamps = []
    for approach, det_list in detections.items():
        for det in det_list:
            recent_timestamps.append((approach, det['timestamp']))

    # Group by time window and count models
    # ...
```

**3. Adaptive thresholds:**
```python
def adaptive_threshold(detection_history):
    # Lower threshold if false negative rate is high
    # Raise threshold if false positive rate is high
    # (requires some ground truth feedback during training)
```

## Troubleshooting

**Problem:** "Connection refused" when calling MCP servers
- **Solution:** Wait for services to be healthy. Check `docker compose ps`

**Problem:** "No detections found"
- **Solution:** Models need time to process data. Wait 30-60 seconds after startup.

**Problem:** Too many alerts
- **Solution:** Increase your voting threshold (require more models to agree)

**Problem:** Missing alerts
- **Solution:** Lower your threshold or use "any 2 models" instead of specific pairs

## Next Steps

1. **Improve your logic** - Try different voting strategies
2. **Add logging** - Track why you made each decision
3. **Visualize** - Build a dashboard showing model agreement
4. **Optimize** - Tune for best precision/recall balance

## Good Luck! 🚀

Read the full challenge details in `HACKATHON_CHALLENGE.md`
