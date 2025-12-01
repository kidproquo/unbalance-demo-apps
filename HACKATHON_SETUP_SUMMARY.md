# Hackathon Setup Summary

## Overview

The hackathon configuration creates an interesting multi-model consensus detection challenge where participants build agents to analyze detections from three ML models (CNN, FFT, RFC) and determine when there's genuine agreement on unbalance events.

## System Configuration

### Data Distribution
- **92% Normal** (0E - no unbalance)
- **8% Unbalanced** distributed across:
  - ~2.7% Level 1 (subtle unbalance)
  - ~2.7% Level 2 (moderate unbalance)
  - ~2.7% Level 3 (moderate-severe unbalance)
- **Publishing rate:** 1.5 windows/second

### Model Thresholds (Creates Disagreement)
- **CNN:** 0.90 (most balanced, catches subtle cases)
- **FFT:** 0.95 (most conservative, only high-confidence detections)
- **RFC:** 0.92 (middle ground)

## Verified Performance Metrics

Based on 5-minute test run with clean data:

### Detection Rates
- **CNN:** 38 detections (most sensitive)
- **FFT:** 27 detections (most selective)
- **RFC:** 18 detections (fewest detections)

### Consensus Statistics
- **Consensus frequency:** 5.4 events/minute (~1 every 11 seconds)
- **2-model consensus:** 38.5% (interesting disagreement patterns)
- **3-model consensus:** 61.5% (confident detections)

### Consensus Patterns
- **CNN+FFT+RFC:** 61.5% (all three models agree - high confidence)
- **CNN+FFT:** 34.6% (RFC misses - most common disagreement)
- **CNN+RFC:** 3.8% (FFT too conservative - rare)
- **FFT+RFC:** 0% (CNN rarely misses when FFT/RFC agree)

### Timing Patterns
- **Large gaps:** 32s, 25s, 15s, 13s (good analysis time)
- **Quick bursts:** 3-7s (when unbalances cluster - realistic!)
- **Average interval:** 11 seconds between consensus events

## Why This Configuration Is Good

### ✅ Interesting Disagreement
- 38.5% of consensus events involve only 2 models
- Creates challenging decision-making scenarios
- Multiple alliance patterns to discover

### ✅ Not Overwhelming
- ~11 seconds between events on average
- Enough gaps (15-32s) for analysis and decision-making
- Not too frequent to cause alert fatigue

### ✅ Realistic Patterns
- Mix of bursts (3-7s) and calm periods (15-32s)
- Mimics real industrial monitoring scenarios
- Different severity levels create different agreement patterns

### ✅ Clear Dominant Pattern
- CNN+FFT is the most common 2-model pair (34.6%)
- Participants can learn that FFT is most conservative
- RFC often misses what CNN+FFT catch

## Expected Participant Behavior

### Level 1 (Subtle Unbalances)
- Often only CNN detects
- Creates "only-1 model" scenarios
- Good test of conservative vs aggressive strategies

### Level 2 (Moderate Unbalances)
- CNN+RFC typically agree
- FFT misses ~50% of the time due to high threshold
- Most common source of 2-model consensus

### Level 3 (Moderate-Severe Unbalances)
- All 3 models usually agree
- High confidence scores (>0.90)
- Clear consensus events

## n8n Workflow Integration

The hackathon includes an n8n workflow (`n8n-hackathon-workflow.json`) that:

1. **Connects to MCP servers** on ports 12001 (CNN), 12002 (FFT), 12003 (RFC)
2. **AI Agent analyzes consensus** using OpenAI with detailed system prompt
3. **Deduplication logic** via Redis Chat Memory (conversation-based)
4. **Conditional figure loading** - only reads PNG files when consensus=true
5. **Automatic execution** with hardcoded prompt

### Key Features:
- **Temporal correlation:** 5-second window for grouping detections
- **Decision criteria:** 2+ models + confidence ≥0.85 + not previously reported
- **JSON output format:** consensus, models_agreeing, confidence, timestamp, reason, figure filenames
- **Volume mount:** `/unbalance-outputs/` mapped to `./outputs/`

## Starting the Hackathon

```bash
cd /root/dev/ps_workshop/unbalance-demo-apps

# Start all services
docker compose -f docker-compose.hackathon.yml up

# Wait for "Starting continuous publishing" message
# Then import n8n-hackathon-workflow.json into n8n
```

## Monitoring

```bash
# Check coordinator
docker compose -f docker-compose.hackathon.yml logs -f coordinator

# Check individual models
docker compose -f docker-compose.hackathon.yml logs -f approach-cnn
docker compose -f docker-compose.hackathon.yml logs -f approach-fft
docker compose -f docker-compose.hackathon.yml logs -f approach-rfc

# View detection counts
docker compose -f docker-compose.hackathon.yml logs approach-cnn | grep "UNBALANCE DETECTED" | wc -l
```

## Files and Outputs

### Detection Logs (No Ground Truth)
- `./outputs/cnn/detections.jsonl` - CNN detections
- `./outputs/fft/detections.jsonl` - FFT detections
- `./outputs/rfc/detections.jsonl` - RFC detections

Each detection contains:
```json
{
  "timestamp": "2025-12-01 17:30:44",
  "window_idx": 123,
  "start_idx": 503808,
  "end_idx": 507904,
  "prediction": 0.9876,
  "confidence": 0.9876,
  "figure_file": "unbalance_detection_*.png",
  "approach": "cnn"
}
```

**Note:** No `dataset` field (ground truth removed for hackathon)

### Performance Reports (Organizers Only)
- `./outputs/*/performance_report_*.txt` - Contains ground truth metrics
- Internal Docker logs show dataset labels (0E/1E/2E/3E)

### Visualization Figures
- `./outputs/cnn/*.png` - CNN detection plots
- `./outputs/fft/*.png` - FFT detection plots
- `./outputs/rfc/*.png` - RFC detection plots

## Ground Truth Handling

**CRITICAL:** The system handles ground truth in two modes:

### Production Mode (Hackathon Participants)
- ❌ Detection logs (`detections.jsonl`) - NO ground truth
- ❌ MCP tool responses - NO ground truth
- ✅ Participants build consensus logic without knowing true labels

### Monitoring Mode (Organizers)
- ✅ Redis stream messages - Contains dataset labels
- ✅ Performance reports - Contains accuracy metrics
- ✅ Docker logs - Shows actual unbalance levels
- ✅ Internal metrics tracking - TP/FP/TN/FN calculated

This allows organizers to monitor performance while participants work blindly.

## Tips for Participants

1. **Start simple:** 2-of-3 voting logic
2. **Weight by confidence:** Higher confidence = more reliable
3. **Use temporal correlation:** Group detections within 5 seconds
4. **Learn model patterns:** CNN+FFT is most common pair
5. **Adjust thresholds:** Balance precision vs recall

## Success Metrics

After the hackathon, evaluate participant agents on:

1. **Precision:** True alerts / (True alerts + False alerts)
2. **Recall:** True alerts / Total real unbalances
3. **Latency:** Time from first detection to alert
4. **False alarm rate:** False alerts / Total windows

Ground truth will be revealed after the challenge for scoring.
