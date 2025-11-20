# Synchronized Real-Time Unbalance Detection

This directory contains a complete implementation of synchronized processing across three different unbalance detection approaches using Redis Streams for coordination.

## Overview

All three approaches can now process **the exact same windows** simultaneously, enabling direct performance comparison on identical data in real-time.

### Architecture

```
┌─────────────────┐
│ Data Coordinator│  Publishes windows to Redis Stream
│  (data_coordinator.py) │  (Weighted random sampling: 90% normal, 10% unbalanced)
└────────┬────────┘
         │
         v
   ┌─────────────┐
   │ Redis Stream│  "windows" stream
   └─────┬───────┘
         │
         ├──────────────┬──────────────┬─────────────┐
         v              v              v             v
   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │Consumer │   │Consumer │   │Consumer │   │ Future  │
   │Group:   │   │  CNN    │   │  FFT    │   │Consumer │
   │detectors│   └─────────┘   └─────────┘   └─────────┘
   └─────────┘        │              │              │
                      v              v              v
              ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
              │PerformanceP │ │PerformanceP │ │PerformanceP │
              │  Reports    │ │  Reports    │ │  Reports    │
              └─────────────┘ └─────────────┘ └─────────────┘
```

### Key Features

1. **Synchronized Processing**: All approaches process identical windows
2. **Weighted Random Sampling**: 90% normal (0E), 10% unbalanced (1E-4E)
3. **Redis Streams**: Reliable message delivery with acknowledgment
4. **Consumer Groups**: Fan-out pattern ensures all consumers get all messages
5. **Performance Tracking**: TP/FP/TN/FN metrics per dataset and overall
6. **Docker-Ready**: Complete Docker Compose configuration
7. **Standalone Mode**: Each approach can still run independently

## Components

### 1. Redis Client (`utils/redis_client.py`)

Provides `WindowPublisher` and `WindowConsumer` classes for Redis Stream operations.

**Key Features:**
- Auto-detection of Docker environment (via `REDIS_HOST`, `REDIS_PORT` env vars)
- Connection retry logic with exponential backoff
- Message acknowledgment (XACK) for reliability
- Stream creation and consumer group management

### 2. Data Coordinator (`data_coordinator.py`)

Publishes window selections to Redis Stream.

**Usage:**
```bash
# Publish 100 windows
python data_coordinator.py --max-windows 100

# Continuous mode (infinite)
python data_coordinator.py

# Custom sampling weights
python data_coordinator.py --normal-weight 0.8 --max-windows 500
```

**Arguments:**
- `--max-windows`: Number of windows to publish (default: infinite)
- `--time-window`: Window duration in seconds (default: 10)
- `--normal-weight`: Probability of selecting 0E (default: 0.9)
- `--publish-rate`: Windows/second (0=maximum) (default: 0)
- `--redis-host`: Redis host (default: localhost)
- `--redis-port`: Redis port (default: 6379)
- `--redis-stream`: Stream name (default: windows)

### 3. Approach Consumers

Each approach can run in Redis consumer mode:

**Approach 1 (CNN):**
```bash
python approach_1_cnn/approach_1_cnn.py \
    --redis-mode \
    --redis-stream windows \
    --consumer-name cnn \
    --dataset all
```

**Approach 2 (FFT FCN):**
```bash
python approach_2_fft/approach_2_fft.py \
    --redis-mode \
    --redis-stream windows \
    --consumer-name fft \
    --dataset all
```

**Approach 3 (Minimal RFC):**
```bash
python approach_3_minimal_rfc/approach_3_minimal_rfc.py \
    --redis-mode \
    --redis-stream windows \
    --consumer-name rfc \
    --dataset all
```

## Setup

### Prerequisites

1. **Python 3.11** (for TensorFlow compatibility)
2. **Redis Server** (local or Docker)
3. **Dataset** downloaded and extracted

### Installation

1. Install Redis (choose one):
   ```bash
   # Option 1: Docker
   docker run -d -p 6379:6379 redis:7-alpine

   # Option 2: Homebrew (macOS)
   brew install redis
   brew services start redis

   # Option 3: apt (Ubuntu/Debian)
   sudo apt install redis-server
   sudo systemctl start redis
   ```

2. Install Python dependencies:
   ```bash
   # Install redis for all approaches
   cd apps
   pip install redis>=5.0.0

   # Or reinstall all requirements
   pip install -r approach_1_cnn/requirements.txt
   pip install -r approach_2_fft/requirements.txt
   pip install -r approach_3_minimal_rfc/requirements.txt
   ```

## Usage

### Option 1: Test Script (Easiest)

Run all approaches simultaneously with a single command:

```bash
cd apps
chmod +x test_synchronized.sh
./test_synchronized.sh 20  # Process 20 windows
```

This script:
- Starts the data coordinator
- Launches all 3 approaches in Redis mode
- Monitors progress
- Saves logs to `logs/` directory
- Automatically cleans up on exit

### Option 2: Manual (More Control)

**Terminal 1 - Start Coordinator:**
```bash
python data_coordinator.py --max-windows 100
```

**Terminal 2 - Start CNN:**
```bash
python approach_1_cnn/approach_1_cnn.py --redis-mode --consumer-name cnn --dataset all
```

**Terminal 3 - Start FFT:**
```bash
python approach_2_fft/approach_2_fft.py --redis-mode --consumer-name fft --dataset all
```

**Terminal 4 - Start RFC:**
```bash
python approach_3_minimal_rfc/approach_3_minimal_rfc.py --redis-mode --consumer-name rfc --dataset all
```

### Option 3: Docker Compose (Production)

**Build and run everything:**
```bash
cd apps
docker-compose up --build
```

**Run specific services:**
```bash
# Start only coordinator and CNN
docker-compose up coordinator approach-cnn

# Scale to multiple CNN instances
docker-compose up --scale approach-cnn=3
```

**Stop all services:**
```bash
docker-compose down
```

## Output

### Performance Reports

Each approach generates timestamped performance reports:
```
figures/detections/performance_report_20251120_173045.txt
```

Format:
```
Dataset    Total     TP       FP       TN       FN       Accuracy   Precision  Recall
----------------------------------------------------------------------------------
0E         45        0        2        43       0        0.956      0.000      0.000
1E         12        11       0        0        1        0.917      1.000      0.917
2E         14        13       0        0        1        0.929      1.000      0.929
3E         16        16       0        0        0        1.000      1.000      1.000
4E         13        13       0        0        0        1.000      1.000      1.000
----------------------------------------------------------------------------------
Overall    100       53       2        43       2        0.960      0.964      0.964
```

### Detection Figures

Unbalance detections are saved with visualization:
```
figures/detections/unbalance_detection_3E_20251120_173045_window42_row614400.png
```

### Detection Logs (JSONL)

Machine-readable logs for MCP server:
```
figures/detections/detections.jsonl
```

## Performance Comparison

Since all approaches process identical windows, you can directly compare:

1. **Accuracy**: TP+TN / Total
2. **Precision**: TP / (TP+FP)
3. **Recall**: TP / (TP+FN)
4. **Detection Rate**: % of windows where unbalance detected
5. **Processing Speed**: Windows processed per second

## Configuration

### Environment Variables (Docker)

- `REDIS_HOST`: Redis hostname (default: localhost)
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_DB`: Redis database (default: 0)

### Consumer Groups

All approaches use the same consumer group (`detectors`) to ensure each message is delivered to all consumers (fan-out pattern).

To add a new consumer:
```bash
python your_approach.py \
    --redis-mode \
    --consumer-group detectors \
    --consumer-name your_consumer_name
```

## Troubleshooting

### Redis Connection Failed

```bash
# Check Redis is running
redis-cli ping
# Should return: PONG

# If not running, start it
docker run -d -p 6379:6379 redis:7-alpine
```

### Stream Not Found

The coordinator must be running first to create the stream. Consumers will wait up to 30 seconds for the stream to appear.

### Approaches Not Processing

Check that all approaches are using:
- Same `--redis-stream` name (default: windows)
- Same `--consumer-group` (default: detectors)
- Different `--consumer-name` values

### Performance Issues

- Reduce `--time-window` for faster windows
- Increase `--log-interval` to reduce console output
- Use `--dataset 0E` to test with single dataset first

## Architecture Decisions

### Why Redis Streams?

1. **Guaranteed Delivery**: Messages are not lost
2. **Acknowledgment**: Track which consumers processed which messages
3. **Consumer Groups**: Built-in fan-out pattern
4. **Persistence**: Messages survive restarts
5. **Scalability**: Can add consumers without coordinator changes

### Why Consumer Groups?

Using a single consumer group (`detectors`) with different consumer names ensures:
- All consumers receive all messages (fan-out)
- No message is processed twice by the same consumer
- Each consumer can acknowledge independently

### Why Weighted Random Sampling?

Real-world scenarios have imbalanced data (mostly normal operation). The 90/10 split reflects typical industrial conditions while ensuring sufficient unbalanced samples for evaluation.

## Future Enhancements

Potential additions:
1. **Real-time Dashboard**: Web UI showing live performance metrics
2. **Performance Comparison Tool**: Auto-generate comparison reports
3. **Adaptive Sampling**: Adjust weights based on detection rates
4. **Multi-Stream Support**: Process different datasets simultaneously
5. **Result Aggregation**: Ensemble predictions from multiple approaches

## References

- Redis Streams: https://redis.io/docs/data-types/streams/
- Consumer Groups: https://redis.io/docs/data-types/streams/#consumer-groups
- Docker Compose: https://docs.docker.com/compose/

## License

See parent directory for license information.
