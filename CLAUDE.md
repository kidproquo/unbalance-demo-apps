# Unbalance Detection Demo Apps

Docker-based demonstration of three machine learning approaches for detecting unbalance in rotating machinery using the Fraunhofer EAS dataset.

## Architecture

- **Redis Streams**: Coordinates synchronized window processing across all approaches
- **Data Coordinator**: Publishes time windows to Redis stream
- **Three ML Approaches**: CNN, FFT-FCN, and Random Forest classifiers consume windows in parallel
- **MCP Servers**: Each approach exposes metrics via FastMCP (HTTP streamable transport)

## Directory Structure

```
apps/
├── approach_1_cnn/          # CNN on raw sensor data
│   ├── approach_1_cnn.py
│   ├── mcp_server.py
│   └── requirements.txt
├── approach_2_fft/          # FFT-FCN on frequency data
│   ├── approach_2_fft.py
│   ├── mcp_server.py
│   └── requirements.txt
├── approach_3_minimal_rfc/  # Random Forest on minimal features
│   ├── approach_3_minimal_rfc.py
│   ├── mcp_server.py
│   └── requirements.txt
├── utils/                   # Shared utilities (Redis consumer/producer)
├── models/                  # Pre-trained model files
├── data_coordinator.py      # Publishes windows to Redis
├── docker-compose.yml       # Container orchestration
├── Dockerfile              # Multi-approach container image
└── docker-build.sh         # Build script with auto-versioning
```

## Running

```bash
# Pull and run
docker-compose pull
docker-compose up

# Or build locally
./docker-build.sh
docker-compose up
```

## Startup Sequence

1. Redis (health check: ping)
2. CNN approach (loads model/data, writes /tmp/ready)
3. FFT approach (waits for CNN healthy)
4. RFC approach (waits for FFT healthy)
5. Coordinator (waits for all approaches healthy, starts publishing)

## MCP Servers

Each approach runs an MCP server with prefixed tool names:

| Approach | Port  | Tools |
|----------|-------|-------|
| CNN      | 12001 | `cnn_get_performance_metrics`, `cnn_get_recent_detections`, `cnn_get_system_status` |
| FFT      | 12002 | `fft_get_performance_metrics`, `fft_get_recent_detections`, `fft_get_system_status` |
| RFC      | 12003 | `rfc_get_performance_metrics`, `rfc_get_recent_detections`, `rfc_get_system_status` |

## Key Configuration

- **Data path**: `/app/data/fraunhofer_eas_dataset_for_unbalance_detection_v1.zip`
- **Models**: `/app/models/` (cnn_3_layers.h5, fft_fcn_4_layers.h5, minimal_rfc.joblib)
- **Outputs**: `./outputs/{cnn,fft,rfc}/` - performance reports and detection figures
- **Output directories are cleared on each startup**

## Development Notes

- Dockerfile is layer-optimized: requirements installed before code copy
- docker-build.sh auto-increments version and removes previous image
- Health checks use file-based signals (/tmp/ready)
- Redis connection is tested at startup when --redis-mode is enabled
