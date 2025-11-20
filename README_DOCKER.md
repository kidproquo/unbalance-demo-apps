# Docker Build and Deployment

This directory contains Docker configuration for the unbalance detection system.

## Quick Start

### Build Image

```bash
# Build with auto-incremented patch version (0.1.0 -> 0.1.1)
./docker-build.sh

# Build and push to GitHub Container Registry
./docker-build.sh --push

# Increment minor version (0.1.0 -> 0.2.0)
./docker-build.sh --minor

# Increment major version (0.1.0 -> 1.0.0)
./docker-build.sh --major
```

### Run Services

```bash
# Run all services (coordinator + 3 approaches)
docker-compose up

# Run specific service
docker-compose up approach-cnn

# Scale a service
docker-compose up --scale approach-cnn=3
```

## Docker Build Script

The `docker-build.sh` script provides automated versioning:

### Features

- **Auto-versioning**: Reads version from `.docker-version` and increments
- **Semantic versioning**: Supports major.minor.patch (e.g., 0.1.0)
- **Multiple tags**: Creates both versioned and `:latest` tags
- **Registry push**: Optional push to GitHub Container Registry
- **Build validation**: Only updates version if build succeeds

### Usage Examples

```bash
# Basic build (increments patch: 0.1.0 -> 0.1.1)
./docker-build.sh

# Build with minor version bump (0.1.1 -> 0.2.0)
./docker-build.sh --minor

# Build with major version bump (0.2.0 -> 1.0.0)
./docker-build.sh --major

# Build and push to registry
./docker-build.sh --push

# Build minor version and push
./docker-build.sh --minor --push
```

### Version File

The current version is stored in `.docker-version`:
```
0.1.0
```

This file is tracked in git to maintain version history.

## GitHub Container Registry Setup

### Authentication

To push images to `ghcr.io`, you need a GitHub Personal Access Token:

1. Create token at: https://github.com/settings/tokens
   - Select: `write:packages` and `read:packages`

2. Login to registry:
   ```bash
   echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin
   ```

3. Build and push:
   ```bash
   ./docker-build.sh --push
   ```

### Image Location

Images are pushed to:
```
ghcr.io/kidproquo/unbalanced-demo-apps:v0.1.0
ghcr.io/kidproquo/unbalanced-demo-apps:latest
```

### Pull Image

```bash
# Pull specific version
docker pull ghcr.io/kidproquo/unbalanced-demo-apps:v0.1.0

# Pull latest
docker pull ghcr.io/kidproquo/unbalanced-demo-apps:latest
```

## Docker Compose

The `docker-compose.yml` file defines 5 services:

1. **redis**: Redis server for coordination
2. **coordinator**: Publishes windows to stream
3. **approach-cnn**: CNN-based detection
4. **approach-fft**: FFT FCN-based detection
5. **approach-rfc**: Random Forest-based detection

### Configuration

Edit `docker-compose.yml` to customize:

```yaml
coordinator:
  command: >
    python -u data_coordinator.py
    --max-windows 100        # Number of windows
    --time-window 10         # Window size in seconds
    --normal-weight 0.9      # 90% normal, 10% unbalanced
```

### Volumes

- **Data**: Mount dataset as read-only volume
  ```yaml
  volumes:
    - ../data:/app/data:ro
  ```

- **Output**: Shared output directory for detection figures
  ```yaml
  volumes:
    - ./figures/detections:/app/figures/detections
  ```

### Environment Variables

Set Redis connection via environment:
```yaml
environment:
  - REDIS_HOST=redis
  - REDIS_PORT=6379
```

## Dockerfile

The `Dockerfile` creates a single image containing:

- All 3 approaches (CNN, FFT, RFC)
- All 3 trained models
- Data coordinator
- Shared utilities
- All dependencies

### Build Arguments

```bash
docker build \
  --build-arg VERSION=0.1.0 \
  -t ghcr.io/kidproquo/unbalanced-demo-apps:v0.1.0 \
  .
```

### Image Labels

The image includes OCI-compliant labels:
```
org.opencontainers.image.title="Unbalanced Demo Apps"
org.opencontainers.image.version="0.1.0"
org.opencontainers.image.created="2025-11-20T17:30:00Z"
```

View labels:
```bash
docker inspect ghcr.io/kidproquo/unbalanced-demo-apps:latest | grep -A 10 Labels
```

## Development Workflow

### Local Testing

```bash
# Build image
./docker-build.sh

# Test with docker-compose
docker-compose up

# Or test individual service
docker run --rm ghcr.io/kidproquo/unbalanced-demo-apps:latest \
  python data_coordinator.py --max-windows 10
```

### Release Workflow

```bash
# 1. Make changes to code
# 2. Test locally
docker-compose up

# 3. Build and push new version
./docker-build.sh --minor --push

# 4. Tag release in git
git tag v0.2.0
git push origin v0.2.0
```

## Troubleshooting

### Build Fails

Check the build log:
```bash
./docker-build.sh 2>&1 | tee build.log
```

Common issues:
- Missing model files in `models/`
- Requirements.txt conflicts
- Insufficient disk space

### Version Not Incrementing

The script only updates version if build succeeds. Check:
```bash
cat .docker-version  # Current version
```

Manually set version:
```bash
echo "0.2.0" > .docker-version
```

### Push Fails

Ensure you're logged in:
```bash
docker login ghcr.io
```

Check image name matches:
```bash
docker images | grep unbalanced-demo-apps
```

### Image Too Large

Current image size: ~2.5GB (includes TensorFlow)

To reduce:
- Use multi-stage build
- Remove unnecessary dependencies
- Use alpine-based images (requires compilation)

## Production Deployment

### Using Pre-built Image

```bash
# Pull from registry
docker pull ghcr.io/kidproquo/unbalanced-demo-apps:v0.1.0

# Update docker-compose.yml to use pre-built image
services:
  coordinator:
    image: ghcr.io/kidproquo/unbalanced-demo-apps:v0.1.0
    # Remove: build: .
```

### Health Checks

Redis includes health check:
```yaml
healthcheck:
  test: ["CMD", "redis-cli", "ping"]
  interval: 5s
  timeout: 3s
  retries: 5
```

Add health checks to approaches:
```yaml
approach-cnn:
  healthcheck:
    test: ["CMD", "python", "-c", "import redis; redis.Redis().ping()"]
    interval: 30s
    timeout: 10s
    retries: 3
```

### Monitoring

View logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f approach-cnn

# Last 100 lines
docker-compose logs --tail=100 approach-fft
```

## References

- [Docker Build](https://docs.docker.com/engine/reference/commandline/build/)
- [Docker Compose](https://docs.docker.com/compose/)
- [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [OCI Image Spec](https://github.com/opencontainers/image-spec/blob/main/annotations.md)
