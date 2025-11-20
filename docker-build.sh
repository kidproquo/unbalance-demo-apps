#!/bin/bash
# Docker build script with auto-versioning for unbalanced-demo-apps
#
# Usage:
#   ./docker-build.sh           # Build and increment patch version (0.1.0 -> 0.1.1)
#   ./docker-build.sh --push    # Build, increment, and push to registry
#   ./docker-build.sh --major   # Increment major version (0.1.0 -> 1.0.0)
#   ./docker-build.sh --minor   # Increment minor version (0.1.0 -> 0.2.0)

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
REGISTRY="ghcr.io"
REPO="kidproquo/unbalanced-demo-apps"
VERSION_FILE=".docker-version"
IMAGE_NAME="${REGISTRY}/${REPO}"

# Parse arguments
PUSH=false
VERSION_BUMP="patch"  # default: patch

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --major)
            VERSION_BUMP="major"
            shift
            ;;
        --minor)
            VERSION_BUMP="minor"
            shift
            ;;
        --patch)
            VERSION_BUMP="patch"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: $0 [--push] [--major|--minor|--patch]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Docker Build Script - Unbalanced Demo Apps${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo

# Read current version
if [[ ! -f "$VERSION_FILE" ]]; then
    echo -e "${YELLOW}Version file not found. Creating with initial version 0.1.0${NC}"
    echo "0.1.0" > "$VERSION_FILE"
fi

CURRENT_VERSION=$(cat "$VERSION_FILE")
echo -e "${BLUE}Current version: ${GREEN}v${CURRENT_VERSION}${NC}"

# Parse version components
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

# Increment version based on bump type
case $VERSION_BUMP in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
esac

NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"
echo -e "${BLUE}New version:     ${GREEN}v${NEW_VERSION}${NC}"
echo

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
echo -e "${BLUE}Tags:${NC}"
echo -e "  - ${IMAGE_NAME}:v${NEW_VERSION}"
echo -e "  - ${IMAGE_NAME}:latest"
echo

docker build \
    -t "${IMAGE_NAME}:v${NEW_VERSION}" \
    -t "${IMAGE_NAME}:latest" \
    --build-arg VERSION="${NEW_VERSION}" \
    .

if [[ $? -eq 0 ]]; then
    echo
    echo -e "${GREEN}✓ Docker image built successfully${NC}"

    # Update version file only if build succeeded
    echo "$NEW_VERSION" > "$VERSION_FILE"
    echo -e "${GREEN}✓ Version updated to v${NEW_VERSION}${NC}"

    # Show image info
    echo
    echo -e "${BLUE}Image details:${NC}"
    docker images "${IMAGE_NAME}" | head -2
else
    echo
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

# Push to registry if requested
if [[ "$PUSH" = true ]]; then
    echo
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${YELLOW}Pushing to registry...${NC}"
    echo -e "${BLUE}================================================================================${NC}"
    echo

    # Check if logged in to GitHub Container Registry
    echo -e "${YELLOW}Checking authentication...${NC}"
    if ! docker login ${REGISTRY} 2>/dev/null; then
        echo -e "${RED}Not logged in to ${REGISTRY}${NC}"
        echo -e "${YELLOW}Please login first:${NC}"
        echo -e "  echo \$GITHUB_TOKEN | docker login ${REGISTRY} -u USERNAME --password-stdin"
        exit 1
    fi

    echo -e "${YELLOW}Pushing ${IMAGE_NAME}:v${NEW_VERSION}...${NC}"
    docker push "${IMAGE_NAME}:v${NEW_VERSION}"

    echo -e "${YELLOW}Pushing ${IMAGE_NAME}:latest...${NC}"
    docker push "${IMAGE_NAME}:latest"

    echo
    echo -e "${GREEN}✓ Images pushed successfully${NC}"
    echo
    echo -e "${BLUE}Pull with:${NC}"
    echo -e "  docker pull ${IMAGE_NAME}:v${NEW_VERSION}"
    echo -e "  docker pull ${IMAGE_NAME}:latest"
fi

echo
echo -e "${BLUE}================================================================================${NC}"
echo -e "${GREEN}Build complete!${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo
echo -e "${BLUE}Image tags:${NC}"
echo -e "  ${IMAGE_NAME}:v${NEW_VERSION}"
echo -e "  ${IMAGE_NAME}:latest"
echo
echo -e "${BLUE}Run with docker-compose:${NC}"
echo -e "  docker-compose up"
echo
echo -e "${BLUE}Or run specific service:${NC}"
echo -e "  docker run --rm ${IMAGE_NAME}:v${NEW_VERSION} python data_coordinator.py --max-windows 10"
echo
