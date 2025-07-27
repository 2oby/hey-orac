#!/bin/bash
set -e

# Script to build the Hey ORAC Docker image

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
TAG="${1:-hey-orac:latest}"
PLATFORM="${2:-linux/arm64,linux/arm/v7}"

echo "Building Hey ORAC Docker image..."
echo "Tag: $TAG"
echo "Platform: $PLATFORM"

# Change to project directory
cd "$PROJECT_DIR"

# Build the image
docker buildx build \
    --platform "$PLATFORM" \
    --tag "$TAG" \
    --load \
    .

echo "Build complete!"
echo "Run with: docker run --device /dev/snd -v \$(pwd)/config:/config -p 7171:7171 $TAG"