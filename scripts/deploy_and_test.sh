#!/usr/bin/env bash
set -euo pipefail

# Hey Orac Deploy and Test Script
# Usage: ./scripts/deploy_and_test.sh [commit_message]
# Example: ./scripts/deploy_and_test.sh "Add audio buffer implementation"

# Default parameters
COMMIT_MSG=${1:-"Option 3: Web Backend as a Service - Multi-process implementation"}
REMOTE_ALIAS="pi"
PROJECT_NAME="hey-orac"

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Hey Orac Deployment Script${NC}"
echo -e "${BLUE}============================${NC}"
echo -e "${YELLOW}Deploying to: $REMOTE_ALIAS${NC}"
echo -e "${YELLOW}Project: $PROJECT_NAME${NC}"
echo -e "${YELLOW}Commit message: $COMMIT_MSG${NC}"
echo -e "${BLUE}============================${NC}"

# Check if we're connected to the Pi
echo -e "${YELLOW}👉 Checking connection to $REMOTE_ALIAS...${NC}"
if ! ssh -q -o BatchMode=yes -o ConnectTimeout=5 "$REMOTE_ALIAS" exit; then
    echo -e "${RED}❌ Cannot connect to $REMOTE_ALIAS. Please check your SSH configuration.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Connected to $REMOTE_ALIAS${NC}"

# 1) Local: commit and push
echo -e "${YELLOW}👉 Pushing local commits to master...${NC}"

# Add all changes
echo -e "${YELLOW}Adding all changes${NC}"
git add -A

# Commit if there are any changes
if git diff --cached --quiet; then
    echo -e "${YELLOW}No changes to commit${NC}"
else
    echo -e "${YELLOW}Committing with message: $COMMIT_MSG${NC}"
    git commit -m "$COMMIT_MSG"
    
    echo -e "${YELLOW}Pushing to origin/option-3-web-backend-as-service${NC}"
    git push origin option-3-web-backend-as-service
fi

# Get current commit hash and branch
COMMIT_HASH=$(git rev-parse HEAD)
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)

echo -e "${YELLOW}Deploying commit: $COMMIT_HASH on branch: $BRANCH_NAME${NC}"

# 2) Remote: pull, build & test
echo -e "${YELLOW}👉 Running remote update & deployment on $REMOTE_ALIAS...${NC}"
ssh "$REMOTE_ALIAS" "\
    set -euo pipefail; \
    echo '${BLUE}📂 Updating code from repository...${NC}'; \
    cd \$HOME/$PROJECT_NAME; \
    git fetch origin; \
    git checkout $BRANCH_NAME || git checkout -b $BRANCH_NAME origin/$BRANCH_NAME; \
    git reset --hard $COMMIT_HASH; \
    git clean -fd; \
    
    echo '${BLUE}🔍 Checking system resources...${NC}'; \
    echo 'Memory:'; \
    free -h; \
    echo 'Disk:'; \
    df -h | grep -E '/$|/home'; \
    
    echo '${BLUE}🔍 Checking audio devices...${NC}'; \
    arecord -l || echo 'No audio devices found'; \
    
    echo '${BLUE}🔍 Checking Docker...${NC}'; \
    docker --version; \
    docker-compose --version; \
    
    echo '${BLUE}🧹 Cleaning up old Docker resources...${NC}'; \
    echo 'Disk space before cleanup:'; \
    df -h | grep -E '/$|/home'; \
    \
    echo 'Stopping old containers...'; \
    docker-compose down --remove-orphans 2>/dev/null || true; \
    docker container prune -f 2>/dev/null || true; \
    \
    echo 'Removing unused images (keeping last 2)...'; \
    docker image prune -a -f --filter \"until=24h\" 2>/dev/null || true; \
    \
    echo 'Removing unused volumes...'; \
    docker volume prune -f 2>/dev/null || true; \
    \
    echo 'Removing unused networks...'; \
    docker network prune -f 2>/dev/null || true; \
    \
    echo 'Cleaning build cache...'; \
    docker builder prune -f --filter \"until=24h\" 2>/dev/null || true; \
    \
    echo 'Disk space after cleanup:'; \
    df -h | grep -E '/$|/home'; \
    \
    echo '${GREEN}✓ Docker cleanup completed${NC}'; \
    
    echo '${BLUE}🐳 Building & starting containers...${NC}'; \
    docker-compose up --build -d; \
    
    echo '${BLUE}🔍 Checking container logs...${NC}'; \
    sleep 3; \
    docker-compose logs hey-orac | tail -n 10; \
    
    echo '${BLUE}📊 Checking resource usage...${NC}'; \
    echo 'Container status:'; \
    docker-compose ps; \
    \
    echo 'Memory usage:'; \
    free -h; \
    \
    echo 'Disk usage:'; \
    df -h | grep -E '/$|/home'; \
    
    echo '${BLUE}🔍 Checking container health...${NC}'; \
    docker-compose ps; \
    
    echo '${BLUE}🧪 Testing shared memory activation system...${NC}'; \
    echo 'Running activation system tests...'; \
    docker-compose exec -T hey-orac bash -c 'cd /app && python3 -c "import sys; sys.path.insert(0, \"/app\"); from src.shared_memory_ipc import shared_memory_ipc; shared_memory_ipc.update_activation_state(True, \"Test Model\", 0.85); print(\"✅ SharedMemoryIPC working\"); shared_memory_ipc.update_activation_state(False); print(\"✅ SharedMemoryIPC test completed\")"'; \
    
    echo '${BLUE}🌐 Testing web API endpoints...${NC}'; \
    echo 'Testing /api/activation endpoint...'; \
    curl -s http://localhost:7171/api/activation | python3 -m json.tool || echo '❌ Activation endpoint test failed'; \
    \
    echo 'Testing /api/detections endpoint...'; \
    curl -s http://localhost:7171/api/detections | python3 -m json.tool || echo '❌ Detections endpoint test failed'; \
    \
    echo '${BLUE}👀 Monitoring for 10 seconds to check for activation updates...${NC}'; \
    timeout 10 bash -c '
    while true; do
        data=$(curl -s http://localhost:7171/api/activation 2>/dev/null)
        if [ $? -eq 0 ]; then
            is_listening=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get(\"is_listening\", False))")
            rms=$(echo "$data" | python3 -c "import sys, json; print(json.load(sys.stdin).get(\"current_rms\", 0))")
            timestamp=$(date "+%H:%M:%S")
            if [ "$is_listening" = "True" ]; then
                echo "🎯 [$timestamp] ACTIVATION: Listening for wake word (RMS: $rms)"
            else
                echo "🔇 [$timestamp] Not listening (RMS: $rms)"
            fi
        else
            echo "❌ [$timestamp] Failed to get activation data"
        fi
        sleep 1
    done
    ' || echo 'Monitoring completed'; \
    
    echo '${GREEN}✓ Shared memory activation system tests completed${NC}'; \
    
    echo '${GREEN}✓ Deployment completed successfully${NC}'; \
"

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "${BLUE}📊 To monitor the service:${NC}"
echo -e "${YELLOW}  ssh pi 'cd ~/hey-orac && docker-compose logs -f hey-orac'${NC}"
echo -e "${BLUE}📊 To check container status:${NC}"
echo -e "${YELLOW}  ssh pi 'cd ~/hey-orac && docker-compose ps'${NC}"
echo -e "${BLUE}🔧 To restart the service:${NC}"
echo -e "${YELLOW}  ssh pi 'cd ~/hey-orac && docker-compose restart hey-orac'${NC}"
echo -e "${BLUE}🛑 To stop the service:${NC}"
echo -e "${YELLOW}  ssh pi 'cd ~/hey-orac && docker-compose down'${NC}" 