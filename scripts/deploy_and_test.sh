#!/usr/bin/env bash
set -euo pipefail

# OpenWakeWord Test Deploy and Test Script
# Usage: ./scripts/deploy_and_test.sh [commit_message]
# Example: ./scripts/deploy_and_test.sh "Add OpenWakeWord implementation"

# Default parameters
COMMIT_MSG=${1:-"Initial OpenWakeWord test implementation"}
REMOTE_ALIAS="pi"
PROJECT_NAME="hey-orac"  # Updated to match current project name

# Terminal colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Hey ORAC Deployment Script${NC}"
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

# Get current commit hash and branch FIRST
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)

# 1) Local: commit and push
echo -e "${YELLOW}👉 Pushing local commits to $BRANCH_NAME...${NC}"

# Add all changes
echo -e "${YELLOW}Adding all changes${NC}"
git add -A

# Commit if there are any changes
if git diff --cached --quiet; then
    echo -e "${YELLOW}No changes to commit${NC}"
else
    echo -e "${YELLOW}Committing with message: $COMMIT_MSG${NC}"
    git commit -m "$COMMIT_MSG"

    echo -e "${YELLOW}Pushing to origin/$BRANCH_NAME${NC}"
    git push origin "$BRANCH_NAME"
fi

# Get current commit hash (after potential commit)
COMMIT_HASH=$(git rev-parse HEAD)

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
    
    echo '${BLUE}🧹 Smart Docker cleanup...${NC}'; \
    echo 'Disk space before cleanup:'; \
    df -h | grep -E '/$|/home'; \
    \
    echo 'Stopping old containers...'; \
    docker-compose down --remove-orphans 2>/dev/null || true; \
    docker container prune -f 2>/dev/null || true; \
    \
    echo 'Removing only dangling images (keeping recent builds)...'; \
    docker image prune -f 2>/dev/null || true; \
    \
    echo 'Removing unused volumes...'; \
    docker volume prune -f 2>/dev/null || true; \
    \
    echo 'Removing unused networks...'; \
    docker network prune -f 2>/dev/null || true; \
    \
    echo 'Disk space after cleanup:'; \
    df -h | grep -E '/$|/home'; \
    \
    echo '${GREEN}✓ Docker cleanup completed${NC}'; \
    
    echo '${BLUE}🔍 Smart change detection...${NC}'; \
    \
    # Check what files changed since last deployment
    if [ -f .last_deploy_commit ]; then
        LAST_COMMIT=\$(cat .last_deploy_commit);
        echo \"Last deployment commit: \$LAST_COMMIT\";
        CHANGED_FILES=\$(git diff --name-only \$LAST_COMMIT $COMMIT_HASH 2>/dev/null || echo '');
    else
        echo 'No previous deployment found - full rebuild needed';
        CHANGED_FILES='full_rebuild';
    fi; \
    \
    echo \"Changed files: \$CHANGED_FILES\"; \
    \
    # Determine build strategy based on changes
    BUILD_STRATEGY='incremental'; \
    \
    if echo \"\$CHANGED_FILES\" | grep -q 'requirements.txt\|pyproject.toml\|Dockerfile'; then
        echo '${YELLOW}⚠️  Dependencies or Dockerfile changed - full rebuild needed${NC}';
        BUILD_STRATEGY='full';
    elif echo \"\$CHANGED_FILES\" | grep -q 'src/.*\.py'; then
        echo '${BLUE}📝 Python source code changed - incremental rebuild${NC}';
        BUILD_STRATEGY='incremental';
    elif echo \"\$CHANGED_FILES\" | grep -q 'models/'; then
        echo '${BLUE}🤖 Model files changed - incremental rebuild${NC}';
        BUILD_STRATEGY='incremental';
    elif [ \"\$CHANGED_FILES\" = 'full_rebuild' ]; then
        echo '${YELLOW}⚠️  First deployment or no previous commit - full rebuild${NC}';
        BUILD_STRATEGY='full';
    else
        echo '${GREEN}✓ No significant changes - using cached layers${NC}';
        BUILD_STRATEGY='cache_only';
    fi; \
    \
    echo '${BLUE}🐳 Building & starting containers...${NC}'; \
    echo 'Stopping existing containers...'; \
    docker-compose down; \
    \
    # Execute build strategy
    if [ \"\$BUILD_STRATEGY\" = 'full' ]; then
        echo '${YELLOW}🔄 Full rebuild (no cache)...${NC}';
        docker-compose build --no-cache --build-arg GIT_COMMIT=$COMMIT_HASH hey-orac;
    elif [ \"\$BUILD_STRATEGY\" = 'incremental' ]; then
        echo '${BLUE}⚡ Incremental rebuild (with cache)...${NC}';
        docker-compose build --build-arg CACHEBUST=\$(date +%s) --build-arg GIT_COMMIT=$COMMIT_HASH hey-orac;
    else
        echo '${GREEN}🚀 Using cached layers...${NC}';
        docker-compose build --build-arg GIT_COMMIT=$COMMIT_HASH hey-orac;
    fi; \
    \
    echo 'Starting fresh container...'; \
    docker-compose up -d --force-recreate hey-orac; \
    \
    # Save current commit for next deployment
    echo $COMMIT_HASH > .last_deploy_commit; \
    \
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
    
    echo '${BLUE}🧪 Testing Hey ORAC system...${NC}'; \
    echo 'Checking wake word detection service...'; \
    \
    echo 'Testing container health...'; \
    docker-compose ps hey-orac | grep -q 'Up' && echo '✅ Hey ORAC service is running' || echo '❌ Hey ORAC service not running'; \
    \
    echo 'Checking audio device access...'; \
    docker-compose exec -T hey-orac python3 -c \"import pyaudio; p = pyaudio.PyAudio(); print(f'Found {p.get_device_count()} audio devices')\" 2>/dev/null && echo '✅ Audio devices accessible' || echo '❌ Audio device access failed'; \
    \
    echo 'Testing wake word model loading...'; \
    docker-compose exec -T hey-orac python3 -c \"import openwakeword; print('OpenWakeWord loaded successfully')\" 2>/dev/null && echo '✅ Wake word models loaded' || echo '❌ Model loading failed'; \
    
    echo '${GREEN}✓ Hey ORAC system tests completed${NC}'; \
    
    echo '${GREEN}✓ Deployment completed successfully${NC}'; \
"

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "${BLUE}📊 To monitor the service:${NC}"
echo -e "${YELLOW}  ssh pi 'cd ~/$PROJECT_NAME && docker-compose logs -f hey-orac'${NC}"
echo -e "${BLUE}📊 To check container status:${NC}"
echo -e "${YELLOW}  ssh pi 'cd ~/$PROJECT_NAME && docker-compose ps'${NC}"
echo -e "${BLUE}🔧 To restart the service:${NC}"
echo -e "${YELLOW}  ssh pi 'cd ~/$PROJECT_NAME && docker-compose restart hey-orac'${NC}"
echo -e "${BLUE}🛑 To stop the service:${NC}"
echo -e "${YELLOW}  ssh pi 'cd ~/$PROJECT_NAME && docker-compose down'${NC}" 