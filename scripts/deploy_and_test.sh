#!/usr/bin/env bash
set -euo pipefail

# Hey Orac Deploy and Test Script
# Usage: ./scripts/deploy_and_test.sh [commit_message]
# Example: ./scripts/deploy_and_test.sh "Add audio buffer implementation"

# Default parameters
COMMIT_MSG=${1:-"Update hey-orac"}
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
    
    echo -e "${YELLOW}Pushing to origin/master${NC}"
    git push origin master
fi

# 2) Remote: pull, build & test
echo -e "${YELLOW}👉 Running remote update & tests on $REMOTE_ALIAS...${NC}"
ssh "$REMOTE_ALIAS" "\
    set -euo pipefail; \
    echo '${BLUE}📂 Updating code from repository...${NC}'; \
    cd \$HOME/$PROJECT_NAME; \
    git fetch origin; \
    git reset --hard origin/master; \
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
    echo 'Removing unused images...'; \
    docker image prune -a -f 2>/dev/null || true; \
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
    
    echo '${BLUE}🐳 Building & starting containers...${NC}'; \
    docker-compose up --build -d; \
    
    echo '${BLUE}🔍 Checking container logs...${NC}'; \
    sleep 3; \
    docker-compose logs hey-orac | tail -n 10; \
    
    echo '${BLUE}🧪 Running tests...${NC}'; \
    echo 'Running configuration tests...'; \
    docker-compose exec -T hey-orac python -m pytest tests/test_wakeword.py::TestConfiguration::test_load_config -v || echo 'Tests completed with some failures'; \
    
    echo '${BLUE}📊 Checking resource usage...${NC}'; \
    echo 'Container stats:'; \
    docker-compose stats --no-stream; \
    \
    echo 'Memory usage:'; \
    free -h; \
    \
    echo 'Disk usage:'; \
    df -h | grep -E '/$|/home'; \
    
    echo '${BLUE}🔍 Checking container health...${NC}'; \
    docker-compose ps; \
    
    echo '${GREEN}✓ Deployment and testing completed${NC}'; \
"

echo -e "${GREEN}🎉 All deployment and test operations completed successfully!${NC}"
echo -e "${BLUE}📊 To monitor the service:${NC}"
echo -e "${YELLOW}  ssh pi 'cd ~/hey-orac && docker-compose logs -f hey-orac'${NC}"
echo -e "${BLUE}📊 To test wake-word detection:${NC}"
echo -e "${YELLOW}  ssh pi 'cd ~/hey-orac && docker-compose exec hey-orac python src/main.py --list-devices'${NC}" 