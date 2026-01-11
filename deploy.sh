#!/bin/bash

# Deploy Script for Minkowski Spacetime Streamlit App
# This script pushes changes to GitHub and deploys to the server

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Server configuration
SERVER_USER="alan"
SERVER_IP="192.168.1.162"
SERVER_PATH="/opt/stacks/my-streamlit-app"

echo -e "${BLUE}=== Minkowski Spacetime Deployment ===${NC}"

# Step 1: Push to GitHub
echo -e "\n${GREEN}[1/4] Pushing changes to GitHub...${NC}"
git add .
echo "Enter commit message (or press Enter for default):"
read COMMIT_MSG
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Update: $(date '+%Y-%m-%d %H:%M:%S')"
fi
git commit -m "$COMMIT_MSG" || echo "No changes to commit"
git push origin main

# Step 2: SSH into server and deploy
echo -e "\n${GREEN}[2/4] Connecting to server...${NC}"
ssh ${SERVER_USER}@${SERVER_IP} << 'ENDSSH'
    set -e
    
    # Navigate to application directory
    cd /opt/stacks/my-streamlit-app
    
    echo "Pulling latest changes..."
    git pull origin main
    
    echo "Building and starting container..."
    docker compose down
    docker compose up -d --build
    
    echo "Deployment complete!"
    
    # Show container status
    echo -e "\nContainer status:"
    docker compose ps
ENDSSH

# Step 3: Verify deployment
echo -e "\n${GREEN}[3/4] Verifying deployment...${NC}"
sleep 3
echo "Checking if application is accessible..."
if curl -s http://${SERVER_IP}:8504 > /dev/null; then
    echo -e "${GREEN}✓ Application is running at http://${SERVER_IP}:8504${NC}"
else
    echo -e "${RED}⚠ Application might still be starting up. Check manually at http://${SERVER_IP}:8504${NC}"
fi

# Step 4: Show logs
echo -e "\n${GREEN}[4/4] Recent container logs:${NC}"
ssh ${SERVER_USER}@${SERVER_IP} "cd ${SERVER_PATH} && docker compose logs --tail=20"

echo -e "\n${BLUE}=== Deployment Complete ===${NC}"
echo -e "Access your app at: ${GREEN}http://${SERVER_IP}:8504${NC}"
