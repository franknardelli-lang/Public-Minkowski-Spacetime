# Deployment Guide for Minkowski Spacetime App

## Overview
This guide explains how to deploy the Streamlit application to your Linux server using Docker and Dockge.

## Server Details
- **User**: alan
- **IP**: 192.168.2.90
- **Directory**: /opt/stacks/my-streamlit-app
- **Host Port**: 8504
- **Container Port**: 8501

## One-Time Server Setup

Run these commands **on your server** (SSH into it first):

```bash
# SSH into the server
ssh alan@192.168.2.90

# Create the application directory
sudo mkdir -p /opt/stacks/my-streamlit-app

# Set ownership to alan user
sudo chown -R alan:alan /opt/stacks/my-streamlit-app

# Navigate to the directory
cd /opt/stacks/my-streamlit-app

# Clone your repository (replace with your actual GitHub repo URL)
git clone https://github.com/franknardelli-lang/Public-Minkowski-Spacetime.git .

# Verify files are present
ls -la

# Build and start the container for the first time
docker compose up -d --build

# Check if container is running
docker compose ps
```

## Local Development Setup

1. **Make the deploy script executable**:
   ```bash
   chmod +x deploy.sh
   ```

2. **Set up SSH key authentication** (recommended):
   ```bash
   # Generate SSH key if you don't have one
   ssh-keygen -t ed25519 -C "your_email@example.com"
   
   # Copy SSH key to server
   ssh-copy-id alan@192.168.2.90
   ```

## Deployment Workflow

### Option 1: Using the deploy script (Recommended)
```bash
./deploy.sh
```

This script will:
1. Commit and push your changes to GitHub
2. SSH into the server
3. Pull the latest changes
4. Rebuild and restart the Docker container
5. Show you the container status and logs

### Option 2: Manual deployment
```bash
# 1. Push changes to GitHub
git add .
git commit -m "Your commit message"
git push origin main

# 2. SSH into server and deploy
ssh alan@192.168.2.90
cd /opt/stacks/my-streamlit-app
git pull origin main
docker compose down
docker compose up -d --build
docker compose ps
exit
```

## Accessing the Application

Once deployed, access your application at:
**http://192.168.2.90:8504**

## Useful Commands

### On the server:
```bash
# View logs
cd /opt/stacks/my-streamlit-app
docker compose logs -f

# Restart the container
docker compose restart

# Stop the container
docker compose down

# Rebuild and start
docker compose up -d --build

# Check container status
docker compose ps

# Access container shell
docker compose exec minkowski-spacetime-app bash
```

### Local testing:
```bash
# Test the app locally before deploying
streamlit run app.py

# Test Docker build locally
docker build -t minkowski-test .
docker run -p 8504:8501 minkowski-test
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose logs

# Check if port is in use
sudo netstat -tlnp | grep 8504

# Rebuild from scratch
docker compose down
docker system prune -f
docker compose up -d --build
```

### Git pull fails
```bash
# Check Git status
git status

# Reset local changes (careful!)
git reset --hard origin/main
git pull
```

### Cannot SSH into server
```bash
# Test SSH connection
ssh -v alan@192.168.2.90

# Check if SSH key is loaded
ssh-add -l
```

## File Structure
```
.
├── app.py                 # Main Streamlit application
├── Dockerfile            # Docker image definition
├── compose.yaml          # Docker Compose configuration
├── deploy.sh             # Deployment automation script
├── requirements.txt      # Python dependencies
├── DEPLOYMENT.md         # This file
└── (other app files)
```

## Security Notes

1. Consider using environment variables for sensitive data
2. Set up a firewall to restrict access to port 8504 if needed
3. Use SSH keys instead of passwords for authentication
4. Keep your Docker images updated regularly

## Updating Dependencies

If you add new Python packages:

1. Update [requirements.txt](requirements.txt)
2. Run `./deploy.sh` (it will rebuild with new dependencies)

## Dockge Management

You can also manage this container through the Dockge web interface:
1. Access Dockge on your server (usually port 5001)
2. Find "minkowski-spacetime-app" in the stacks list
3. Use the UI to start/stop/restart/view logs
