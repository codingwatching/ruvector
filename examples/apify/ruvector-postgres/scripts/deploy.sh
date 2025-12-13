#!/bin/bash
# Deploy Self-Learning Postgres DB Actor to Apify
# Uses APIFY_API_TOKEN from root .env file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ROOT_DIR="$(cd "$PROJECT_DIR/../../.." && pwd)"

echo "=== Self-Learning Postgres DB - Apify Deployment ==="
echo "Project: $PROJECT_DIR"
echo "Root: $ROOT_DIR"

# Load environment variables from root .env
if [ -f "$ROOT_DIR/.env" ]; then
  echo "Loading environment from $ROOT_DIR/.env"
  set -a
  source "$ROOT_DIR/.env"
  set +a
else
  echo "Warning: .env file not found at $ROOT_DIR/.env"
  echo "Looking for APIFY_API_TOKEN in environment..."
fi

# Check APIFY_API_TOKEN
if [ -z "$APIFY_API_TOKEN" ]; then
  echo "Error: APIFY_API_TOKEN not set"
  echo ""
  echo "Set it in one of these ways:"
  echo "  1. Add APIFY_API_TOKEN=your_token to $ROOT_DIR/.env"
  echo "  2. export APIFY_API_TOKEN=your_token"
  echo "  3. Get your token from https://console.apify.com/account/integrations"
  exit 1
fi

cd "$PROJECT_DIR"

# Install dependencies if needed
if [ ! -d "node_modules" ] || [ "package.json" -nt "node_modules" ]; then
  echo "Installing dependencies..."
  npm install
fi

# Login to Apify
echo ""
echo "Logging in to Apify..."
npx apify login -t "$APIFY_API_TOKEN"

# Push to Apify
echo ""
echo "Deploying actor to Apify..."
npx apify push

echo ""
echo "=== Deployment Complete ==="
echo "View your actor at: https://console.apify.com/actors"
echo "Actor: https://apify.com/ruv/self-learning-postgres-db"
