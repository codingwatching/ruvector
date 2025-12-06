#!/bin/bash
# Build custom RuVector Studio image
# This script clones the Supabase repo, applies our modifications, and builds the Docker image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUVECTOR_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${BUILD_DIR:-/tmp/ruvector-studio-build}"

echo "=== Building RuVector Studio ==="
echo "Build directory: $BUILD_DIR"
echo "RuVector root: $RUVECTOR_ROOT"

# Clean up any previous build
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"

# Clone the Supabase repo
echo "Cloning Supabase repository..."
git clone --depth 1 --branch studio-v1.24.08 https://github.com/supabase/supabase.git .

# Apply our custom modifications
echo "Applying RuVector modifications..."

# Copy RuVectorHome component
mkdir -p apps/studio/components/interfaces/RuVector
cp "$RUVECTOR_ROOT/studio/components/interfaces/RuVector/RuVectorHome.tsx" apps/studio/components/interfaces/RuVector/

# Copy custom pages
mkdir -p apps/studio/pages/project/\[ref\]/vectors
mkdir -p apps/studio/pages/project/\[ref\]/attention
mkdir -p apps/studio/pages/project/\[ref\]/gnn
mkdir -p apps/studio/pages/project/\[ref\]/hyperbolic
mkdir -p apps/studio/pages/project/\[ref\]/learning
mkdir -p apps/studio/pages/project/\[ref\]/routing

# Copy index page
cp "$RUVECTOR_ROOT/studio/pages/project/[ref]/index.tsx" "apps/studio/pages/project/[ref]/index.tsx"

# Copy feature pages
cp "$RUVECTOR_ROOT/studio/pages/project/[ref]/vectors/index.tsx" "apps/studio/pages/project/[ref]/vectors/index.tsx"
cp "$RUVECTOR_ROOT/studio/pages/project/[ref]/attention/index.tsx" "apps/studio/pages/project/[ref]/attention/index.tsx"
cp "$RUVECTOR_ROOT/studio/pages/project/[ref]/gnn/index.tsx" "apps/studio/pages/project/[ref]/gnn/index.tsx"
cp "$RUVECTOR_ROOT/studio/pages/project/[ref]/hyperbolic/index.tsx" "apps/studio/pages/project/[ref]/hyperbolic/index.tsx"
cp "$RUVECTOR_ROOT/studio/pages/project/[ref]/learning/index.tsx" "apps/studio/pages/project/[ref]/learning/index.tsx"
cp "$RUVECTOR_ROOT/studio/pages/project/[ref]/routing/index.tsx" "apps/studio/pages/project/[ref]/routing/index.tsx"

echo "Building studio..."

# Install dependencies
npm install -g pnpm@10
pnpm install --frozen-lockfile

# Build the studio
cd apps/studio
SKIP_ASSET_UPLOAD=1 pnpm build

# Build Docker image
echo "Building Docker image..."
docker build -t ruvector-studio:latest -f - . <<'DOCKERFILE'
FROM node:20-alpine

WORKDIR /app

ENV NODE_ENV=production

COPY .next/standalone ./
COPY .next/static ./.next/static
COPY public ./public

EXPOSE 3000

ENV PORT=3000

CMD ["node", "server.js"]
DOCKERFILE

echo "=== Build complete! ==="
echo "Run with: docker run -p 3001:3000 ruvector-studio:latest"
