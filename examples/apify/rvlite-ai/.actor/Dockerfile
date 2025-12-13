FROM node:22-slim

# Install build dependencies for native modules
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    python3 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./

RUN npm install --omit=dev

COPY . ./

CMD ["node", "src/main.js"]
