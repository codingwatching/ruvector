# RuVector PostgreSQL Apify Actor
# Uses pre-built ruvnet/ruvector-postgres image

FROM ruvnet/ruvector-postgres:latest

# Install Node.js 22
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install --omit=dev

# Copy source code
COPY . ./

# Copy startup script
COPY scripts/start-with-postgres.sh ./start.sh
RUN chmod +x ./start.sh

CMD ["./start.sh"]
