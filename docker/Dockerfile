# RuVector - Complete AI Vector Database Platform
# Multi-stage build with PostgreSQL extension + npm packages
#
# Usage:
#   docker build -t ruvnet/ruvector:latest -f docker/Dockerfile .
#   docker run -d -p 5432:5432 ruvnet/ruvector:latest
#
# Build Arguments:
#   --build-arg PG_VERSION=17          PostgreSQL version (14, 15, 16, 17)
#   --build-arg INCLUDE_NPM=true       Include Node.js and npm packages
#   --build-arg INCLUDE_SONA=true      Include SONA self-learning module
#   --build-arg INCLUDE_RUVLLM=true    Include ruvLLM integration
#   --build-arg INCLUDE_CLI=true       Include postgres-cli tool

ARG PG_VERSION=17
ARG INCLUDE_NPM=true
ARG INCLUDE_SONA=true
ARG INCLUDE_RUVLLM=true
ARG INCLUDE_CLI=true

# ============================================================================
# Stage 1: PostgreSQL Extension Builder
# ============================================================================
FROM rust:1.83-bookworm AS rust-builder

ARG PG_VERSION

# Add PostgreSQL APT repository
RUN sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt bookworm-pgdg main" > /etc/apt/sources.list.d/pgdg.list' && \
    wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -

# Install build dependencies
RUN apt-get update && apt-get install -y \
    postgresql-${PG_VERSION} \
    postgresql-server-dev-${PG_VERSION} \
    libclang-dev \
    clang \
    pkg-config \
    libssl-dev \
    cmake \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install pgrx
RUN cargo install cargo-pgrx --version 0.12.6 --locked

# Initialize pgrx
RUN cargo pgrx init --pg${PG_VERSION} /usr/lib/postgresql/${PG_VERSION}/bin/pg_config

ENV PGRX_PG_CONFIG_PATH=/usr/lib/postgresql/${PG_VERSION}/bin/pg_config
ENV PGRX_HOME=/root/.pgrx

# Build postgres crate in ISOLATION (avoids workspace dependency issues)
WORKDIR /build/ruvector-postgres

# Copy ONLY the postgres crate (standalone, no workspace)
COPY crates/ruvector-postgres/Cargo.toml ./
COPY crates/ruvector-postgres/build.rs ./
COPY crates/ruvector-postgres/ruvector.control ./
COPY crates/ruvector-postgres/src ./src/
COPY crates/ruvector-postgres/sql ./sql/
COPY crates/ruvector-postgres/benches ./benches/

# Build the extension
RUN cargo pgrx package \
    --pg-config /usr/lib/postgresql/${PG_VERSION}/bin/pg_config \
    --features pg${PG_VERSION}

# Copy SQL file to extension directory
RUN cp sql/ruvector--0.1.0.sql target/release/ruvector-pg${PG_VERSION}/usr/share/postgresql/${PG_VERSION}/extension/

# ============================================================================
# Stage 2: Runtime Image
# ============================================================================
FROM postgres:${PG_VERSION}-bookworm AS runtime

ARG PG_VERSION
ARG INCLUDE_NPM
ARG INCLUDE_SONA
ARG INCLUDE_RUVLLM
ARG INCLUDE_CLI

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    curl \
    jq \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS (conditional on INCLUDE_NPM)
RUN if [ "$INCLUDE_NPM" = "true" ]; then \
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
        apt-get install -y nodejs && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Copy PostgreSQL extension from builder
COPY --from=rust-builder /build/ruvector-postgres/target/release/ruvector-pg${PG_VERSION}/usr/share/postgresql/${PG_VERSION}/extension/* /usr/share/postgresql/${PG_VERSION}/extension/
COPY --from=rust-builder /build/ruvector-postgres/target/release/ruvector-pg${PG_VERSION}/usr/lib/postgresql/${PG_VERSION}/lib/* /usr/lib/postgresql/${PG_VERSION}/lib/

# Install RuVector npm packages based on build args
RUN if [ "$INCLUDE_NPM" = "true" ]; then \
        # Always install core ruvector package
        npm install -g ruvector 2>/dev/null || true; \
        # Conditionally install SONA
        if [ "$INCLUDE_SONA" = "true" ]; then \
            npm install -g @ruvector/sona 2>/dev/null || true; \
        fi; \
        # Conditionally install ruvLLM
        if [ "$INCLUDE_RUVLLM" = "true" ]; then \
            npm install -g ruvllm 2>/dev/null || true; \
        fi; \
        # Conditionally install postgres-cli
        if [ "$INCLUDE_CLI" = "true" ]; then \
            npm install -g @ruvector/postgres-cli 2>/dev/null || true; \
        fi; \
    fi

# Create initialization script
COPY --chmod=644 docker/init.sql /docker-entrypoint-initdb.d/00-init-ruvector.sql

# Create helpful scripts
RUN cat > /usr/local/bin/ruvector-info << 'EOF'
#!/bin/bash
echo "=========================================="
echo "  RuVector AI Vector Database Platform"
echo "=========================================="
echo ""
echo "PostgreSQL Extension:"
psql -U $POSTGRES_USER -d $POSTGRES_DB -c "SELECT ruvector_version();" 2>/dev/null || echo "  Connect to PostgreSQL first"
echo ""
if command -v npm &> /dev/null; then
    echo "Available npm packages:"
    npm list -g --depth=0 2>/dev/null | grep -E "(ruvector|ruvllm|sona)" || echo "  (none installed)"
else
    echo "npm: Not installed (INCLUDE_NPM=false)"
fi
echo ""
echo "Quick Start:"
echo "  psql -U ruvector -d ruvector_db"
echo "  CREATE EXTENSION ruvector;"
echo ""
echo "Documentation: https://github.com/ruvnet/ruvector"
EOF
RUN chmod +x /usr/local/bin/ruvector-info

# Set build metadata as environment variables (for runtime introspection)
ENV RUVECTOR_INCLUDE_NPM=${INCLUDE_NPM}
ENV RUVECTOR_INCLUDE_SONA=${INCLUDE_SONA}
ENV RUVECTOR_INCLUDE_RUVLLM=${INCLUDE_RUVLLM}
ENV RUVECTOR_INCLUDE_CLI=${INCLUDE_CLI}

# Environment variables
ENV POSTGRES_USER=ruvector
ENV POSTGRES_PASSWORD=ruvector
ENV POSTGRES_DB=ruvector_db

# Expose PostgreSQL port
EXPOSE 5432

# Health check
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD pg_isready -U $POSTGRES_USER -d $POSTGRES_DB || exit 1

# Labels
LABEL org.opencontainers.image.title="RuVector"
LABEL org.opencontainers.image.description="Complete AI Vector Database Platform - Vector Search, GNN, Attention, Hyperbolic, Routing, and PostgreSQL"
LABEL org.opencontainers.image.version="0.2.5"
LABEL org.opencontainers.image.vendor="ruv.io"
LABEL org.opencontainers.image.source="https://github.com/ruvnet/ruvector"
LABEL org.opencontainers.image.documentation="https://github.com/ruvnet/ruvector#readme"
LABEL org.opencontainers.image.licenses="MIT"
