#!/bin/bash
# =============================================================================
# Add a sub-service to this agentify container project
# =============================================================================
# Usage: ./scripts/add-service.sh <service-name> [git-repo-url]
#
# Examples:
#   ./scripts/add-service.sh openmemory https://github.com/zimdin12/mem0-ollama-hybrid.git
#   ./scripts/add-service.sh llama-cpp
#   ./scripts/add-service.sh qdrant

set -e

SERVICE_NAME="${1:?Usage: $0 <service-name> [git-repo-url]}"
GIT_REPO="${2:-}"
SERVICE_DIR="services/${SERVICE_NAME}"

if [ -d "${SERVICE_DIR}" ]; then
    echo "Error: Service directory ${SERVICE_DIR} already exists"
    exit 1
fi

echo "Adding sub-service: ${SERVICE_NAME}"

if [ -n "${GIT_REPO}" ]; then
    echo "Adding as git submodule from ${GIT_REPO}..."
    git submodule add "${GIT_REPO}" "${SERVICE_DIR}"
    echo "Submodule added. You may need to configure its docker-compose.yml."
else
    echo "Creating service scaffold..."
    mkdir -p "${SERVICE_DIR}"

    # Copy and customize the template (portable sed)
    if sed --version 2>/dev/null | grep -q GNU; then
        # GNU sed (Linux)
        sed "s/sub-service/${SERVICE_NAME}/g; s/your-image:latest/${SERVICE_NAME}:latest/g" \
            services/docker-compose.sub-service.example.yml > "${SERVICE_DIR}/docker-compose.yml"
    else
        # BSD sed (macOS)
        sed "s/sub-service/${SERVICE_NAME}/g; s/your-image:latest/${SERVICE_NAME}:latest/g" \
            services/docker-compose.sub-service.example.yml > "${SERVICE_DIR}/docker-compose.yml"
    fi

    echo "Created ${SERVICE_DIR}/docker-compose.yml"
    echo "Edit it to configure your sub-service."
fi

echo ""
echo "To run with this sub-service:"
echo "  docker compose -f docker-compose.yml -f ${SERVICE_DIR}/docker-compose.yml up -d"
echo ""
echo "Or add '${SERVICE_NAME}' to SUB_SERVICES in your .env file"
echo "and use: bash scripts/compose-up.sh -d"
