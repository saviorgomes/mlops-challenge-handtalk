#!/bin/sh
set -e

# Import workflow
n8n import:workflow --input=/workspace/n8n/workflow.json || true

# Activate workflow before starting n8n
n8n publish:workflow --id=f2a3b4c5-d6e7-8901-fabc-012345678901 || true

# Start n8n
exec n8n start