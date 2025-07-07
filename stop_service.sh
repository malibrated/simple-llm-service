#!/bin/bash
# Stop the LLM Service

echo "Stopping LLM Service..."

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Try to get port from .port file first
if [ -f ".port" ]; then
    PORT=$(cat .port)
    echo "Found service port from .port file: $PORT"
elif [ -f ".env" ]; then
    # Fallback to reading from .env file
    PORT=$(grep "^PORT=" .env 2>/dev/null | cut -d'=' -f2 || echo "8000")
    echo "Using port from .env file: $PORT"
else
    # Final fallback to default
    PORT=${PORT:-8000}
    echo "Using default port: $PORT"
fi

# Find process running on the port
PID=$(lsof -ti :$PORT)

if [ -z "$PID" ]; then
    echo "No service found running on port $PORT"
    
    # Clean up .port file if it exists
    if [ -f ".port" ]; then
        echo "Removing stale .port file"
        rm -f .port
    fi
else
    echo "Found service with PID $PID"
    kill -TERM $PID
    echo "Service stopped"
    
    # The service should remove .port file on shutdown, but ensure it's gone
    if [ -f ".port" ]; then
        rm -f .port
    fi
fi