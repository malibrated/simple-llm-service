#!/bin/bash
# Start the LLM Service

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found!"
    echo "Please copy .env.example to .env and configure your models."
    exit 1
fi

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "No virtual environment found. Using system Python."
fi

# Check if requirements are installed
python -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Dependencies not installed. Installing..."
    pip install -r requirements.txt
fi

# Get port from .env file, default to 8000 if not found
PORT=$(grep "^PORT=" .env 2>/dev/null | cut -d'=' -f2 || echo "8000")

# Start the service
echo "Starting LLM Service..."
echo "Service will be available at http://localhost:${PORT}"
echo "API docs available at http://localhost:${PORT}/docs"
echo ""
echo "Port discovery: The service writes its port to '.port' file"
echo ""
echo "Press Ctrl+C to stop the service"
echo ""

# Run the service
# Try running as a module first, fallback to direct execution
python -m server 2>/dev/null || python server.py