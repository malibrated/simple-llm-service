#!/bin/bash
# Setup script for LLM Service

echo "Setting up LLM Service..."

# Check Python version
python3 --version >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if on macOS for MLX
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    read -p "Install MLX support for Apple Silicon? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing MLX..."
        pip install mlx mlx-lm
    fi
fi

# Copy .env.example if .env doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Copying .env.example to .env..."
    cp .env.example .env
    echo "Please edit .env to configure your model paths"
else
    echo ".env file already exists"
fi

# Make scripts executable
chmod +x start_service.sh stop_service.sh setup.sh

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env to configure your model paths"
echo "2. Run ./start_service.sh to start the service"
echo ""
echo "To activate the virtual environment manually:"
echo "  source .venv/bin/activate"