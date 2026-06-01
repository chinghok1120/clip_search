#!/bin/bash
# Start CLIP Search Web Interface

set -e

echo "================================================================================"
echo "CLIP Search Web Interface"
echo "================================================================================"

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "Error: Virtual environment not found at ../venv"
    echo "Please create it first:"
    echo "  cd .."
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ../venv/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing web interface dependencies..."
    pip install -r requirements.txt
fi

# Start server
echo ""
echo "Starting server..."
echo ""
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
