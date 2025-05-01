#!/bin/bash

# Document Q&A Generator Runner Script

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not found."
    exit 1
fi

# Check if requirements are installed
if [ "$1" == "install" ]; then
    echo "Installing requirements..."
    pip3 install -r requirements.txt
    echo "Requirements installed!"
    exit 0
fi

# Function to check if a Python package is installed
check_package() {
    python3 -c "import $1" 2>/dev/null
    return $?
}

# Check if key packages are installed
echo "Checking dependencies..."
MISSING=0

if ! check_package fitz; then
    echo "Missing package: PyMuPDF (fitz)"
    MISSING=1
fi

if ! check_package pdfplumber; then
    echo "Missing package: pdfplumber"
    MISSING=1
fi

if ! check_package transformers; then
    echo "Missing package: transformers"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo "Some dependencies are missing. Run './run.sh install' to install them."
    exit 1
fi

# Check for Tesseract OCR
if ! command -v tesseract &> /dev/null; then
    echo "Warning: Tesseract OCR not found. OCR functionality will be limited."
    echo "Install Tesseract OCR for better results:"
    echo "  - macOS: brew install tesseract"
    echo "  - Ubuntu/Debian: apt-get install tesseract-ocr"
    echo "  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
fi

# Check for Ollama (optional)
OLLAMA_AVAILABLE=0
if command -v ollama &> /dev/null && check_package ollama; then
    echo "Ollama is available! You can use it for better question generation."
    OLLAMA_AVAILABLE=1
else
    echo "Note: Ollama is not available. Using default transformer model."
    echo "To use Ollama:"
    echo "  1. Install Ollama from https://ollama.ai"
    echo "  2. Install Python package: pip install ollama"
    echo "  3. Pull a model: ollama pull llama2"
fi

# Run the appropriate command based on arguments
if [ "$1" == "demo" ]; then
    echo "Running demo script..."
    python3 demo.py
elif [ "$1" == "test" ]; then
    echo "Running tests..."
    python3 test_qa_system.py
else
    echo "Starting Document Q&A web server..."
    echo "Open your browser at http://localhost:5000"
    python3 qa_system.py
fi 