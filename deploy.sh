#!/bin/bash

# Enhanced LLM Benchmarking Tool Deployment Script

set -e

echo "🚀 Starting deployment of Enhanced LLM Benchmarking Tool..."

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Run setup script
echo "⚙️ Running setup script..."
python setup.py

# Run health check
echo "🏥 Running health check..."
python health_check.py

echo "✅ Deployment completed successfully!"
echo ""
echo "🎯 To run the tool:"
echo "   source venv/bin/activate"
echo "   python enhanced_main.py"
echo ""
echo "🔍 To run health checks:"
echo "   python health_check.py"
echo ""
echo "🧪 To run tests:"
echo "   python -m unittest discover tests/ -v"