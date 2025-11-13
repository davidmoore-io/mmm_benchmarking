#!/usr/bin/env python3
"""
Setup script for enhanced LLM benchmarking tool.
"""

import os
import sys
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")

def install_requirements():
    """Install required packages."""
    try:
        logger.info("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        sys.exit(1)

def download_nltk_data():
    """Download required NLTK data using our safe import."""
    try:
        logger.info("Setting up NLTK data...")
        from utils import safe_nltk_import
        
        if safe_nltk_import():
            logger.info("NLTK data setup completed successfully")
        else:
            logger.warning("NLTK data setup failed, but continuing (functionality may be limited)")
    except Exception as e:
        logger.error(f"Failed to setup NLTK data: {e}")
        # Don't exit - continue with setup as NLTK is optional

def create_directories():
    """Create necessary directories."""
    directories = ['results', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def check_env_file():
    """Check if .env file exists."""
    if not os.path.exists('.env'):
        if os.path.exists('.env.sample'):
            logger.warning("No .env file found. Please copy .env.sample to .env and configure your API keys.")
        else:
            logger.error("No .env.sample file found")
    else:
        logger.info(".env file exists")

def run_basic_tests():
    """Run basic import tests."""
    try:
        logger.info("Running basic import tests...")
        
        # Test basic imports
        from enhanced_quality_metrics import EnhancedQualityMetrics
        from human_evaluation import HumanEvaluator
        from ab_testing import ABTestManager
        
        logger.info("All imports successful")
        
        # Test basic functionality
        metrics = EnhancedQualityMetrics()
        evaluator = HumanEvaluator()
        ab_manager = ABTestManager()
        
        logger.info("Basic functionality tests passed")
        
    except Exception as e:
        logger.error(f"Import or functionality test failed: {e}")
        sys.exit(1)

def main():
    """Main setup function."""
    logger.info("Starting enhanced LLM benchmarking tool setup...")
    
    check_python_version()
    install_requirements()
    download_nltk_data()
    create_directories()
    check_env_file()
    run_basic_tests()
    
    logger.info("Setup completed successfully!")
    logger.info("You can now run the tool with: python enhanced_main.py")

if __name__ == "__main__":
    main()