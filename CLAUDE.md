# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Python-based LLM benchmarking tool that compares the performance and quality of various large language model APIs. The tool measures response times and evaluates output quality using BLEU and ROUGE metrics across multiple providers including OpenAI, Azure OpenAI, Anthropic, AWS Bedrock, Hugging Face, and local OpenAI-compatible endpoints.

## Development Commands

### Setup and Installation
```bash
# Create virtual environment (recommended Python 3.12.2)
pyenv install 3.12.2
pyenv virtualenv 3.12.2 llm-benchmarking
pyenv activate llm-benchmarking

# Install dependencies
pip install -r requirements.txt

# Install NLTK data for BLEU score calculation
python -c "import nltk; nltk.download('punkt')"
```

### Running the Application
```bash
# Run the original benchmarking tool
python main.py

# Run the enhanced benchmarking tool with all new features
python enhanced_main.py
```

### Testing
```bash
# Run all unit tests
python -m unittest discover tests/ -v

# Run specific test module
python -m unittest tests.test_enhanced_quality_metrics -v
python -m unittest tests.test_human_evaluation -v
python -m unittest tests.test_ab_testing -v
```

### Environment Configuration
- Copy `.env.sample` to `.env` and configure API keys and endpoints
- Required environment variables are documented in `.env.sample`

## Architecture

### Core Components

**Entry Point (`main.py:16-111`)**
- Interactive CLI that prompts users to select APIs, models, and iteration counts
- Orchestrates benchmark execution and results display
- Handles colorized output and user input parsing

**Configuration (`config.py`)**
- `API_MODELS`: Dictionary mapping API names to available model lists
- `QUERIES_AND_REFERENCES`: List of test queries with reference answers for quality evaluation
- `MAX_TOKENS`: Global token limit for responses (1024)

**Base Architecture (`benchmarks/base.py:5-37`)**
- `BaseBenchmark`: Abstract base class defining the benchmark interface
- Three key abstract methods: `setup_client()`, `invoke_model()`, `extract_output()`
- Common `run()` method that handles timing and error handling

**Benchmark Implementations (`benchmarks/`)**
Each API has its own benchmark class inheriting from `BaseBenchmark`:
- `OpenAIBenchmark`: Standard OpenAI API
- `AzureOpenAIBenchmark`: Azure OpenAI service
- `AnthropicBenchmark`: Anthropic Claude models
- `LocalOpenAIBenchmark`: Local OpenAI-compatible servers (e.g., LM Studio)
- `HuggingFaceBenchmark`: Hugging Face Inference API
- `AWSBedrockBenchmark`: AWS Bedrock models

**Quality Metrics (`quality_metrics.py`)**
- Implements BLEU and ROUGE score calculations
- Compares model outputs against reference answers
- Returns normalized scores for consistent evaluation

**Utilities (`utils.py`)**
- User input handling and validation
- Colorized terminal output functions
- Comma-separated input parsing

### Key Design Patterns

**Plugin Architecture**: Each API provider is implemented as a separate benchmark class, making it easy to add new providers by implementing the three abstract methods.

**Configuration-Driven**: Models and test queries are centralized in `config.py`, allowing easy modification without code changes.

**Quality Assessment**: Uses established NLP metrics (BLEU, ROUGE) with reference answers to provide objective quality measurements alongside performance metrics.

## Adding New API Providers

To add a new benchmark:

1. Create new file in `benchmarks/` directory (e.g., `new_api_benchmark.py`)
2. Implement class inheriting from `BaseBenchmark`
3. Implement the three abstract methods:
   - `setup_client()`: Initialize API client
   - `invoke_model()`: Make API call with query, model, and max_tokens
   - `extract_output()`: Extract text response from API response
4. Add to `benchmarks/__init__.py` imports
5. Update `config.py` with new API models
6. Add to `benchmarks` dictionary in `main.py:21-28`

## Test Queries and Quality Assessment

The tool uses predefined queries in `config.py:12-41` covering diverse topics:
- Factual questions (capital cities)
- Technical explanations (machine learning)
- Analytical tasks (literature analysis)
- Creative tasks (reducing plastic waste)
- Complex reasoning (AI in healthcare, climate change)

Each query has a comprehensive reference answer used for BLEU and ROUGE score calculations, ensuring consistent quality evaluation across all models and providers.

## Enhanced Features (New)

### Enhanced Quality Metrics (`enhanced_quality_metrics.py`)

**PerplexityCalculator**: Calculates perplexity scores using n-gram probability distributions to assess language model fluency.

**SemanticSimilarityCalculator**: Uses sentence transformers to calculate semantic similarity metrics:
- Cosine similarity between embeddings
- Euclidean and Manhattan distance-based similarities
- Semantic textual similarity scores

**TaskSpecificEvaluator**: Evaluates responses for specific tasks:
- Factual accuracy assessment
- Coherence scoring
- Key fact extraction and comparison

**EnhancedQualityMetrics**: Main class combining all enhanced metrics with comprehensive error handling and logging.

### Human Evaluation System (`human_evaluation.py`)

**EvaluationDatabase**: SQLite-based storage for evaluation tasks, ratings, and sessions.

**HumanEvaluator**: Manages human evaluation workflows:
- Creates evaluation tasks from benchmark results
- Tracks evaluation sessions and progress
- Generates comprehensive statistics and reports

**EvaluationInterface**: Interactive CLI for human evaluators to rate responses across multiple criteria (relevance, coherence, accuracy, completeness, clarity, creativity, helpfulness).

### A/B Testing Framework (`ab_testing.py`)

**ABTestManager**: Manages controlled comparison experiments:
- Creates pairwise comparison tasks from benchmark results
- Randomizes response order to avoid bias
- Tracks preference choices and confidence levels

**ABTestDatabase**: Stores experiment data, comparison tasks, and user preferences.

**ABTestInterface**: Interactive interface for conducting blind comparison studies.

### Enhanced Main Application (`enhanced_main.py`)

**EnhancedBenchmarkingTool**: Unified interface combining all features:
- Standard benchmarking with basic metrics
- Enhanced benchmarking with advanced quality metrics
- Human evaluation setup and execution
- A/B testing experiment management
- Comprehensive result export capabilities

## Usage Patterns

### Running Enhanced Benchmarks
```bash
python enhanced_main.py
# Select option 2 for enhanced benchmarks with perplexity, semantic similarity, and task-specific metrics
```

### Setting up Human Evaluation
```bash
python enhanced_main.py
# Select option 3 to create evaluation tasks, then option 4 to run evaluation sessions
```

### Conducting A/B Testing
```bash
python enhanced_main.py
# Select option 5 to create experiment, then option 6 to run comparison sessions
```

### Key Files for Enhancement
- `enhanced_quality_metrics.py`: Advanced quality assessment
- `human_evaluation.py`: Manual evaluation system
- `ab_testing.py`: Controlled comparison experiments
- `enhanced_main.py`: Unified interface
- `tests/`: Comprehensive unit test coverage