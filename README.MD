# MMM Benchmarking - An MMM & LLM Performance and Quality Benchmarking Tool

## Overview

This project provides a Python-based tool to benchmark the performance and quality of various large language model (LLM) APIs. It sends sample queries to each API, measures response times, calculates average response times, and assesses the quality of responses using the BLEU score metric.

The tool is designed to support API and model assessment for various use cases, considering both speed and output quality. Future versions will support deeper evaluation across multiple metrics and multi-modal model assessments.

### Current Features

- Supports multiple LLM APIs: OpenAI, Azure OpenAI, Anthropic, Custom OpenAI-compatible endpoints (e.g., LM Studio), Hugging Face, and AWS Bedrock.
- Allows users to select specific APIs and models to benchmark through a command-line interface.
- Uses predefined sample queries with reference answers for consistent evaluation.
- Constrains response length to a fixed number of tokens for fair assessment.
- Measures and reports average response times for each API and model combination.
- Calculates BLEU scores to assess the quality of generated responses.
- Provides color-coded output for easy interpretation of performance and quality metrics.

### Project Structure

```
.
├── main.py
├── config.py
├── utils.py
├── quality_metrics.py
└── benchmarks/
    ├── __init__.py
    ├── base.py
    ├── openai_benchmark.py
    ├── azure_openai_benchmark.py
    ├── anthropic_benchmark.py
    ├── local_openai_benchmark.py
    ├── huggingface_benchmark.py
    └── aws_bedrock_benchmark.py
```

- `main.py`: The entry point of the application
- `config.py`: Contains configuration settings like API models, sample queries, and reference answers
- `utils.py`: Utility functions for user input and output formatting
- `quality_metrics.py`: Implements quality assessment metrics (currently BLEU score)
- `benchmarks/`: Directory containing benchmark classes for each API

## Prerequisites

To run the benchmarking tool, you need:

- Python 3.12.2 installed
- Libraries defined in the requirements.txt file
- API credentials and access tokens for the APIs you want to benchmark

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/llm-benchmarking-tool.git
   cd llm-benchmarking-tool
   ```

2. Install the required dependencies (assuming you're using pyenv):
   ```
   pyenv install 3.12.2
   pyenv virtualenv 3.12.2 llm-benchmarking
   pyenv activate llm-benchmarking
   pip install -r requirements.txt
   ```

3. Set up API credentials:
   - Obtain necessary API keys, access tokens, or authentication credentials for each API you want to benchmark.
   - Create a `.env` file in the root directory, using the `.env-sample` file as a template. Add your API keys and endpoints as needed.

4. Install NLTK data for BLEU score calculation:
   ```
   python -c "import nltk; nltk.download('punkt')"
   ```

## Usage

Run the benchmarking tool with:

```
python main.py
```

Follow the prompts to:
1. Select the APIs you want to benchmark
2. Choose specific models for each selected API
3. Set the number of iterations for each benchmark

The tool will run the benchmarks and display results, showing the average response time and BLEU score for each API and model combination.

## Benchmarking Process

1. Load configuration settings and environment variables.
2. Present available APIs and prompt user for selection.
3. For each selected API, show available models and ask user to choose.
4. User specifies the number of iterations for each benchmark.
5. Run benchmarks, sending predefined queries to each selected API and model.
6. Measure response time and calculate BLEU score for each query.
7. Display results, with color-coding based on average response times and BLEU scores.

## Adding New Benchmarks

To add a new benchmark for a different API:

1. Create a new file in the `benchmarks/` directory, e.g., `new_api_benchmark.py`.
2. Define a new class that inherits from `BaseBenchmark` in `benchmarks/base.py`.
3. Implement the `setup_client()`, `invoke_model()`, and `extract_output()` methods for the new API.
4. Add the new benchmark class to `benchmarks/__init__.py`.
5. Update `config.py` to include the new API and its available models.
6. Modify `main.py` to include the new benchmark in the `benchmarks` dictionary.

Example of a new benchmark class:

```python
from .base import BaseBenchmark

class NewAPIBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("New API")

    def setup_client(self):
        # Initialize and return the client for the new API
        pass

    def invoke_model(self, client, query, model, max_tokens):
        # Implement the API call and return the response
        pass

    def extract_output(self, response):
        # Extract and return the generated text from the API response
        pass
```

## Future Developments

# The (very long) list of potential future developments

## LLM Todo List
- [x] Add ability to select different models available on the API endpoints.
- [x] Implement basic quality assessment functionality (BLEU score) to evaluate the responses generated by each API.
- [ ] Enhance quality assessment functionality:
  - [ ] Human Evaluation:
    - [ ] Develop a user interface for manual review and rating of response quality.
    - [ ] Define criteria for assessing response quality (e.g., relevance, coherence, accuracy).
    - [ ] Implement a scoring system for human evaluators to rate responses.
    - [ ] Calculate average quality scores for each API based on human evaluations.
  - [ ] Additional Automated Metrics:
    - [ ] Implement ROUGE metric for assessing the quality of generated summaries or text.
    - [ ] Explore perplexity as a measure of language model performance.
    - [ ] Calculate and compare automated metric scores for each API.
  - [ ] Contextual Embedding Similarity:
    - [ ] Explore techniques for comparing generated responses with reference responses or domain knowledge.
    - [ ] Implement cosine similarity or semantic similarity measures.
    - [ ] Calculate similarity scores between generated responses and reference data.
    - [ ] Evaluate the contextual relevance of responses from each API.
  - [ ] Task-Specific Evaluation:
    - [ ] Design specific tasks or questions with known correct answers or expected outputs.
    - [ ] Implement functions to compare generated responses against expected answers.
    - [ ] Calculate accuracy or other relevant metrics for task-specific evaluation.
    - [ ] Analyse task-specific performance of each API.

## Multimodal Assessment TODO List
- [ ] **Long Context Model Benchmarking
  - [ ] Needle in a haystack test

- [ ] **Image Captioning**
  - [ ] Provide the model with images and ask it to generate descriptive captions or summaries
  - [ ] Evaluate the generated captions for accuracy, relevance, and level of detail
  - [ ] Compare the model's performance with human-generated captions or existing image captioning benchmarks

- [ ] **Visual Question Answering (VQA)**
  - [ ] Present the model with an image and a related question
  - [ ] Assess the model's ability to understand the visual content and provide accurate and relevant answers to the questions
  - [ ] Use established VQA datasets or create custom questions to cover a range of visual reasoning tasks

- [ ] **Image-to-Text Generation**
  - [ ] Give the model an image as input and ask it to generate a coherent and descriptive text based on the visual content
  - [ ] Evaluate the generated text for its quality, coherence, and alignment with the image
  - [ ] Consider factors such as the inclusion of relevant details, the logical flow of the generated text, and the model's ability to capture the main aspects of the image

- [ ] **Text-to-Image Generation**
  - [ ] Provide the model with textual descriptions or prompts and assess its ability to generate corresponding images
  - [ ] Evaluate the generated images for their visual quality, adherence to the textual description, and creativity
  - [ ] Compare the model's performance with existing text-to-image generation models or human-created illustrations

- [ ] **Multimodal Dialogue**
  - [ ] Engage the model in a conversation that involves both text and images
  - [ ] Assess the model's ability to understand and generate responses that integrate information from both modalities
  - [ ] Evaluate the coherence and relevance of the model's responses, as well as its ability to maintain context across multiple turns of the conversation

- [ ] **Multimodal Sentiment Analysis**
  - [ ] Present the model with a combination of text and images that convey a particular sentiment or emotion
  - [ ] Assess the model's ability to accurately identify and classify the sentiment based on the multimodal input
  - [ ] Compare the model's performance with existing sentiment analysis benchmarks that involve both text and images

- [ ] **Cross-Modal Retrieval**
  - [ ] Provide the model with an image and ask it to retrieve relevant textual information or vice versa
  - [ ] Evaluate the model's ability to establish meaningful connections between the visual and textual modalities
  - [ ] Assess the relevance and accuracy of the retrieved information based on the given multimodal query

## Testing TODO List
  - [ ] A/B Testing:
    - [ ] Design controlled experiments for comparing model performance based on user preferences.
    - [ ] Develop a system for presenting responses from different APIs to users without revealing the source.
    - [ ] Collect user preferences, ratings, or other metrics during A/B testing.
    - [ ] Analyse A/B testing results to determine the relative performance of each API.

## Contribution

Contributions are welcome! If you have ideas, suggestions, or bug reports, please open an issue or submit a pull request. Please follow the existing code style and provide appropriate documentation for your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

We'd like to thank the developers and maintainers of the various language models, their APIs, and the tools used in this benchmarking project. This tech continues to blow my mind on a daily basis. 