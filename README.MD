# Multi-Modal Model Benchmarking Project

## Language Model Benchmarking

This project provides a Python script to benchmark the latency and response times of various large language model APIs. It sends sample queries to each API, measures the response times, and calculates the average response time for each API and model combination.

It's intended to support API and model assessment and selection for various time-sensitive use cases.

Future versions will support deeper evaluation across multiple metrics and multi-modal model assessments, per the todo list in this README.

### Current Features

- Supports multiple language model APIs, including OpenAI, Custom OpenAI endpoints (LM Studio etc), Azure OpenAI, Anthropic, Hugging Face, and AWS Bedrock.
- Uses a set of predefined sample queries to ensure a fair assessment across different services.
- Constrains the response length to a fixed number of tokens, to further ensure fair assessment.
- Allows users to select which APIs and models to benchmark through a command-line interface.
- Measures response times for each API and model combination, calculating the average response time across multiple iterations of the same query.
- Provides color-coded output to indicate the performance of each API based on the average response time.

### Project Structure

```
.
├── main.py
├── config.py
├── utils.py
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
- `config.py`: Contains configuration settings like API models and sample queries
- `utils.py`: Utility functions for user input and output formatting
- `benchmarks/`: Directory containing benchmark classes for each API

### The Queries

[The existing query list and reasoning remain unchanged]

### Supported APIs

The following LLM APIs are currently supported:

- OpenAI
- Azure OpenAI 
- Anthropic's Claude
- Custom OpenAI compatible endpoints - useful for LM Studio and other locally or remotely hosted API's
- HuggingFace
- AWS Bedrock

## Prerequisites

To run the benchmarking script, you need to have the following:

- Python 3.12.2 installed
- The libraries defined in the requirements.txt file
- API credentials and access tokens for the APIs you want to benchmark

## Setup

1. Clone the repository:
   ```
   git clone git@github.com:davidmoore-io/mmm_benchmarking.git
   ```

2. Install the required dependencies. This assumes you're using pyenv:
   ```
   pyenv install 3.12.2
   pyenv virtualenv 3.12.2 mmm-benchmarking
   pyenv activate mmm-benchmarking
   pip install -r requirements.txt
   ```

3. Set up the API credentials:
   - For each API you want to benchmark, obtain the necessary API keys, access tokens, or authentication credentials.
   - Create a .env file in the root directory, using the .env-sample file as a template. Add your API keys and endpoints as needed.

## Usage

To run the benchmarking tool, use the following command:
```
python main.py
```

Follow the prompts to:
1. Select the APIs you want to benchmark
2. Choose specific models for each selected API
3. Set the number of iterations for each benchmark

The tool will then run the benchmarks and display the results, showing the average response time for each API and model combination.

## Benchmarking Process

The benchmarking process involves the following steps:

1. Load configuration settings and environment variables.
2. Present a list of available APIs and prompt the user to select which ones to benchmark.
3. For each selected API, show available models and ask the user to choose which ones to test.
4. The user specifies the number of iterations for each benchmark.
5. Run the benchmarks, sending predefined queries to each selected API and model.
6. Measure the response time for each query and calculate the average response time.
7. Display the results, with color-coding based on the average response times.

## Quality Assessment

The current version of the script does not include quality assessment of the generated responses; it currently only benchmarks latency and is intended to be used to assess capacity and speed in the various cloud services vs known use cases.

[The existing quality assessment plans remain unchanged]

## Adding New Benchmarks

To add a new benchmark for a different API:

1. Create a new file in the `benchmarks/` directory, e.g., `new_api_benchmark.py`.
2. Define a new class that inherits from `BaseBenchmark` in `benchmarks/base.py`.
3. Implement the `setup_client()` and `invoke_model()` methods for the new API.
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
```

# The (very long) list of potential future developments

# The (very long) list of potential future developments

## LLM Todo List
- [x] Add ability to select different models available on the API endpoints. 
- [ ] Implement quality assessment functionality to evaluate the responses generated by each API.
  - [ ] Human Evaluation:
    - [ ] Develop a user interface for manual review and rating of response quality.
    - [ ] Define criteria for assessing response quality (e.g., relevance, coherence, accuracy).
    - [ ] Implement a scoring system for human evaluators to rate responses.
    - [ ] Calculate average quality scores for each API based on human evaluations.
  - [ ] Automated Metrics:
    - [ ] Integrate BLEU metric for evaluating response quality against reference responses.
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

Contributions to this project are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. Please try to follow the existing code style and provide appropriate documentation for your changes.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

We'd like to thank the developers and maintainers of the various language models, their APIs, and the tools used in this benchmarking project for their crazy cool contributions to the field of natural language processing. This tech has changed our lives, and the possibilities blow our minds on a daily basis.