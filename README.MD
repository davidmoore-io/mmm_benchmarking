# Multi-Modal Model Benchmarking Project

## Language Model Benchmarking

This project provides a Python script to benchmark the latency and response times of various large language model APIs. It sends sample queries to each API, measures the response times, and calculates the average response time for each API.

It's intended to support API and model assessment and selection for various time sensitive use cases.

Future versions will support deeper evaluation across mutliple metrics and multi-modal model assessments, per the todo list in this README.

### Current Features
git pus
- Supports multiple language model APIs, including OpenAI, Custom OpenAI endpoints (LM Studio etc), Azure OpenAI, Anthropic, Hugging Face, and AWS Bedrock.
- Uses a set of predefined sample queries to ensure a fair assessment across different services.
- Constrains the response length to a fixed number of tokens, to further ensure fair assessment.
- Allows users to select which APIs to benchmark through a command-line interface.
- Measures response times for each API and calculates the average response time across multiple iterations of the same query.
- Provides RAG'd (Red Amber Green) assessment to indicate the performance of each API based on the average response time.

### The Queries

- "What is the capital of France?"
  - Reasoning:
    - A simple factual question that tests the model's ability to provide accurate information from its knowledge base.
    - It assesses the model's capability to handle straightforward questions and retrieve specific facts.

- "Explain the concept of machine learning in simple terms."
  - Reasoning:
    - This evaluates the model's ability to provide a clear and concise explanation of a technical concept.
    - It tests the model's capacity to break down complex topics into easily understandable language, catering to a non-technical audience.

- "What are the main differences between renewable and non-renewable energy sources?"
  - Reasoning:
    - This query assesses the model's ability to compare and contrast two related concepts.
    - It requires the model to identify key distinguishing factors and present them in a structured manner, demonstrating its understanding of the subject matter.

- "Suggest five creative ways to reduce plastic waste in everyday life."
  - Reasoning:
    - This query challenges the model's creativity and problem-solving skills.
    - It tests the model's ability to generate multiple unique ideas and provide practical solutions to a given problem.
    - It evaluates the model's capacity for original thought and its awareness of environmental issues.

- "Analyze the main themes and symbolism in the novel 'To Kill a Mockingbird' by Harper Lee."
  - Reasoning:
    - This query assesses the model's ability to perform literary analysis and interpretation.
    - It tests the model's understanding of the deeper meanings, themes, and symbolic elements within a well-known literary work.
    - It evaluates the model's capacity to provide insights and draw connections between different aspects of the novel.

- "What are the potential benefits and drawbacks of artificial intelligence in healthcare?"
  - Reasoning:
    - This query evaluates the model's ability to discuss the implications and considerations surrounding the application of AI in a specific domain.
    - It tests the model's capacity to present a balanced perspective, considering both the positive and negative aspects of the topic.
    - It assesses the model's understanding of the ethical and practical implications of AI in healthcare.

- "How can individuals and communities contribute to reducing the impact of climate change?"
  - Reasoning:
    - This query assesses the model's ability to provide actionable advice and recommendations on a global issue.
    - It tests the model's understanding of climate change and its capacity to suggest practical steps that individuals and communities can take to mitigate its effects.
    - It evaluates the model's awareness of environmental sustainability and its ability to provide guidance on making a positive impact.

### Supported APIs

The following LLM APIs are currently supported:

- Anthropic's Claude
- OpenAI
- Azure OpenAI 
- Custom OpenAI compatible endpoints - useful for LM Studio and other locally or remotely hosted API's
- HuggingFace - currently untested
- AWS Bedrock - currently untested

Model ID's for each endpoint are currently hard coded, aligned to the newest version available.
## Prerequisites

To run the benchmarking script, you need to have the following:

- Python 3.12.2 installed
- The libraries defined in requirements.txt file.
- API credentials and access tokens for the APIs you want to benchmark

## Setup

1. Clone the repository:
   ```
   git clone REPO URL
   ```

2. Install the required dependencies. This assumes you're using pyenv"
   ```
   pyenv install 3.12.2
   pyenv virtualenv-3.12.2 mmm-benchmarking
   pyenv activate mmm-benchmarking
   pip install -r requirements.txt
   ```

3. Set up the API credentials:
   - For each API you want to benchmark, obtain the necessary API keys, access tokens, or authentication credentials.
   - Create a .env file, using the .env-sample file as a template. Entering `NONE` next to an API key will skip testing that service.

## Usage

To run the benchmarking script, use the following command:
```
python benchmark.py
```

The script will iterate over the defined APIs, send sample queries, restraining the token length of repsones to ensure a like-for-like comparison between services, and measure the response times. It will output the average response time for each API.

## Benchmarking Process

The benchmarking process involves the following steps:

1. Define the sample queries to be sent to each API.
2. Set the maximum number of tokens for the generated responses to ensure a fair comparison.
3. Iterate over each API and perform the following:
   - Send each query to the API and measure the response time.
   - Accumulate the total response time for each API.
4. Calculate and print the average response time for each API.

## Quality Assessment

The current version of the script does not include quality assessment of the generated responses, it currently only benchmarks latency, and is intended to be used to assess capacity and speed in the various cloud services vs known use cases.

 However, there are plans to incorporate various other evaluation approaches in future updates. Some potential methods for assessing response quality include:

- Human evaluation
- Automated metrics (e.g., BLEU, ROUGE, perplexity)
- Task-specific evaluation
- Contextual embedding similarity
- A/B testing

# The (very long) list of potential future developments. 

If you've got some time on your hands, feel free to get involved in the below:

## LLM Todo List
- [ ] Add ability to select different models available on the API endpoints. It's currently hardcoded to the most recent models.
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

Contributions to this project are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request. Please try follow the existing code style and provide appropriate documentation for your changes.

## Licence

This project is licensed under the [MIT Licence](LICENSE).

## Acknowledgements

We'd like to thank the developers and maintainers of the various language models, their APIs, and the tools used in this benchmarking project for their crazy cool contributions to the field of natural language processing. This tech has changed our lives, and the possibilities blow our minds on a daily basis. 