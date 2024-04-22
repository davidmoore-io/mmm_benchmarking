"""
Language Model Benchmarking Project

This script benchmarks the latency and response times of various large language model APIs.
It sends sample queries to each API, measures the response times, and calculates the average
response time for each API.

To ensure a fair assessment across different services, the script:
- Uses the same set of sample queries for all APIs.
- Sets a consistent maximum number of tokens for the generated responses.
- Iterates over each API and measures the response time under similar conditions.
- Calculates the average response time across multiple iterations to account for variability.

Note: This script currently focuses on benchmarking response times. Quality assessment of the
generated responses is planned for future updates.
"""

import time
import os
from dotenv import load_dotenv
import anthropic
import openai
from openai import AzureOpenAI
from openai import OpenAI
import requests
import boto3
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

# Load environment variables from .env file
load_dotenv()
print("Loaded environment variables:")
for key, value in os.environ.items():
    print(f"{key}: {value}")

# Set the maximum number of tokens for the response
MAX_TOKENS = 1024

# Benchmarking function for Anthropic's API
def benchmark_anthropic(query):
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    print(f"Anthropic API Key: {anthropic_api_key}")  # Print the API key for debugging
    
    print(Fore.CYAN + "Testing Anthropic API..." + Style.RESET_ALL)
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    start_time = time.time()
    response = client.messages.create(
        model="claude-3-opus-20240229",  # Use an appropriate model identifier
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": query}]
    )
    end_time = time.time()
    return end_time - start_time

# Benchmarking function for OpenAI's API 
def benchmark_openai(query):
    print(f"OpenAI API Key: {os.getenv("OPENAI_API_KEY")}")  # Print the API key for debugging
    
    print(Fore.CYAN + "Testing OpenAI API..." + Style.RESET_ALL)
    start_time = time.time()

    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY")
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Replace with the desired model ID
        messages=[{"role": "user", "content": query}],
        max_tokens=MAX_TOKENS,
        n=1,
        stop=None,
        temperature=0.7,
    )
    end_time = time.time()
    print(response)  # Access the response variable
    return end_time - start_time

# Benchmarking function for Azure Open AI API endpoints
def benchmark_azure_openai(query):
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    print(f"Azure OpenAI API Key: {azure_openai_api_key}")  # Print the API key for debugging

    print(Fore.CYAN + "Testing Azure Open AI API..." + Style.RESET_ALL)
    start_time = time.time()

    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version = "2024-02-01",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    response = client.chat.completions.create(
        model="gpt-4",  # Replace with the desired model ID
        messages=[{"role": "user", "content": query}],
        max_tokens=MAX_TOKENS,
        n=1,
        stop=None,
        temperature=0.7,
    )
    print(response)  # Access the response variable
    end_time = time.time()
    return end_time - start_time

#Benchmarking function for Local OpenAI Compatible API endpoints
def benchmark_local_openai(query):
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    # client = OpenAI(base_url=os.getenv("LOCAL_OPENAI_BASE_URL"), api_key=os.getenv("LOCAL_OPENAI_API_KEY"))
    print(Fore.CYAN + "Testing Local OpenAI API..." + Style.RESET_ALL)
    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-4",  # Replace with the desired model ID
        messages=[{"role": "user", "content": query}],
        max_tokens=MAX_TOKENS,
        n=1,
        stop=None,
        temperature=0.7,
    )
    end_time = time.time()
    print(response)  # Access the response variable
    return end_time - start_time

# Benchmarking function for Hugging Face's API
def benchmark_huggingface(query):
    huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
    print(f"Hugging Face API Token: {huggingface_api_token}")  # Print the API token for debugging
    
    if not huggingface_api_token or huggingface_api_token.strip() == "":
        print(Fore.YELLOW + "Skipping Hugging Face API benchmarking. API token not provided or invalid." + Style.RESET_ALL)
        return None
    
    print(Fore.CYAN + "Testing Hugging Face API..." + Style.RESET_ALL)
    start_time = time.time()
    response = requests.post(
        "https://api-inference.huggingface.co/models/gpt2",
        headers={"Authorization": f"Bearer {huggingface_api_token}"},
        json={"inputs": query, "max_length": MAX_TOKENS}
    )
    print(response)  # Access the response variable
    end_time = time.time()
    return end_time - start_time

# Benchmarking function for AWS Bedrock
def benchmark_aws_bedrock(query):
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION")
    
    if not aws_access_key_id or not aws_secret_access_key or not aws_region:
        print(Fore.YELLOW + "Skipping AWS Bedrock benchmarking. AWS credentials not provided or invalid." + Style.RESET_ALL)
        return None
    
    print(Fore.CYAN + "Testing AWS Bedrock..." + Style.RESET_ALL)
    bedrock = boto3.client(
        "bedrock",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region
    )
    start_time = time.time()
    response = bedrock.generate_text(
        model="gpt-3.5-turbo",
        prompt=query,
        max_tokens=MAX_TOKENS
    )
    print(response)  # Access the response variable
    end_time = time.time()
    return end_time - start_time


# List of sample queries to test
queries = [
    "What is the capital of France?",
    # Reasoning: This query is a simple factual question that tests the model's ability to provide accurate information from its knowledge base.
    # It assesses the model's capability to handle straightforward questions and retrieve specific facts.

    "Explain the concept of machine learning in simple terms.",
    # Reasoning: This query evaluates the model's ability to provide a clear and concise explanation of a technical concept.
    # It tests the model's capacity to break down complex topics into easily understandable language, catering to a non-technical audience.

    "What are the main differences between renewable and non-renewable energy sources?",
    # Reasoning: This query assesses the model's ability to compare and contrast two related concepts.
    # It requires the model to identify key distinguishing factors and present them in a structured manner, demonstrating its understanding of the subject matter.

    "Suggest five creative ways to reduce plastic waste in everyday life.",
    # Reasoning: This query challenges the model's creativity and problem-solving skills.
    # It tests the model's ability to generate multiple unique ideas and provide practical solutions to a given problem.
    # It evaluates the model's capacity for original thought and its awareness of environmental issues.

    "Analyze the main themes and symbolism in the novel 'Hitchhiker's Guide to the Galaxy by Douglas Adams.",
    # Reasoning: This query assesses the model's ability to perform literary analysis and interpretation.
    # It tests the model's understanding of the deeper meanings, themes, and symbolic elements within a well-known literary work.
    # It evaluates the model's capacity to provide insights and draw connections between different aspects of the novel.

    "What are the potential benefits and drawbacks of artificial intelligence in healthcare?",
    # Reasoning: This query evaluates the model's ability to discuss the implications and considerations surrounding the application of AI in a specific domain.
    # It tests the model's capacity to present a balanced perspective, considering both the positive and negative aspects of the topic.
    # It assesses the model's understanding of the ethical and practical implications of AI in healthcare.

    "How can individuals and communities contribute to reducing the impact of climate change?",
    # Reasoning: This query assesses the model's ability to provide actionable advice and recommendations on a global issue.
    # It tests the model's understanding of climate change and its capacity to suggest practical steps that individuals and communities can take to mitigate its effects.
    # It evaluates the model's awareness of environmental sustainability and its ability to provide guidance on making a positive impact.
]

# Number of iterations for each API
num_iterations = 10

# Dictionary to store the total response time for each API
api_total_times = {}

# List of available APIs
available_apis = ["OpenAI", "Local OpenAI", "Azure OpenAI", "Anthropic", "Hugging Face", "AWS Bedrock"]

# Benchmarking loop for each API and query
def main():
    print("\nAPI Benchmarking Tool\n---------------------")

    # List of available APIs
    available_apis = ["OpenAI", "Local OpenAI", "Azure OpenAI", "Anthropic", "Hugging Face", "AWS Bedrock"]

    # Display available APIs and prompt user for selection
    print("Available APIs:")
    for i, api in enumerate(available_apis, start=1):
        print(f"{i}. {api}")

    # Prompt user to enter comma-separated API numbers
    selected_api_numbers = input("Enter the numbers of the APIs you want to benchmark (comma-separated): ")

    # Convert user input to a list of selected API names
    selected_api_numbers = [int(num.strip()) for num in selected_api_numbers.split(",")]
    selected_apis = [available_apis[num - 1] for num in selected_api_numbers]

    if not selected_apis:
        print("\nNo APIs selected. Please select one or more APIs to benchmark.")
        return

    # Initialize dictionary to store total times for each API
    api_total_times = {}

    for api_name in selected_apis:
        benchmark_func = globals()[f"benchmark_{api_name.lower().replace(' ', '_')}"]
        num_iterations = int(input(f"How many iterations would you like to run for {api_name}? "))

        print(f"\nBenchmarking {api_name}...")

        total_time = 0
        for _ in range(num_iterations):
            for query in queries:
                response_time = benchmark_func(query)
                if response_time is not None:
                    total_time += response_time

        if total_time > 0:
            api_total_times[api_name] = total_time

    # Calculate and print the average response time for each API
    print("\nBenchmarking Results:")
    for api_name, total_time in api_total_times.items():
        avg_time = total_time / (num_iterations * len(queries))

        if avg_time < 0.5:
            color = Fore.GREEN
        elif avg_time < 1.0:
            color = Fore.YELLOW
        else:
            color = Fore.RED

        print(f"{api_name} - Average response time: {color}{avg_time:.4f} seconds{Style.RESET_ALL}")

if __name__ == "__main__":
    main()