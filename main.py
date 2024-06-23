import os
from dotenv import load_dotenv
from colorama import init
from config import API_MODELS, QUERIES, MAX_TOKENS
from benchmarks import (
    OpenAIBenchmark,
    AzureOpenAIBenchmark,
    AnthropicBenchmark,
    LocalOpenAIBenchmark,
    HuggingFaceBenchmark,
    AWSBedrockBenchmark
)
from utils import print_colored, get_user_input, parse_comma_separated_input

def main():
    init()  # Initialize colorama
    load_dotenv()  # Load environment variables
    print_colored("\nAPI Benchmarking Tool\n---------------------", color="cyan")

    benchmarks = {
        "OpenAI": OpenAIBenchmark(),
        "Azure OpenAI": AzureOpenAIBenchmark(),
        "Anthropic": AnthropicBenchmark(),
        "Local OpenAI": LocalOpenAIBenchmark(),
        "Hugging Face": HuggingFaceBenchmark(),
        "AWS Bedrock": AWSBedrockBenchmark(),
    }

    # Display available APIs and prompt user for selection
    available_apis = list(API_MODELS.keys())
    print_colored("Available APIs:", color="cyan")
    for i, api in enumerate(available_apis, start=1):
        print(f"{i}. {api}")

    selected_api_numbers = get_user_input("Enter the numbers of the APIs you want to benchmark (comma-separated): ")
    selected_api_numbers = parse_comma_separated_input(selected_api_numbers)
    if not selected_api_numbers:
        return

    selected_apis = [available_apis[num - 1] for num in selected_api_numbers]

    results = {}

    for api_name in selected_apis:
        print_colored(f"\nAvailable models for {api_name}:", color="cyan")
        for i, model in enumerate(API_MODELS[api_name], start=1):
            print(f"{i}. {model}")
        
        selected_model_numbers = get_user_input(f"Enter the numbers of the models you want to benchmark for {api_name} (comma-separated): ")
        selected_model_numbers = parse_comma_separated_input(selected_model_numbers)
        if not selected_model_numbers:
            continue

        selected_models = [API_MODELS[api_name][num - 1] for num in selected_model_numbers]

        num_iterations = int(get_user_input(f"How many iterations would you like to run for {api_name}? "))

        benchmark = benchmarks[api_name]
        for model in selected_models:
            total_time = 0
            for _ in range(num_iterations):
                for query in QUERIES:
                    response_time = benchmark.run(query, model, MAX_TOKENS)
                    if response_time is not None:
                        total_time += response_time

            if total_time > 0:
                avg_time = total_time / (num_iterations * len(QUERIES))
                results[(api_name, model)] = avg_time

    # Print results
    print_colored("\nBenchmarking Results:", color="green")
    for (api_name, model), avg_time in results.items():
        if avg_time < 0.5:
            color = "green"
        elif avg_time < 1.0:
            color = "yellow"
        else:
            color = "red"

        print_colored(f"{api_name} ({model}) - Average response time: {avg_time:.4f} seconds", color=color)

if __name__ == "__main__":
    main()