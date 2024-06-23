import os
from dotenv import load_dotenv
from colorama import init
from config import API_MODELS, QUERIES_AND_REFERENCES, MAX_TOKENS
from benchmarks import (
    OpenAIBenchmark,
    AzureOpenAIBenchmark,
    AnthropicBenchmark,
    LocalOpenAIBenchmark,
    HuggingFaceBenchmark,
    AWSBedrockBenchmark
)
from utils import print_colored, get_user_input, parse_comma_separated_input
from quality_metrics import calculate_bleu_score

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
            total_bleu = 0
            for _ in range(num_iterations):
                for query_ref in QUERIES_AND_REFERENCES:
                    query = query_ref["query"]
                    reference = query_ref["reference"]
                    latency, output = benchmark.run(query, model, MAX_TOKENS)
                    if latency is not None and output is not None:
                        total_time += latency
                        bleu_score = calculate_bleu_score(reference, output)
                        total_bleu += bleu_score

            if total_time > 0:
                avg_time = total_time / (num_iterations * len(QUERIES_AND_REFERENCES))
                avg_bleu = total_bleu / (num_iterations * len(QUERIES_AND_REFERENCES))
                results[(api_name, model)] = (avg_time, avg_bleu)

    # Print results
    print_colored("\nBenchmarking Results:", color="green")
    for (api_name, model), (avg_time, avg_bleu) in results.items():
        if avg_time < 0.5:
            time_color = "green"
        elif avg_time < 1.0:
            time_color = "yellow"
        else:
            time_color = "red"

        if avg_bleu > 0.5:
            bleu_color = "green"
        elif avg_bleu > 0.3:
            bleu_color = "yellow"
        else:
            bleu_color = "red"

        print_colored(f"{api_name} ({model}):", color="cyan")
        print_colored(f"  Average response time: {avg_time:.4f} seconds", color=time_color)
        print_colored(f"  Average BLEU score: {avg_bleu:.4f}", color=bleu_color)

if __name__ == "__main__":
    main()