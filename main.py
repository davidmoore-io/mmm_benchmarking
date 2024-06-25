import os
from dotenv import load_dotenv
from colorama import init, Fore
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
from quality_metrics import calculate_quality_metrics

def main():
    init()  # Initialize colorama
    load_dotenv()  # Load environment variables
    print_colored("\nAPI Benchmarking Tool\n---------------------", color=Fore.CYAN)

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
    print_colored("Available APIs:", color=Fore.CYAN)
    for i, api in enumerate(available_apis, start=1):
        print(f"{i}. {api}")

    selected_api_numbers = get_user_input("Enter the numbers of the APIs you want to benchmark (comma-separated): ")
    selected_api_numbers = parse_comma_separated_input(selected_api_numbers)
    if not selected_api_numbers:
        return

    selected_apis = [available_apis[num - 1] for num in selected_api_numbers if 0 < num <= len(available_apis)]

    results = {}

    for api_name in selected_apis:
        print_colored(f"\nAvailable models for {api_name}:", color=Fore.CYAN)
        for i, model in enumerate(API_MODELS[api_name], start=1):
            print(f"{i}. {model}")
        
        selected_model_numbers = get_user_input(f"Enter the numbers of the models you want to benchmark for {api_name} (comma-separated): ")
        selected_model_numbers = parse_comma_separated_input(selected_model_numbers)
        if not selected_model_numbers:
            continue

        selected_models = [API_MODELS[api_name][num - 1] for num in selected_model_numbers if 0 < num <= len(API_MODELS[api_name])]

        num_iterations = int(get_user_input(f"How many iterations would you like to run for {api_name}? "))

        benchmark = benchmarks[api_name]
        for model in selected_models:
            total_time = 0
            total_metrics = {
                'bleu': 0,
                'rouge-1': 0,
                'rouge-2': 0,
                'rouge-l': 0
            }
            for _ in range(num_iterations):
                for query_ref in QUERIES_AND_REFERENCES:
                    query = query_ref["query"]
                    reference = query_ref["reference"]
                    latency, output = benchmark.run(query, model, MAX_TOKENS)
                    if latency is not None and output is not None:
                        total_time += latency
                        metrics = calculate_quality_metrics(reference, output)
                        for key, value in metrics.items():
                            total_metrics[key] += value

            if total_time > 0:
                avg_time = total_time / (num_iterations * len(QUERIES_AND_REFERENCES))
                avg_metrics = {k: v / (num_iterations * len(QUERIES_AND_REFERENCES)) for k, v in total_metrics.items()}
                results[(api_name, model)] = (avg_time, avg_metrics)

    # Print results
    print_colored("\nBenchmarking Results:", color=Fore.GREEN)
    for (api_name, model), (avg_time, avg_metrics) in results.items():
        print_colored(f"{api_name} ({model}):", color=Fore.CYAN)
        print_colored(f"  Average response time: {avg_time:.4f} seconds", color=get_time_color(avg_time))
        print_colored(f"  Average BLEU score: {avg_metrics['bleu']:.4f}", color=get_metric_color(avg_metrics['bleu']))
        print_colored(f"  Average ROUGE-1 score: {avg_metrics['rouge-1']:.4f}", color=get_metric_color(avg_metrics['rouge-1']))
        print_colored(f"  Average ROUGE-2 score: {avg_metrics['rouge-2']:.4f}", color=get_metric_color(avg_metrics['rouge-2']))
        print_colored(f"  Average ROUGE-L score: {avg_metrics['rouge-l']:.4f}", color=get_metric_color(avg_metrics['rouge-l']))

def get_time_color(avg_time):
    if avg_time < 0.5:
        return Fore.GREEN
    elif avg_time < 1.0:
        return Fore.YELLOW
    else:
        return Fore.RED

def get_metric_color(score):
    if score > 0.5:
        return Fore.GREEN
    elif score > 0.3:
        return Fore.YELLOW
    else:
        return Fore.RED

if __name__ == "__main__":
    main()