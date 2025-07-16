"""
Enhanced Main Module

This module integrates all the new features including enhanced quality metrics,
human evaluation, and A/B testing with the existing benchmarking system.
"""

import os
import json
from datetime import datetime
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
from enhanced_quality_metrics import EnhancedQualityMetrics
from human_evaluation import HumanEvaluator, EvaluationInterface
from ab_testing import ABTestManager, ABTestInterface, ComparisonType
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedBenchmarkingTool:
    """
    Enhanced benchmarking tool with advanced quality metrics, human evaluation,
    and A/B testing capabilities.
    """
    
    def __init__(self):
        self.benchmarks = {
            "OpenAI": OpenAIBenchmark(),
            "Azure OpenAI": AzureOpenAIBenchmark(),
            "Anthropic": AnthropicBenchmark(),
            "Local OpenAI": LocalOpenAIBenchmark(),
            "Hugging Face": HuggingFaceBenchmark(),
            "AWS Bedrock": AWSBedrockBenchmark(),
        }
        self.enhanced_metrics = EnhancedQualityMetrics()
        self.human_evaluator = HumanEvaluator()
        self.ab_test_manager = ABTestManager()
        self.results = []
        
    def run_standard_benchmarks(self) -> List[Dict[str, Any]]:
        """Run standard benchmarks with basic metrics."""
        print_colored("\n=== Standard Benchmarking ===", color=Fore.CYAN)
        
        # Get user selections
        selected_apis = self._get_api_selections()
        if not selected_apis:
            return []
        
        results = {}
        benchmark_results = []
        
        for api_name in selected_apis:
            selected_models = self._get_model_selections(api_name)
            if not selected_models:
                continue
                
            num_iterations = int(get_user_input(f"How many iterations for {api_name}? "))
            
            benchmark = self.benchmarks[api_name]
            for model in selected_models:
                print_colored(f"\nBenchmarking {api_name} - {model}", color=Fore.YELLOW)
                
                total_time = 0
                total_metrics = {
                    'bleu': 0,
                    'rouge-1': 0,
                    'rouge-2': 0,
                    'rouge-l': 0
                }
                
                for iteration in range(num_iterations):
                    for query_ref in QUERIES_AND_REFERENCES:
                        query = query_ref["query"]
                        reference = query_ref["reference"]
                        
                        latency, output = benchmark.run(query, model, MAX_TOKENS)
                        if latency is not None and output is not None:
                            total_time += latency
                            metrics = calculate_quality_metrics(reference, output)
                            
                            for key, value in metrics.items():
                                total_metrics[key] += value
                            
                            # Store individual result for later analysis
                            benchmark_results.append({
                                'query': query,
                                'reference': reference,
                                'response': output,
                                'api_name': api_name,
                                'model_name': model,
                                'latency': latency,
                                'iteration': iteration,
                                'basic_metrics': metrics
                            })
                
                if total_time > 0:
                    num_queries = num_iterations * len(QUERIES_AND_REFERENCES)
                    avg_time = total_time / num_queries
                    avg_metrics = {k: v / num_queries for k, v in total_metrics.items()}
                    results[(api_name, model)] = (avg_time, avg_metrics)
        
        # Display results
        self._display_standard_results(results)
        
        self.results = benchmark_results
        return benchmark_results
    
    def run_enhanced_benchmarks(self) -> List[Dict[str, Any]]:
        """Run benchmarks with enhanced quality metrics."""
        print_colored("\n=== Enhanced Benchmarking ===", color=Fore.CYAN)
        
        if not self.results:
            print_colored("No previous benchmark results found. Running standard benchmarks first...", 
                         color=Fore.YELLOW)
            self.run_standard_benchmarks()
        
        enhanced_results = []
        
        print_colored("Calculating enhanced metrics...", color=Fore.YELLOW)
        
        for result in self.results:
            print_colored(f"Processing {result['api_name']} - {result['model_name']}", color=Fore.CYAN)
            
            # Calculate enhanced metrics
            enhanced_metrics = self.enhanced_metrics.calculate_all_metrics(
                result['reference'], result['response']
            )
            
            # Combine with existing result
            enhanced_result = result.copy()
            enhanced_result['enhanced_metrics'] = enhanced_metrics
            enhanced_results.append(enhanced_result)
        
        # Display enhanced results
        self._display_enhanced_results(enhanced_results)
        
        return enhanced_results
    
    def setup_human_evaluation(self) -> str:
        """Set up human evaluation tasks."""
        print_colored("\n=== Human Evaluation Setup ===", color=Fore.CYAN)
        
        if not self.results:
            print_colored("No benchmark results available for human evaluation.", color=Fore.RED)
            return None
        
        # Create evaluation tasks
        task_ids = self.human_evaluator.create_evaluation_tasks(self.results)
        
        print_colored(f"Created {len(task_ids)} evaluation tasks", color=Fore.GREEN)
        
        return task_ids
    
    def run_human_evaluation(self, evaluator_id: str):
        """Run human evaluation session."""
        print_colored("\n=== Human Evaluation Session ===", color=Fore.CYAN)
        
        interface = EvaluationInterface(self.human_evaluator)
        interface.run_evaluation_session(evaluator_id)
        
        # Show statistics
        stats = self.human_evaluator.get_evaluation_statistics()
        self._display_human_evaluation_stats(stats)
    
    def setup_ab_testing(self) -> str:
        """Set up A/B testing experiment."""
        print_colored("\n=== A/B Testing Setup ===", color=Fore.CYAN)
        
        if not self.results:
            print_colored("No benchmark results available for A/B testing.", color=Fore.RED)
            return None
        
        # Get experiment details
        name = get_user_input("Experiment name: ")
        description = get_user_input("Experiment description: ")
        
        # Extract models from results
        models_under_test = list(set([
            (r['api_name'], r['model_name']) for r in self.results
        ]))
        
        # Create experiment
        experiment_id = self.ab_test_manager.create_experiment(
            name, description, models_under_test
        )
        
        # Add comparison tasks
        task_ids = self.ab_test_manager.add_comparison_tasks(
            experiment_id, self.results, ComparisonType.PAIRWISE
        )
        
        print_colored(f"Created experiment '{name}' with {len(task_ids)} comparison tasks", 
                     color=Fore.GREEN)
        
        return experiment_id
    
    def run_ab_testing(self, experiment_id: str, evaluator_id: str):
        """Run A/B testing session."""
        print_colored("\n=== A/B Testing Session ===", color=Fore.CYAN)
        
        interface = ABTestInterface(self.ab_test_manager)
        interface.run_comparison_session(experiment_id, evaluator_id)
        
        # Show statistics
        stats = self.ab_test_manager.get_experiment_statistics(experiment_id)
        self._display_ab_testing_stats(stats)
    
    def export_results(self, output_dir: str = "results"):
        """Export all results to files."""
        print_colored("\n=== Exporting Results ===", color=Fore.CYAN)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export benchmark results
        if self.results:
            benchmark_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
            with open(benchmark_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print_colored(f"Benchmark results exported to: {benchmark_file}", color=Fore.GREEN)
        
        # Export human evaluation results
        eval_file = os.path.join(output_dir, f"human_evaluation_{timestamp}.json")
        if self.human_evaluator.export_evaluation_results(eval_file):
            print_colored(f"Human evaluation results exported to: {eval_file}", color=Fore.GREEN)
        
        print_colored("Export completed", color=Fore.GREEN)
    
    def _get_api_selections(self) -> List[str]:
        """Get API selections from user."""
        available_apis = list(API_MODELS.keys())
        print_colored("Available APIs:", color=Fore.CYAN)
        for i, api in enumerate(available_apis, start=1):
            print(f"{i}. {api}")
        
        selected_api_numbers = get_user_input(
            "Enter the numbers of the APIs you want to benchmark (comma-separated): "
        )
        selected_api_numbers = parse_comma_separated_input(selected_api_numbers)
        
        if not selected_api_numbers:
            return []
        
        return [available_apis[num - 1] for num in selected_api_numbers 
                if 0 < num <= len(available_apis)]
    
    def _get_model_selections(self, api_name: str) -> List[str]:
        """Get model selections for a specific API."""
        print_colored(f"\nAvailable models for {api_name}:", color=Fore.CYAN)
        for i, model in enumerate(API_MODELS[api_name], start=1):
            print(f"{i}. {model}")
        
        selected_model_numbers = get_user_input(
            f"Enter the numbers of the models for {api_name} (comma-separated): "
        )
        selected_model_numbers = parse_comma_separated_input(selected_model_numbers)
        
        if not selected_model_numbers:
            return []
        
        return [API_MODELS[api_name][num - 1] for num in selected_model_numbers 
                if 0 < num <= len(API_MODELS[api_name])]
    
    def _display_standard_results(self, results: Dict):
        """Display standard benchmark results."""
        print_colored("\n=== Standard Benchmark Results ===", color=Fore.GREEN)
        
        for (api_name, model), (avg_time, avg_metrics) in results.items():
            print_colored(f"\n{api_name} ({model}):", color=Fore.CYAN)
            print_colored(f"  Average response time: {avg_time:.4f} seconds", 
                         color=self._get_time_color(avg_time))
            print_colored(f"  Average BLEU score: {avg_metrics['bleu']:.4f}", 
                         color=self._get_metric_color(avg_metrics['bleu']))
            print_colored(f"  Average ROUGE-1 score: {avg_metrics['rouge-1']:.4f}", 
                         color=self._get_metric_color(avg_metrics['rouge-1']))
            print_colored(f"  Average ROUGE-2 score: {avg_metrics['rouge-2']:.4f}", 
                         color=self._get_metric_color(avg_metrics['rouge-2']))
            print_colored(f"  Average ROUGE-L score: {avg_metrics['rouge-l']:.4f}", 
                         color=self._get_metric_color(avg_metrics['rouge-l']))
    
    def _display_enhanced_results(self, results: List[Dict]):
        """Display enhanced benchmark results."""
        print_colored("\n=== Enhanced Benchmark Results ===", color=Fore.GREEN)
        
        # Group results by API and model
        grouped = {}
        for result in results:
            key = (result['api_name'], result['model_name'])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # Display aggregated results
        for (api_name, model), group_results in grouped.items():
            print_colored(f"\n{api_name} ({model}):", color=Fore.CYAN)
            
            # Calculate averages
            enhanced_metrics = group_results[0]['enhanced_metrics']
            avg_enhanced = {}
            
            for metric_name in enhanced_metrics.keys():
                values = [r['enhanced_metrics'][metric_name] for r in group_results 
                         if isinstance(r['enhanced_metrics'][metric_name], (int, float))]
                if values:
                    avg_enhanced[metric_name] = sum(values) / len(values)
            
            # Display enhanced metrics
            for metric_name, value in avg_enhanced.items():
                color = self._get_metric_color(value) if value <= 1 else Fore.WHITE
                print_colored(f"  {metric_name}: {value:.4f}", color=color)
    
    def _display_human_evaluation_stats(self, stats: Dict):
        """Display human evaluation statistics."""
        print_colored("\n=== Human Evaluation Statistics ===", color=Fore.GREEN)
        
        print_colored(f"Total tasks: {stats.get('total_tasks', 0)}", color=Fore.CYAN)
        print_colored(f"Completed tasks: {stats.get('completed_tasks', 0)}", color=Fore.CYAN)
        print_colored(f"Completion rate: {stats.get('completion_rate', 0):.2%}", color=Fore.CYAN)
        
        criterion_stats = stats.get('criterion_statistics', {})
        if criterion_stats:
            print_colored("\nCriterion Statistics:", color=Fore.CYAN)
            for criterion, stat in criterion_stats.items():
                print_colored(f"  {criterion}: {stat['average_score']:.2f} (n={stat['count']})", 
                             color=Fore.WHITE)
    
    def _display_ab_testing_stats(self, stats: Dict):
        """Display A/B testing statistics."""
        print_colored("\n=== A/B Testing Statistics ===", color=Fore.GREEN)
        
        print_colored(f"Total comparisons: {stats.get('total_comparisons', 0)}", color=Fore.CYAN)
        print_colored(f"Average confidence: {stats.get('average_confidence', 0):.2f}", color=Fore.CYAN)
        
        win_rates = stats.get('model_win_rates', {})
        if win_rates:
            print_colored("\nModel Win Rates:", color=Fore.CYAN)
            for model, rate in sorted(win_rates.items(), key=lambda x: x[1], reverse=True):
                print_colored(f"  {model}: {rate:.2%}", color=Fore.WHITE)
        
        preference_dist = stats.get('preference_distribution', {})
        if preference_dist:
            print_colored("\nPreference Distribution:", color=Fore.CYAN)
            total = sum(preference_dist.values())
            for pref, count in preference_dist.items():
                percentage = count / total * 100 if total > 0 else 0
                print_colored(f"  {pref}: {count} ({percentage:.1f}%)", color=Fore.WHITE)
    
    def _get_time_color(self, avg_time: float) -> str:
        """Get color for response time display."""
        if avg_time < 0.5:
            return Fore.GREEN
        elif avg_time < 1.0:
            return Fore.YELLOW
        else:
            return Fore.RED
    
    def _get_metric_color(self, score: float) -> str:
        """Get color for metric score display."""
        if score > 0.5:
            return Fore.GREEN
        elif score > 0.3:
            return Fore.YELLOW
        else:
            return Fore.RED


def main():
    """Main function for enhanced benchmarking tool."""
    init()  # Initialize colorama
    load_dotenv()  # Load environment variables
    
    print_colored("\n=== Enhanced LLM Benchmarking Tool ===", color=Fore.CYAN)
    print_colored("Features: Standard benchmarks, Enhanced metrics, Human evaluation, A/B testing", 
                 color=Fore.YELLOW)
    
    tool = EnhancedBenchmarkingTool()
    
    while True:
        print_colored("\n=== Main Menu ===", color=Fore.CYAN)
        print("1. Run standard benchmarks")
        print("2. Run enhanced benchmarks")
        print("3. Setup human evaluation")
        print("4. Run human evaluation")
        print("5. Setup A/B testing")
        print("6. Run A/B testing")
        print("7. Export results")
        print("8. Exit")
        
        choice = get_user_input("Select option (1-8): ")
        
        if choice == '1':
            tool.run_standard_benchmarks()
        elif choice == '2':
            tool.run_enhanced_benchmarks()
        elif choice == '3':
            tool.setup_human_evaluation()
        elif choice == '4':
            evaluator_id = get_user_input("Enter evaluator ID: ")
            tool.run_human_evaluation(evaluator_id)
        elif choice == '5':
            experiment_id = tool.setup_ab_testing()
            if experiment_id:
                print_colored(f"Experiment ID: {experiment_id}", color=Fore.GREEN)
        elif choice == '6':
            experiment_id = get_user_input("Enter experiment ID: ")
            evaluator_id = get_user_input("Enter evaluator ID: ")
            tool.run_ab_testing(experiment_id, evaluator_id)
        elif choice == '7':
            output_dir = get_user_input("Output directory (default: results): ") or "results"
            tool.export_results(output_dir)
        elif choice == '8':
            print_colored("Goodbye!", color=Fore.GREEN)
            break
        else:
            print_colored("Invalid option. Please try again.", color=Fore.RED)


if __name__ == "__main__":
    main()