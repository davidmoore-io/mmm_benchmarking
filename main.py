import os
from typing import List, Optional
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.prompt import Prompt, IntPrompt
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.align import Align
from rich.live import Live
import time

from config import API_MODELS, QUERIES_AND_REFERENCES, MAX_TOKENS
from benchmarks import (
    OpenAIBenchmark,
    AzureOpenAIBenchmark,
    AnthropicBenchmark,
    LocalOpenAIBenchmark,
    HuggingFaceBenchmark,
    AWSBedrockBenchmark
)
from utils import (
    print_header, print_section, print_success, print_warning, print_error, print_info,
    create_selection_table, create_results_table, get_time_color, get_metric_color,
    get_integer_input, get_confirmation, console, get_multi_selection_input, get_iterations_input,
    display_key_value_pairs, show_menu, get_user_input
)
from quality_metrics import calculate_quality_metrics, EnhancedQualityMetrics
from human_evaluation import HumanEvaluator, EvaluationInterface
from ab_testing import ABTestManager, ABTestInterface, ComparisonType
import json
from datetime import datetime
import logging

app = typer.Typer(
    help="🚀 LLM API Benchmarking Tool - Compare performance and quality across multiple providers",
    no_args_is_help=True
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global benchmark instances
BENCHMARKS = {
    "OpenAI": OpenAIBenchmark(),
    "Azure OpenAI": AzureOpenAIBenchmark(),
    "Anthropic": AnthropicBenchmark(),
    "Local OpenAI": LocalOpenAIBenchmark(),
    "Hugging Face": HuggingFaceBenchmark(),
    "AWS Bedrock": AWSBedrockBenchmark(),
}

class EnhancedBenchmarkingTool:
    """
    Enhanced benchmarking tool with advanced quality metrics, human evaluation,
    and A/B testing capabilities.
    """
    
    def __init__(self):
        self.benchmarks = BENCHMARKS
        self.enhanced_metrics = EnhancedQualityMetrics()
        self.human_evaluator = HumanEvaluator()
        self.ab_test_manager = ABTestManager()
        self.results = []
        
    def run_standard_benchmarks(self) -> List[dict]:
        """Run standard benchmarks with basic metrics."""
        print_section("🚀 Standard Benchmarking")
        
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
                
            num_iterations = get_iterations_input(api_name)
            
            benchmark = self.benchmarks[api_name]
            for model in selected_models:
                # Display model info in a panel
                model_panel = Panel(
                    f"🤖 [bold cyan]Model:[/bold cyan] {model}\n"
                    f"🔧 [bold cyan]Provider:[/bold cyan] {api_name}\n"
                    f"🔄 [bold cyan]Iterations:[/bold cyan] {num_iterations}\n"
                    f"📝 [bold cyan]Queries:[/bold cyan] {len(QUERIES_AND_REFERENCES)}",
                    title="🧪 Running Benchmark",
                    border_style="green"
                )
                console.print(model_panel)
                
                total_time = 0
                total_metrics = {
                    'bleu': 0,
                    'rouge-1': 0,
                    'rouge-2': 0,
                    'rouge-l': 0
                }
                
                total_queries = num_iterations * len(QUERIES_AND_REFERENCES)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    
                    task = progress.add_task(
                        f"🚀 Testing {model}...", 
                        total=total_queries
                    )
                    
                    for iteration in range(num_iterations):
                        for query_ref in QUERIES_AND_REFERENCES:
                            query = query_ref["query"]
                            reference = query_ref["reference"]
                            
                            latency, response = benchmark.run(query, model, MAX_TOKENS)
                            if latency is not None and response is not None:
                                total_time += latency
                                metrics = calculate_quality_metrics(reference, response)
                                
                                for key, value in metrics.items():
                                    total_metrics[key] += value
                                
                                # Store individual result for later analysis
                                benchmark_results.append({
                                    'query': query,
                                    'reference': reference,
                                    'response': response,
                                    'api_name': api_name,
                                    'model_name': model,
                                    'latency': latency,
                                    'iteration': iteration,
                                    'basic_metrics': metrics
                                })
                            
                            progress.update(task, advance=1)
                
                if total_time > 0:
                    avg_time = total_time / total_queries
                    avg_metrics = {k: v / total_queries for k, v in total_metrics.items()}
                    results[(api_name, model)] = (avg_time, avg_metrics)
                    
                    # Display completion in a panel
                    completion_panel = Panel(
                        f"✅ [green]Benchmark Complete![/green]\n\n"
                        f"⏱️  [bold]Average Response Time:[/bold] {avg_time:.4f}s\n"
                        f"🎯 [bold]BLEU Score:[/bold] {avg_metrics['bleu']:.4f}\n"
                        f"📝 [bold]ROUGE-1 Score:[/bold] {avg_metrics['rouge-1']:.4f}",
                        title=f"📊 {model} Results",
                        border_style="green"
                    )
                    console.print(completion_panel)
        
        # Display results
        display_beautiful_results(results)
        
        self.results = benchmark_results
        return benchmark_results
    
    def run_enhanced_benchmarks(self) -> List[dict]:
        """Run benchmarks with enhanced quality metrics."""
        print_section("⚡ Enhanced Benchmarking")
        
        if not self.results:
            print_warning("No previous benchmark results found. Running standard benchmarks first...")
            self.run_standard_benchmarks()
        
        enhanced_results = []
        
        print_info("Calculating enhanced metrics...")
        
        for result in self.results:
            print_info(f"Processing {result['api_name']} - {result['model_name']}")
            
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
        print_section("👥 Human Evaluation Setup")
        
        if not self.results:
            print_error("No benchmark results available for human evaluation.")
            return None
        
        # Create evaluation tasks
        task_ids = self.human_evaluator.create_evaluation_tasks(self.results)
        
        print_success(f"Created {len(task_ids)} evaluation tasks")
        
        return task_ids
    
    def run_human_evaluation(self, evaluator_id: str):
        """Run human evaluation session."""
        print_section("🔍 Human Evaluation Session")
        
        interface = EvaluationInterface(self.human_evaluator)
        interface.run_evaluation_session(evaluator_id)
        
        # Show statistics
        stats = self.human_evaluator.get_evaluation_statistics()
        self._display_human_evaluation_stats(stats)
    
    def setup_ab_testing(self) -> str:
        """Set up A/B testing experiment."""
        print_section("🆚 A/B Testing Setup")
        
        if not self.results:
            print_error("No benchmark results available for A/B testing.")
            return None
        
        # Get experiment details
        name = get_user_input("🔬 Experiment name")
        description = get_user_input("📝 Experiment description")
        
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
        
        print_success(f"Created experiment '{name}' with {len(task_ids)} comparison tasks")
        
        return experiment_id
    
    def run_ab_testing(self, experiment_id: str, evaluator_id: str):
        """Run A/B testing session."""
        print_section("📊 A/B Testing Session")
        
        interface = ABTestInterface(self.ab_test_manager)
        interface.run_comparison_session(experiment_id, evaluator_id)
        
        # Show statistics
        stats = self.ab_test_manager.get_experiment_statistics(experiment_id)
        self._display_ab_testing_stats(stats)
    
    def export_results(self, output_dir: str = "results"):
        """Export all results to files."""
        print_section("💾 Exporting Results")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export benchmark results
        if self.results:
            benchmark_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
            with open(benchmark_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print_success(f"Benchmark results exported to: {benchmark_file}")
        
        # Export human evaluation results
        eval_file = os.path.join(output_dir, f"human_evaluation_{timestamp}.json")
        if self.human_evaluator.export_evaluation_results(eval_file):
            print_success(f"Human evaluation results exported to: {eval_file}")
        
        print_success("Export completed")
    
    def _get_api_selections(self) -> List[str]:
        """Get API selections from user."""
        available_apis = list(API_MODELS.keys())
        return get_multi_selection_input(available_apis, "APIs")
    
    def _get_model_selections(self, api_name: str) -> List[str]:
        """Get model selections for a specific API."""
        models = API_MODELS[api_name]
        return get_multi_selection_input(models, f"models for {api_name}")
    
    def _display_enhanced_results(self, results: List[dict]):
        """Display enhanced benchmark results."""
        print_section("⚡ Enhanced Benchmark Results")
        
        if not results:
            print_warning("No results to display")
            return
        
        # Group results by API and model
        grouped = {}
        for result in results:
            key = (result['api_name'], result['model_name'])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # Display aggregated results
        for (api_name, model), group_results in grouped.items():
            print_section(f"🔍 {api_name} - {model}")
            
            # Calculate averages
            enhanced_metrics = group_results[0]['enhanced_metrics']
            avg_enhanced = {}
            
            for metric_name in enhanced_metrics.keys():
                values = [r['enhanced_metrics'][metric_name] for r in group_results 
                         if isinstance(r['enhanced_metrics'][metric_name], (int, float))]
                if values:
                    avg_enhanced[metric_name] = sum(values) / len(values)
            
            # Display enhanced metrics in a table
            from utils import display_key_value_pairs
            display_key_value_pairs(avg_enhanced, f"📈 Enhanced Metrics for {model}")
            console.print()
    
    def _display_human_evaluation_stats(self, stats: dict):
        """Display human evaluation statistics."""
        print_section("👥 Human Evaluation Statistics")
        
        overview_stats = {
            "📊 Total tasks": stats.get('total_tasks', 0),
            "✅ Completed tasks": stats.get('completed_tasks', 0),
            "📈 Completion rate": f"{stats.get('completion_rate', 0):.2%}"
        }
        
        from utils import display_key_value_pairs
        display_key_value_pairs(overview_stats, "📋 Overview")
        
        criterion_stats = stats.get('criterion_statistics', {})
        if criterion_stats:
            criterion_display = {}
            for criterion, stat in criterion_stats.items():
                criterion_display[f"🎯 {criterion}"] = f"{stat['average_score']:.2f} (n={stat['count']})"
            
            display_key_value_pairs(criterion_display, "📊 Criterion Statistics")
    
    def _display_ab_testing_stats(self, stats: dict):
        """Display A/B testing statistics."""
        print_section("🆚 A/B Testing Statistics")
        
        overview_stats = {
            "📊 Total comparisons": stats.get('total_comparisons', 0),
            "📈 Average confidence": f"{stats.get('average_confidence', 0):.2f}"
        }
        
        from utils import display_key_value_pairs
        display_key_value_pairs(overview_stats, "📋 Overview")
        
        win_rates = stats.get('model_win_rates', {})
        if win_rates:
            win_rate_display = {}
            for model, rate in sorted(win_rates.items(), key=lambda x: x[1], reverse=True):
                win_rate_display[f"🏆 {model}"] = f"{rate:.2%}"
            
            display_key_value_pairs(win_rate_display, "🏆 Model Win Rates")
        
        preference_dist = stats.get('preference_distribution', {})
        if preference_dist:
            pref_display = {}
            total = sum(preference_dist.values())
            for pref, count in preference_dist.items():
                percentage = count / total * 100 if total > 0 else 0
                pref_display[f"📊 {pref}"] = f"{count} ({percentage:.1f}%)"
            
            display_key_value_pairs(pref_display, "📊 Preference Distribution")


def display_beautiful_results(results: dict):
    """Display results in a beautiful table format."""
    if not results:
        print_warning("No results to display")
        return
    
    print_section("🏆 Benchmarking Results")
    
    # Create results table
    table = Table(title="📊 Performance & Quality Metrics", show_header=True, header_style="bold green")
    table.add_column("Provider", style="cyan", width=15)
    table.add_column("Model", style="white", width=20)
    table.add_column("⏱️ Avg Time (s)", style="yellow", justify="right", width=12)
    table.add_column("🎯 BLEU", style="green", justify="right", width=8)
    table.add_column("📝 ROUGE-1", style="green", justify="right", width=10)
    table.add_column("📝 ROUGE-2", style="green", justify="right", width=10)
    table.add_column("📝 ROUGE-L", style="green", justify="right", width=10)
    
    # Sort results by average time
    sorted_results = sorted(results.items(), key=lambda x: x[1][0])
    
    for (api_name, model), (avg_time, avg_metrics) in sorted_results:
        time_color = get_time_color(avg_time)
        bleu_color = get_metric_color(avg_metrics['bleu'])
        rouge1_color = get_metric_color(avg_metrics['rouge-1'])
        rouge2_color = get_metric_color(avg_metrics['rouge-2'])
        rougel_color = get_metric_color(avg_metrics['rouge-l'])
        
        table.add_row(
            f"[{time_color}]{api_name}[/{time_color}]",
            f"[{time_color}]{model}[/{time_color}]",
            f"[{time_color}]{avg_time:.4f}[/{time_color}]",
            f"[{bleu_color}]{avg_metrics['bleu']:.4f}[/{bleu_color}]",
            f"[{rouge1_color}]{avg_metrics['rouge-1']:.4f}[/{rouge1_color}]",
            f"[{rouge2_color}]{avg_metrics['rouge-2']:.4f}[/{rouge2_color}]",
            f"[{rougel_color}]{avg_metrics['rouge-l']:.4f}[/{rougel_color}]"
        )
    
    console.print(table)
    
    # Show performance insights
    if len(sorted_results) > 1:
        fastest = sorted_results[0]
        slowest = sorted_results[-1]
        
        best_bleu = max(results.values(), key=lambda x: x[1]['bleu'])
        best_rouge1 = max(results.values(), key=lambda x: x[1]['rouge-1'])
        
        insights = Panel(
            f"🏃‍♂️ [green]Fastest:[/green] {fastest[0][0]} ({fastest[0][1]}) - {fastest[1][0]:.4f}s\n"
            f"🐌 [red]Slowest:[/red] {slowest[0][0]} ({slowest[0][1]}) - {slowest[1][0]:.4f}s\n"
            f"🎯 [green]Best BLEU:[/green] {best_bleu[1]['bleu']:.4f}\n"
            f"📝 [green]Best ROUGE-1:[/green] {best_rouge1[1]['rouge-1']:.4f}",
            title="🔍 Performance Insights",
            border_style="cyan"
        )
        console.print(insights)

def display_enhanced_results(benchmark_results: List[dict]):
    """Display enhanced benchmark results."""
    print_section("⚡ Enhanced Quality Metrics")
    
    if not benchmark_results:
        print_warning("No enhanced results to display")
        return
    
    # Group results by API and model
    grouped = {}
    for result in benchmark_results:
        if 'enhanced_metrics' not in result:
            continue
            
        key = (result['api_name'], result['model_name'])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)
    
    # Display aggregated enhanced results
    for (api_name, model), group_results in grouped.items():
        print_section(f"🔍 {api_name} - {model}")
        
        # Calculate averages for enhanced metrics
        enhanced_metrics = group_results[0]['enhanced_metrics']
        avg_enhanced = {}
        
        for metric_name in enhanced_metrics.keys():
            values = [r['enhanced_metrics'][metric_name] for r in group_results 
                     if isinstance(r['enhanced_metrics'][metric_name], (int, float))]
            if values:
                avg_enhanced[metric_name] = sum(values) / len(values)
        
        # Display enhanced metrics in a formatted way
        if avg_enhanced:
            # Format the metrics nicely
            formatted_metrics = {}
            for metric, value in avg_enhanced.items():
                if metric == 'perplexity':
                    formatted_metrics[f"📊 {metric}"] = f"{value:.2f} (lower is better)"
                elif 'similarity' in metric:
                    formatted_metrics[f"📈 {metric}"] = f"{value:.4f}"
                elif 'accuracy' in metric or 'precision' in metric or 'recall' in metric:
                    formatted_metrics[f"🎯 {metric}"] = f"{value:.4f}"
                else:
                    formatted_metrics[f"📝 {metric}"] = f"{value:.4f}"
            
            display_key_value_pairs(formatted_metrics, f"📊 Enhanced Metrics")
        console.print()

def run_benchmark_with_progress(benchmark, query_ref, model, max_tokens):
    """Run a single benchmark with progress indication."""
    query = query_ref["query"]
    reference = query_ref["reference"]
    
    latency, response = benchmark.run(query, model, max_tokens)
    
    if latency is not None and response is not None:
        metrics = calculate_quality_metrics(reference, response)
        return latency, metrics
    return None, None

@app.command()
def benchmark(
    interactive: bool = typer.Option(True, help="Run in interactive mode"),
    apis: Optional[List[str]] = typer.Option(None, help="APIs to benchmark"),
    models: Optional[List[str]] = typer.Option(None, help="Models to benchmark"),
    iterations: int = typer.Option(3, help="Number of iterations per model"),
    output: Optional[str] = typer.Option(None, help="Output file for results"),
    enhanced: bool = typer.Option(False, help="Include enhanced quality metrics (perplexity, semantic similarity, etc.)")
):
    """
    🚀 Run LLM API benchmarks with beautiful progress tracking and results.
    
    Compare response times and quality metrics across multiple providers.
    Use --enhanced flag to include advanced metrics like perplexity and semantic similarity.
    """
    load_dotenv()
    
    print_header(
        "🚀 LLM API Benchmarking Tool",
        "Compare performance and quality across multiple providers"
    )
    
    if interactive:
        # Interactive mode
        available_apis = list(API_MODELS.keys())
        selected_apis = get_multi_selection_input(available_apis, "APIs")
        
        if not selected_apis:
            print_error("No APIs selected. Exiting...")
            return
        
        # Ask about enhanced metrics if not specified
        if not enhanced:
            enhanced = get_confirmation("Include enhanced quality metrics (perplexity, semantic similarity)?")
        
        results = {}
        benchmark_results = []  # Store individual results for enhanced processing
        enhanced_metrics_calc = EnhancedQualityMetrics() if enhanced else None
        
        for api_name in selected_apis:
            print_section(f"🔧 Configuring {api_name}")
            
            available_models = API_MODELS[api_name]
            selected_models = get_multi_selection_input(available_models, f"models for {api_name}")
            
            if not selected_models:
                print_warning(f"No models selected for {api_name}. Skipping...")
                continue
            
            num_iterations = get_iterations_input(api_name)
            if num_iterations is None:
                continue
            
            benchmark = BENCHMARKS[api_name]
            
            for model in selected_models:
                # Display model info in a panel
                model_panel = Panel(
                    f"🤖 [bold cyan]Model:[/bold cyan] {model}\n"
                    f"🔧 [bold cyan]Provider:[/bold cyan] {api_name}\n"
                    f"🔄 [bold cyan]Iterations:[/bold cyan] {num_iterations}\n"
                    f"📝 [bold cyan]Queries:[/bold cyan] {len(QUERIES_AND_REFERENCES)}",
                    title="🧪 Running Benchmark",
                    border_style="green"
                )
                console.print(model_panel)
                
                total_time = 0
                total_metrics = {
                    'bleu': 0,
                    'rouge-1': 0,
                    'rouge-2': 0,
                    'rouge-l': 0
                }
                
                total_queries = num_iterations * len(QUERIES_AND_REFERENCES)
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    
                    task = progress.add_task(
                        f"🚀 Testing {model}...", 
                        total=total_queries
                    )
                    
                    for iteration in range(num_iterations):
                        for query_ref in QUERIES_AND_REFERENCES:
                            query = query_ref["query"]
                            reference = query_ref["reference"]
                            
                            latency, response = benchmark.run(query, model, MAX_TOKENS)
                            
                            if latency is not None and response is not None:
                                total_time += latency
                                metrics = calculate_quality_metrics(reference, response)
                                
                                for key, value in metrics.items():
                                    total_metrics[key] += value
                                
                                # Store individual result for enhanced processing
                                result = {
                                    'query': query,
                                    'reference': reference,
                                    'response': response,
                                    'api_name': api_name,
                                    'model_name': model,
                                    'latency': latency,
                                    'iteration': iteration,
                                    'basic_metrics': metrics
                                }
                                
                                # Add enhanced metrics if requested
                                if enhanced and enhanced_metrics_calc:
                                    try:
                                        enhanced_metrics = enhanced_metrics_calc.calculate_all_metrics(reference, response)
                                        result['enhanced_metrics'] = enhanced_metrics
                                    except Exception as e:
                                        logger.error(f"Error calculating enhanced metrics: {e}")
                                        result['enhanced_metrics'] = {}
                                
                                benchmark_results.append(result)
                            
                            progress.update(task, advance=1)
                
                if total_time > 0:
                    avg_time = total_time / total_queries
                    avg_metrics = {k: v / total_queries for k, v in total_metrics.items()}
                    results[(api_name, model)] = (avg_time, avg_metrics)
                    
                    # Display completion in a panel
                    completion_panel = Panel(
                        f"✅ [green]Benchmark Complete![/green]\n\n"
                        f"⏱️  [bold]Average Response Time:[/bold] {avg_time:.4f}s\n"
                        f"🎯 [bold]BLEU Score:[/bold] {avg_metrics['bleu']:.4f}\n"
                        f"📝 [bold]ROUGE-1 Score:[/bold] {avg_metrics['rouge-1']:.4f}",
                        title=f"📊 {model} Results",
                        border_style="green"
                    )
                    console.print(completion_panel)
        
        # Display results
        display_beautiful_results(results)
        
        # Display enhanced results if requested
        if enhanced and benchmark_results:
            display_enhanced_results(benchmark_results)
        
        if output:
            # Save results to file
            import json
            save_data = {
                'summary_results': results,
                'detailed_results': benchmark_results,
                'enhanced_metrics_included': enhanced
            }
            with open(output, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            print_success(f"Results saved to {output}")
    
    else:
        # Non-interactive mode (for automation)
        print_error("Non-interactive mode not yet implemented")
        return

@app.command()
def list_providers():
    """📋 List all available API providers and their models."""
    print_header("📋 Available API Providers")
    
    for api_name, models in API_MODELS.items():
        table = create_selection_table(f"🔧 {api_name} Models:", models, show_numbers=False)
        console.print(table)
        console.print()

@app.command()
def info():
    """ℹ️ Show information about the benchmarking tool."""
    print_header("ℹ️ LLM API Benchmarking Tool")
    
    info_text = """
    This tool helps you benchmark and compare different LLM API providers across:
    
    📊 **Performance Metrics:**
    • Response time (latency)
    • Throughput analysis
    
    🎯 **Quality Metrics:**
    • BLEU score (translation quality)
    • ROUGE-1, ROUGE-2, ROUGE-L (summarization quality)
    
    🔧 **Supported Providers:**
    • OpenAI (GPT models)
    • Azure OpenAI
    • Anthropic (Claude models)
    • AWS Bedrock
    • Hugging Face
    • Local OpenAI-compatible servers
    
    🚀 **Features:**
    • Interactive CLI with beautiful progress tracking
    • Multiple iteration support for statistical accuracy
    • Comprehensive results visualization
    • Export results to JSON
    """
    
    console.print(Panel(info_text, title="🔍 Tool Information", border_style="cyan"))

@app.command()
def interactive():
    """
    🎯 Run the interactive enhanced benchmarking tool.
    
    Features: Standard benchmarks, Enhanced metrics, Human evaluation, A/B testing
    """
    load_dotenv()
    
    print_header(
        "🧪 Enhanced LLM Benchmarking Tool",
        "Advanced metrics, human evaluation, and A/B testing capabilities"
    )
    
    tool = EnhancedBenchmarkingTool()
    
    menu_options = [
        "🚀 Run benchmarks (basic metrics)",
        "⚡ Run benchmarks (with enhanced metrics)",
        "👥 Setup human evaluation",
        "🔍 Run human evaluation",
        "🆚 Setup A/B testing",
        "📊 Run A/B testing",
        "💾 Export results",
        "🚪 Exit"
    ]
    
    from utils import show_menu, get_user_input
    while True:
        choice = show_menu("Enhanced Benchmarking Tool", menu_options)
        
        if choice is None:
            break
            
        if choice == 1:
            # Run basic benchmarks using the main benchmark function
            benchmark(interactive=True, enhanced=False)
        elif choice == 2:
            # Run enhanced benchmarks using the main benchmark function  
            benchmark(interactive=True, enhanced=True)
        elif choice == 3:
            tool.setup_human_evaluation()
        elif choice == 4:
            evaluator_id = get_user_input("👤 Enter evaluator ID")
            tool.run_human_evaluation(evaluator_id)
        elif choice == 5:
            experiment_id = tool.setup_ab_testing()
            if experiment_id:
                print_success(f"🆔 Experiment ID: {experiment_id}")
        elif choice == 6:
            experiment_id = get_user_input("🆔 Enter experiment ID")
            evaluator_id = get_user_input("👤 Enter evaluator ID")
            tool.run_ab_testing(experiment_id, evaluator_id)
        elif choice == 7:
            output_dir = get_user_input("📁 Output directory", default="results")
            tool.export_results(output_dir)
        elif choice == 8:
            print_success("👋 Goodbye!")
            break


@app.command()
def setup_human_eval():
    """👥 Setup human evaluation tasks."""
    load_dotenv()
    tool = EnhancedBenchmarkingTool()
    tool.setup_human_evaluation()

@app.command()
def run_human_eval(evaluator_id: str = typer.Argument(..., help="Evaluator ID")):
    """🔍 Run human evaluation session."""
    load_dotenv()
    tool = EnhancedBenchmarkingTool()
    tool.run_human_evaluation(evaluator_id)

@app.command()
def setup_ab_test():
    """🆚 Setup A/B testing experiment."""
    load_dotenv()
    tool = EnhancedBenchmarkingTool()
    experiment_id = tool.setup_ab_testing()
    if experiment_id:
        print_success(f"🆔 Experiment ID: {experiment_id}")

@app.command()
def run_ab_test(
    experiment_id: str = typer.Argument(..., help="Experiment ID"),
    evaluator_id: str = typer.Argument(..., help="Evaluator ID")
):
    """📊 Run A/B testing session."""
    load_dotenv()
    tool = EnhancedBenchmarkingTool()
    tool.run_ab_testing(experiment_id, evaluator_id)

@app.command()
def export_results(output_dir: str = typer.Option("results", "--output", "-o", help="Output directory")):
    """💾 Export all results to files."""
    load_dotenv()
    tool = EnhancedBenchmarkingTool()
    tool.export_results(output_dir)

if __name__ == "__main__":
    app()