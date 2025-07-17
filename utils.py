from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import List, Optional, Union
import time
import nltk
import os

console = Console()

def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    required_resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet'),
        ('omw-1.4', 'corpora/omw-1.4')
    ]
    
    missing_resources = []
    
    for resource_name, resource_path in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            missing_resources.append(resource_name)
    
    if missing_resources:
        print_info(f"Downloading required NLTK data: {', '.join(missing_resources)}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for resource in missing_resources:
                task = progress.add_task(f"Downloading {resource}...")
                try:
                    nltk.download(resource, quiet=True)
                    progress.update(task, description=f"✅ Downloaded {resource}")
                except Exception as e:
                    progress.update(task, description=f"❌ Failed to download {resource}: {str(e)}")
                    print_warning(f"Failed to download {resource}: {str(e)}")
        
        print_success("NLTK data setup complete!")

def safe_nltk_import():
    """Safely import NLTK resources with auto-download."""
    try:
        ensure_nltk_data()
        return True
    except Exception as e:
        print_error(f"Failed to setup NLTK data: {str(e)}")
        return False

def print_colored(message: str, color: str = "white", style: str = ""):
    """Print a colored message using Rich."""
    console.print(message, style=f"{color} {style}".strip())

def print_info(message: str):
    """Print an info message with nice formatting."""
    console.print(f"ℹ️  {message}", style="cyan")

def print_success(message: str):
    """Print a success message with nice formatting."""
    console.print(f"✅ {message}", style="green")

def print_warning(message: str):
    """Print a warning message with nice formatting."""
    console.print(f"⚠️  {message}", style="yellow")

def print_error(message: str):
    """Print an error message with nice formatting."""
    console.print(f"❌ {message}", style="red")

def print_header(title: str, subtitle: str = ""):
    """Print a beautiful header with optional subtitle."""
    if subtitle:
        console.print(Panel(f"[bold cyan]{title}[/bold cyan]\n[dim]{subtitle}[/dim]", 
                          border_style="cyan", padding=(1, 2)))
    else:
        console.print(Panel(f"[bold cyan]{title}[/bold cyan]", 
                          border_style="cyan", padding=(1, 2)))

def print_section(title: str):
    """Print a section header."""
    console.print(f"\n[bold blue]═══ {title} ═══[/bold blue]")

def get_user_input(prompt: str, options: Optional[List[str]] = None, default: str = "") -> str:
    """Get user input with optional validation against a list of options."""
    if options:
        return Prompt.ask(prompt, choices=options, default=default if default else None)
    else:
        return Prompt.ask(prompt, default=default if default else None)

def get_user_choice(prompt: str, choices: List[str]) -> str:
    """Get user choice from a list of options."""
    return Prompt.ask(prompt, choices=choices)

def get_integer_input(prompt: str, minimum: int = 1, maximum: int = None) -> int:
    """Get integer input with validation."""
    while True:
        try:
            value = IntPrompt.ask(f"[bold cyan]{prompt}[/bold cyan]")
            if value < minimum:
                print_warning(f"Value must be at least {minimum}")
                continue
            if maximum and value > maximum:
                print_warning(f"Value must be at most {maximum}")
                continue
            return value
        except KeyboardInterrupt:
            print_warning("Operation cancelled")
            return None

def get_iterations_input(api_name: str, default: int = 3) -> int:
    """Get number of iterations with beautiful prompt."""
    iterations_panel = Panel(
        f"⚡ [bold cyan]Configure Test Iterations for {api_name}[/bold cyan]\n\n"
        f"• More iterations = more accurate results\n"
        f"• Recommended: 3-5 iterations\n"
        f"• Each iteration tests all queries",
        title="🔧 Iteration Settings",
        border_style="yellow"
    )
    console.print(iterations_panel)
    
    return get_integer_input(f"⚡ How many iterations for {api_name}?", minimum=1, maximum=10)

def get_confirmation(prompt: str, default: bool = True) -> bool:
    """Get yes/no confirmation from user."""
    return Confirm.ask(prompt, default=default)

def parse_comma_separated_input(input_string: str) -> Optional[List[int]]:
    """Parse a comma-separated string into a list of integers."""
    try:
        return [int(num.strip()) for num in input_string.split(",")]
    except ValueError:
        print_warning("Invalid input. Please enter comma-separated numbers.")
        return None

def create_selection_table(title: str, items: List[str], 
                          show_numbers: bool = True) -> Table:
    """Create a beautiful selection table."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Option", style="cyan", width=8)
    table.add_column("Name", style="white")
    
    for i, item in enumerate(items, 1):
        if show_numbers:
            table.add_row(str(i), item)
        else:
            table.add_row("", item)
    
    return table

def create_results_table(title: str, headers: List[str]) -> Table:
    """Create a results table with proper styling."""
    table = Table(title=title, show_header=True, header_style="bold green")
    
    for header in headers:
        table.add_column(header, style="white")
    
    return table

def display_progress(description: str, total: int = None):
    """Create a progress display context manager."""
    if total:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        )
    else:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        )

def show_menu(title: str, options: List[str]) -> int:
    """Display a beautiful menu and get user selection."""
    print_header(title)
    
    table = create_selection_table("Select an option:", options)
    console.print(table)
    console.print()
    
    while True:
        try:
            choice = IntPrompt.ask("[bold cyan]Enter your choice[/bold cyan]", 
                                 choices=[str(i) for i in range(1, len(options) + 1)])
            return choice
        except KeyboardInterrupt:
            print_warning("Operation cancelled")
            return None

def get_multi_selection_input(items: List[str], item_type: str, 
                             default_selection: str = "1") -> List[str]:
    """Get multiple selections from user with beautiful interface and validation."""
    # Display options in a beautiful table
    table = create_selection_table(f"Available {item_type}:", items)
    console.print(table)
    console.print()
    
    # Create a panel with instructions
    instructions = Panel(
        f"💡 [bold cyan]Select {item_type}[/bold cyan]\n\n"
        f"• Enter numbers separated by commas (e.g., 1,3,5)\n"
        f"• Or enter a single number (e.g., 1)\n"
        f"• Press Enter for default: {default_selection}",
        title="🎯 Selection Instructions",
        border_style="blue"
    )
    console.print(instructions)
    
    while True:
        selections = Prompt.ask(
            f"[bold green]🔢 Select {item_type}[/bold green]",
            default=default_selection
        )
        
        try:
            numbers = [int(num.strip()) for num in selections.split(",")]
            if all(0 < num <= len(items) for num in numbers):
                selected_items = [items[num - 1] for num in numbers]
                
                # Show confirmation
                selected_panel = Panel(
                    f"✅ [green]Selected {item_type}:[/green]\n" + 
                    "\n".join([f"• {item}" for item in selected_items]),
                    title="📋 Your Selection",
                    border_style="green"
                )
                console.print(selected_panel)
                
                return selected_items
            else:
                print_warning(f"Please enter numbers between 1 and {len(items)}")
        except ValueError:
            print_warning("Please enter valid numbers separated by commas")

def display_key_value_pairs(data: dict, title: str = "Details"):
    """Display key-value pairs in a nice format."""
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="white")
    
    for key, value in data.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))
    
    console.print(table)

def get_time_color(avg_time: float) -> str:
    """Get color for response time display."""
    if avg_time < 0.5:
        return "green"
    elif avg_time < 1.0:
        return "yellow"
    else:
        return "red"

def get_metric_color(score: float) -> str:
    """Get color for metric score display."""
    if score > 0.5:
        return "green"
    elif score > 0.3:
        return "yellow"
    else:
        return "red"

def simulate_typing(text: str, delay: float = 0.02):
    """Simulate typing effect for dramatic presentation."""
    for char in text:
        console.print(char, end="", style="cyan")
        time.sleep(delay)
    console.print()  # New line at the end