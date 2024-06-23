from colorama import Fore, Style

def print_colored(message, color=Fore.WHITE):
    """Print a colored message using colorama."""
    print(color + message + Style.RESET_ALL)

def get_user_input(prompt, options=None):
    """Get user input with optional validation against a list of options."""
    while True:
        user_input = input(prompt).strip()
        if options is None or user_input in options:
            return user_input
        print_colored(f"Invalid input. Please choose from: {', '.join(options)}", Fore.YELLOW)

def parse_comma_separated_input(input_string):
    """Parse a comma-separated string into a list of integers."""
    try:
        return [int(num.strip()) for num in input_string.split(",")]
    except ValueError:
        print_colored("Invalid input. Please enter comma-separated numbers.", Fore.YELLOW)
        return None