import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import parse_comma_separated_input, get_user_input, print_colored


class TestUtils(unittest.TestCase):
    
    def test_parse_comma_separated_input_valid_numbers(self):
        """Test parsing valid comma-separated numbers."""
        result = parse_comma_separated_input("1,2,3")
        self.assertEqual(result, [1, 2, 3])
        
    def test_parse_comma_separated_input_with_spaces(self):
        """Test parsing numbers with spaces."""
        result = parse_comma_separated_input("1, 2 , 3")
        self.assertEqual(result, [1, 2, 3])
        
    def test_parse_comma_separated_input_single_number(self):
        """Test parsing single number."""
        result = parse_comma_separated_input("5")
        self.assertEqual(result, [5])
        
    def test_parse_comma_separated_input_invalid_input(self):
        """Test parsing invalid input returns empty list."""
        result = parse_comma_separated_input("abc,def")
        self.assertEqual(result, [])
        
    def test_parse_comma_separated_input_mixed_valid_invalid(self):
        """Test parsing mixed valid and invalid input."""
        result = parse_comma_separated_input("1,abc,3")
        self.assertEqual(result, [1, 3])
        
    def test_parse_comma_separated_input_empty_string(self):
        """Test parsing empty string."""
        result = parse_comma_separated_input("")
        self.assertEqual(result, [])
        
    @patch('builtins.input')
    def test_get_user_input_returns_input(self, mock_input):
        """Test that get_user_input returns user input."""
        mock_input.return_value = "test input"
        result = get_user_input("Enter something: ")
        self.assertEqual(result, "test input")
        mock_input.assert_called_once_with("Enter something: ")
        
    @patch('builtins.print')
    def test_print_colored_calls_print(self, mock_print):
        """Test that print_colored calls print function."""
        from colorama import Fore
        print_colored("test message", color=Fore.GREEN)
        mock_print.assert_called()


if __name__ == '__main__':
    unittest.main()