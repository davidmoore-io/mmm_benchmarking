import time
from abc import ABC, abstractmethod
from colorama import Fore, Style

class BaseBenchmark(ABC):
    def __init__(self, api_name):
        self.api_name = api_name

    @abstractmethod
    def setup_client(self):
        pass

    @abstractmethod
    def invoke_model(self, query, model, max_tokens):
        pass

    def run(self, query, model, max_tokens):
        print(Fore.CYAN + f"Testing {self.api_name} with model {model}..." + Style.RESET_ALL)
        
        try:
            client = self.setup_client()
            start_time = time.time()
            response = self.invoke_model(client, query, model, max_tokens)
            end_time = time.time()
            
            print(response)  # Access the response variable
            return end_time - start_time
        except Exception as e:
            print(Fore.RED + f"Error benchmarking {self.api_name}: {str(e)}" + Style.RESET_ALL)
            return None