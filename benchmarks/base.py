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
    def invoke_model(self, client, query, model, max_tokens):
        pass

    @abstractmethod
    def extract_output(self, response):
        pass

    def run(self, query, model, max_tokens):
        print(Fore.CYAN + f"Testing {self.api_name} with model {model}..." + Style.RESET_ALL)
        
        try:
            client = self.setup_client()
            start_time = time.time()
            response = self.invoke_model(client, query, model, max_tokens)
            end_time = time.time()
            
            latency = end_time - start_time
            output = self.extract_output(response)
            
            print(f"Response: {output[:100]}...")  # Print first 100 characters of the response
            return latency, output
        except Exception as e:
            print(Fore.RED + f"Error benchmarking {self.api_name}: {str(e)}" + Style.RESET_ALL)
            return None, None