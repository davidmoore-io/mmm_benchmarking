import os
import requests
from .base import BaseBenchmark

class HuggingFaceBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("Hugging Face")

    def setup_client(self):
        api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        if not api_token:
            raise ValueError("Hugging Face API token not provided.")
        return api_token

    def invoke_model(self, api_token, query, model, max_tokens):
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {api_token}"},
            json={"inputs": query, "parameters": {"max_length": max_tokens}}
        )
        return response.json()

    def extract_output(self, response):
        if isinstance(response, list) and len(response) > 0:
            return response[0].get('generated_text', '').strip()
        return ''