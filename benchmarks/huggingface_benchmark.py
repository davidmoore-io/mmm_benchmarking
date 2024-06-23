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
            json={"inputs": query, "max_length": max_tokens}
        )
        return response.json()