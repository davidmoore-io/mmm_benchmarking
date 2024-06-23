import os
import anthropic
from .base import BaseBenchmark

class AnthropicBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("Anthropic")

    def setup_client(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided.")
        return anthropic.Anthropic(api_key=api_key)

    def invoke_model(self, client, query, model, max_tokens):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": query}]
        )
        return response

    def extract_output(self, response):
        return response.content[0].text.strip()