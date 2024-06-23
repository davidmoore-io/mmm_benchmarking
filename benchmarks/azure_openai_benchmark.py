import os
from openai import AzureOpenAI
from .base import BaseBenchmark

class AzureOpenAIBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("Azure OpenAI")

    def setup_client(self):
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            raise ValueError("Azure OpenAI API key or endpoint not provided.")
        return AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=endpoint
        )

    def invoke_model(self, client, query, model, max_tokens):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response