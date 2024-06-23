import os
from openai import OpenAI
from .base import BaseBenchmark

class OpenAIBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("OpenAI")

    def setup_client(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided.")
        return OpenAI(api_key=api_key)

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

    def extract_output(self, response):
        return response.choices[0].message.content.strip()