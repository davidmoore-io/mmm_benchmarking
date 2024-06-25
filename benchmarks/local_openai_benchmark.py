import os
from openai import OpenAI
from .base import BaseBenchmark

class LocalOpenAIBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("Local OpenAI")

    def setup_client(self):
        base_url = os.getenv("LOCAL_OPENAI_BASE_URL", "http://localhost:1234/v1")
        api_key = os.getenv("LOCAL_OPENAI_API_KEY", "sk-111111111111111111111111111111111111111111111111")
        return OpenAI(base_url=base_url, api_key=api_key)

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