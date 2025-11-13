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
        
        # Ensure API key is properly encoded and contains only ASCII characters
        try:
            api_key.encode('ascii')
        except UnicodeEncodeError:
            # If API key contains non-ASCII characters, encode it properly
            api_key = api_key.encode('utf-8', errors='replace').decode('ascii', errors='replace')
        
        return OpenAI(api_key=api_key)

    def invoke_model(self, client, query, model, max_tokens):
        # Ensure query is properly encoded to avoid header encoding issues
        try:
            query.encode('ascii')
        except UnicodeEncodeError:
            # If query contains non-ASCII characters, keep as UTF-8 but ensure it's valid
            query = query.encode('utf-8', errors='replace').decode('utf-8')
        
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