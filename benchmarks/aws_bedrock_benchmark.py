import os
import json
import boto3
from colorama import Fore, Style
from .base import BaseBenchmark

class AWSBedrockBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__("AWS Bedrock")

    def setup_client(self):
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION")
        
        if not aws_access_key_id or not aws_secret_access_key or not aws_region:
            raise ValueError("AWS credentials not provided or invalid.")
        
        return boto3.client(
            "bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )

    def invoke_model(self, client, query, model, max_tokens):
        body = json.dumps({
            "prompt": query,
            "max_tokens_to_sample": max_tokens,
            "temperature": 0.7,
            "top_p": 1,
            "top_k": 250,
            "stop_sequences": []
        })
        
        response = client.invoke_model(
            modelId=model,
            body=body
        )
        return json.loads(response['body'].read())

    def extract_output(self, response):
        return response.get('completion', '').strip()