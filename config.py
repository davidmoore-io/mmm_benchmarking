MAX_TOKENS = 1024

API_MODELS = {
    "OpenAI": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
    "Local OpenAI": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
    "Azure OpenAI": ["gpt-35-turbo", "gpt-4", "gpt-4-32k"],
    "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-2.1"],
    "Hugging Face": ["gpt2", "gpt2-medium", "gpt2-large"],
    "AWS Bedrock": ["anthropic.claude-v2", "ai21.j2-ultra", "amazon.titan-text-express-v1"]
}

QUERIES = [
    "What is the capital of France?",
    "Explain the concept of machine learning in simple terms.",
    "What are the main differences between renewable and non-renewable energy sources?",
    "Suggest five creative ways to reduce plastic waste in everyday life.",
    "Analyze the main themes and symbolism in the novel 'Hitchhiker's Guide to the Galaxy by Douglas Adams.",
    "What are the potential benefits and drawbacks of artificial intelligence in healthcare?",
    "How can individuals and communities contribute to reducing the impact of climate change?"
]