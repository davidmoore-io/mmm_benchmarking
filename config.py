MAX_TOKENS = 1024

API_MODELS = {
    "OpenAI": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
    "Local OpenAI": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
    "Azure OpenAI": ["gpt-35-turbo", "gpt-4", "gpt-4-32k"],
    "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-2.1"],
    "Hugging Face": ["gpt2", "gpt2-medium", "gpt2-large"],
    "AWS Bedrock": ["anthropic.claude-v2", "ai21.j2-ultra", "amazon.titan-text-express-v1"]
}

QUERIES_AND_REFERENCES = [
    {
        "query": "What is the capital of France?",
        "reference": "The capital of France is Paris. Paris is the largest city in France and serves as the country's political, economic, and cultural center."
    },
    {
        "query": "Explain the concept of machine learning in simple terms.",
        "reference": "Machine learning is a type of artificial intelligence that allows computers to learn and improve from experience without being explicitly programmed. It involves feeding large amounts of data into algorithms that can then make predictions or decisions based on new, unseen data."
    },
    {
        "query": "What are the main differences between renewable and non-renewable energy sources?",
        "reference": "Renewable energy sources, such as solar, wind, and hydroelectric power, are replenished naturally and can be used indefinitely. Non-renewable energy sources, like fossil fuels (coal, oil, natural gas), are finite and will eventually be depleted. Renewable sources are generally cleaner and have less environmental impact, while non-renewable sources often produce more pollution and contribute to climate change."
    },
    {
        "query": "Suggest five creative ways to reduce plastic waste in everyday life.",
        "reference": "1. Use reusable cloth bags for grocery shopping. 2. Carry a refillable water bottle instead of buying bottled water. 3. Choose products with minimal or biodegradable packaging. 4. Use beeswax wraps instead of plastic wrap for food storage. 5. Bring your own containers for takeout food and leftovers at restaurants."
    },
    {
        "query": "Analyze the main themes and symbolism in the novel 'To Kill a Mockingbird' by Harper Lee.",
        "reference": "The main themes in 'To Kill a Mockingbird' include racial injustice, the loss of innocence, and the importance of moral education. The mockingbird symbolizes innocence and the idea that it's wrong to harm those who are vulnerable or defenseless. Atticus Finch represents moral integrity and justice, while the trial of Tom Robinson highlights the racial prejudices of the time."
    },
    {
        "query": "What are the potential benefits and drawbacks of artificial intelligence in healthcare?",
        "reference": "Benefits of AI in healthcare include improved diagnostic accuracy, personalized treatment plans, and more efficient patient care. AI can analyze large amounts of medical data quickly, potentially leading to new medical discoveries. Drawbacks include concerns about data privacy, the potential for bias in AI algorithms, and the risk of over-reliance on technology at the expense of human judgment and empathy in patient care."
    },
    {
        "query": "How can individuals and communities contribute to reducing the impact of climate change?",
        "reference": "Individuals can reduce their carbon footprint by using energy-efficient appliances, reducing meat consumption, using public transportation or carpooling, and recycling. Communities can implement renewable energy projects, create green spaces, promote sustainable urban planning, and educate residents about climate change. Both can advocate for policies that address climate change at local and national levels."
    }
]