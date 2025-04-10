import requests
from typing import List, Tuple
from src.config import settings

class LLaMAHandler:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.MODEL_NAME

    def generate_prompt(self, query: str, context: List[Tuple[str, float]]) -> str:
        context_str = "\n".join([text for text, _ in context])
        return f"""You are a helpful assistant that answers questions about Stripe API.
Use the following context to answer the question. If you cannot answer the question based on the context, say so.

Context:
{context_str}

Question: {query}

Answer:"""

    def get_response(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        return response.json()["response"]