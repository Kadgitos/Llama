import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import requests
from src.config import settings

class StripeDocsLoader:
    def __init__(self):
        self.base_urls = [
            "https://stripe.com/docs/api/charges",
            "https://stripe.com/docs/api/customers",
            "https://stripe.com/docs/api/payment_intents",
            "https://stripe.com/docs/api/refunds",
            "https://stripe.com/docs/api/payment_methods"
        ]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    def fetch_docs(self) -> List[str]:
        """Fetch documentation from Stripe's website"""
        documents = []
        for url in self.base_urls:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract main content
                content = soup.find('main')
                if content:
                    documents.append(content.get_text())
        return documents

    def split_documents(self, documents: List[str]) -> List[str]:
        """Split documents into chunks"""
        chunks = []
        for doc in documents:
            chunks.extend(self.text_splitter.split_text(doc))
        return chunks