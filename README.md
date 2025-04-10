# Stripe API Documentation QA System

A system that allows asking questions about Stripe API documentation using a local LLaMA model with RAG approach.

## Components Used

- LLM: LLaMA2 (via Ollama)
- Vector Database: FAISS
- Embeddings: Sentence Transformers (all-MiniLM-L6-v2)
- API: FastAPI
- Other: langchain, beautifulsoup4

## Setup and Running

1. Install Ollama and download LLaMA2:
```bash
curl https://ollama.ai/install.sh | sh
ollama pull llama2
```

2. Build and run the Docker container:
```bash
docker build -t stripe-docs-qa .
docker run -p 8000:8000 --network host stripe-docs-qa
```

The service will be available at http://localhost:8000

## How It Works

1. Documentation Processing:
   - Fetches Stripe API documentation from specified URLs
   - Splits text into chunks of ~250 words

2. Embedding Creation:
   - Creates embeddings using Sentence Transformers
   - Stores embeddings in FAISS vector database

3. Question Answering:
   - Converts question to embedding
   - Finds relevant documentation chunks
   - Creates prompt with context
   - Gets response from LLaMA model

## Example Questions

1. "How do I create a payment with Stripe API?"
2. "What is a Stripe Customer object?"

## API Usage

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"text": "How do I create a payment with Stripe API?"}'
```
