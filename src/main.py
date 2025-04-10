from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.embeddings import EmbeddingsManager
from src.llm import LLaMAHandler
from src.data_loader import StripeDocsLoader
import logging
import traceback

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stripe API Documentation QA System",
    description="A system that allows asking questions about Stripe API documentation using LLaMA model",
    version="1.0.0"
)

embeddings_manager = EmbeddingsManager()
llm_handler = LLaMAHandler()

class Query(BaseModel):
    text: str
    
    class Config:
        schema_extra = {
            "example": {
                "text": "How do I create a payment with Stripe API?"
            }
        }

@app.post("/ask", 
    response_model=dict,
    summary="Ask a question about Stripe API",
    description="Send a question about Stripe API and get an answer based on the documentation"
)
async def ask_question(query: Query):
    try:
        logger.info(f"Processing query: {query.text}")
        
        # Get relevant context
        logger.info("Searching for similar texts...")
        similar_texts = embeddings_manager.search(query.text)
        logger.info(f"Found {len(similar_texts)} similar texts")
        
        # Generate prompt and get response
        logger.info("Generating prompt...")
        prompt = llm_handler.generate_prompt(query.text, similar_texts)
        logger.info("Getting response from LLaMA...")
        response = llm_handler.get_response(prompt)
        logger.info("Got response from LLaMA")
        
        return {"response": response}
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", 
    summary="Root endpoint",
    description="Redirects to API documentation"
)
async def root():
    return {"message": "Welcome to Stripe API QA System. Visit /docs for API documentation."}

@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Starting up application...")
        # Try to load existing embeddings
        logger.info("Attempting to load existing embeddings...")
        embeddings_manager.load()
        logger.info("Successfully loaded existing embeddings")
    except Exception as e:
        logger.warning(f"Failed to load embeddings: {str(e)}")
        logger.info("Creating new embeddings...")
        try:
            loader = StripeDocsLoader()
            logger.info("Fetching docs...")
            documents = loader.fetch_docs()
            logger.info(f"Fetched {len(documents)} documents")
            logger.info("Splitting documents...")
            chunks = loader.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            logger.info("Creating embeddings...")
            embeddings_manager.create_embeddings(chunks)
            logger.info("Saving embeddings...")
            embeddings_manager.save()
            logger.info("Embeddings created and saved successfully")
        except Exception as e:
            logger.error(f"Failed to create embeddings: {str(e)}")
            logger.error(traceback.format_exc())
            raise e

