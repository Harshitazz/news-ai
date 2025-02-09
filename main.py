from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import os
import pickle
import uuid
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
import uvicorn
from celery import Celery
import redis

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize FastAPI app
app = FastAPI()

# Configure Celery to Use Redis
celery = Celery(__name__, broker=redis_url, backend=redis_url)

# Redis client for task tracking
redis_client = redis.Redis.from_url(redis_url, decode_responses=True)

file_path = "faiss_store_groq.pkl"

# Define Pydantic model for request body validation
class UrlInput(BaseModel):
    urls: list[str]

# Initialize LLM using Groq
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key, temperature=0.9, max_tokens=1024)

@app.get("/")
def hello():
    return {"message": "Hello, World!"}

@celery.task
def process_faiss(task_id: str, urls: list[str]):
    try:
        redis_client.set(task_id, "Processing")
        
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(data)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore_groq = FAISS.from_documents(docs, embeddings)

        # Save FAISS index
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_groq, f)

        redis_client.set(task_id, "Completed")
    except Exception as e:
        redis_client.set(task_id, f"Failed: {str(e)}")

@app.post("/initialize_faiss")
async def initialize_faiss(request: UrlInput):
    if not request.urls:
        raise HTTPException(status_code=400, detail="No URLs provided.")

    task_id = str(uuid.uuid4())
    redis_client.set(task_id, "Pending")
    process_faiss.apply_async(args=[task_id, request.urls])  # Send to Celery Worker

    return {"message": "FAISS initialization started in the background.", "task_id": task_id}

@app.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    status = redis_client.get(task_id)
    return {"task_id": task_id, "status": status if status else "Task ID not found"}

@app.post("/ask")
def ask(question: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=500, detail="FAISS index not initialized. Please initialize first.")

    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    result = chain.invoke({"question": question})

    return {"answer": result["answer"], "sources": result.get("sources", "")}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
