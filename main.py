from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import os
import pickle
import time
import uuid
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from dotenv import load_dotenv
import uvicorn
from fastapi import Request

from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for debugging
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly allow OPTIONS
    allow_headers=["Content-Type", "Authorization"],
)


file_path = "faiss_store_groq.pkl"
task_status = {}

# Define Pydantic model for request body validation
class UrlInput(BaseModel):
    urls: list[str]

# Initialize LLM using Groq
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key, temperature=0.9, max_tokens=1024)

@app.get("/")
def hello():
    return {"message": "Hello, World!"}

# Background task function to initialize FAISS
async def process_faiss(task_id: str, urls: list[str]):
    try:
        task_status[task_id] = "Processing"
        print("task1")
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        print("task2")
        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(data)
        print("task3")
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore_groq = FAISS.from_documents(docs, embeddings)
        print("task4")
        # Save FAISS index
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_groq, f)
        print("FAISS index initialized successfully.")
        task_status[task_id] = "Completed"
    except Exception as e:
        task_status[task_id] = f"Failed: {str(e)}"


@app.post("/initialize_faiss")
async def initialize_faiss(background_tasks: BackgroundTasks, request: Request):
    try:
        data = await request.json()  # Ensure JSON is properly parsed
        urls = data.get("urls", [])

        if not urls:
            raise HTTPException(status_code=400, detail="No URLs provided.")

        task_id = str(uuid.uuid4())
        task_status[task_id] = "Pending"
        background_tasks.add_task(process_faiss, task_id, urls)

        return {"message": "FAISS initialization started in the background.", "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    status = task_status.get(task_id, "Task ID not found")
    return {"task_id": task_id, "status": status}

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
