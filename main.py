from fastapi import FastAPI, BackgroundTasks, HTTPException,UploadFile,Depends
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
import fitz 
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
from auth import get_current_user
from s3Upload import upload_file_to_s3, s3_client
from cleanUp import lifespan
# Load environment variables
load_dotenv()

CLERK_SECRET_KEY = os.getenv("CLERK_SECRET_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)
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
async def ask(request: Request):
    data = await request.json()
    question=data.get("question","")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=500, detail="FAISS index not initialized. Please initialize first.")

    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")


    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    result = chain.invoke({"question": question})
    

    return {"answer": result["answer"], "sources": result.get("sources", "")}

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
PDF_FAISS_PATH = "faiss_store_pdf.pkl"
# ✅ **Function to Extract Text from PDF**
def extract_text_from_pdf(pdf_path: str):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    return text



def process_pdf_faiss(task_id: str, user_id: str):
    """Downloads PDFs, generates FAISS index, and stores in S3"""
    try:
        pdf_prefix = f"pdf_uploads/{user_id}/{task_id}/"
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix=pdf_prefix)

        all_docs = []
        if "Contents" not in response:
            raise Exception("No PDFs found in S3 for processing.")

        for obj in response.get("Contents", []):
            pdf_key = obj["Key"]
            local_pdf_path = f"/tmp/{os.path.basename(pdf_key)}"

            print(f"Downloading: {pdf_key}")
            os.makedirs("/tmp", exist_ok=True)
            s3_client.download_file(S3_BUCKET_NAME, pdf_key, local_pdf_path)

            # Extract text
            text = ""
            with fitz.open(local_pdf_path) as doc:
                text = "\n".join([page.get_text("text") for page in doc])

            if not text.strip():
                print(f"Warning: No text extracted from {pdf_key}. Skipping.")
                continue

            print(f"Extracted {len(text)} characters from {pdf_key}")
            all_docs.extend(RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_text(text))

        if not all_docs:
            raise Exception("No valid text found in any PDFs. FAISS cannot be created.")

        # Generate embeddings
        print("Generating FAISS index...")
        vectorstore = FAISS.from_texts(all_docs, embedding_model)

        # Save FAISS index
        local_faiss_path = f"/tmp/{user_id}_faiss.pkl"
        with open(local_faiss_path, "wb") as f:
            pickle.dump(vectorstore, f)

        # Upload FAISS index to S3
        s3_faiss_key = f"{user_id}/faiss/faiss_store.pkl"
        s3_client.upload_file(local_faiss_path, S3_BUCKET_NAME, s3_faiss_key)
        os.remove(local_pdf_path)
        print(f"FAISS index uploaded: {s3_faiss_key}")

    except Exception as e:
        print(f"Error in process_pdf_faiss: {str(e)}")



# ✅ **API Endpoint to Upload PDFs**
@app.post("/upload_pdfs/")
async def upload_pdfs(background_tasks: BackgroundTasks, files: list[UploadFile], request_user: Dict[str, str] = Depends(get_current_user)):
    try:
        user_id = request_user["user_id"]  # Get the user ID from Clerk
        pdf_paths = []
        s3_urls = []
        print(user_id)
        task_id = str(uuid.uuid4())
        for file in files:
            if not file.filename.endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
            
            
            # Save locally
            local_pdf_path = f"uploads/{user_id}_{file.filename}"
            os.makedirs("uploads", exist_ok=True)
            
            with open(local_pdf_path, "wb") as f:
                f.write(await file.read())
            print("task0")
            # Upload to S3
            s3_pdf_key = f"pdf_uploads/{user_id}/{task_id}/{file.filename}"
            s3_url = upload_file_to_s3(local_pdf_path, s3_pdf_key)
            print("task1")
            os.remove(local_pdf_path)
            pdf_paths.append(local_pdf_path)
            s3_urls.append(s3_url)
            print("task2")

        # Start FAISS processing in the background
        
        task_status[task_id] = "Pending"
        background_tasks.add_task(process_pdf_faiss, task_id, user_id)

        return {"message": "PDF processing started", "task_id": task_id, "pdf_s3_urls": s3_urls}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



from pydantic import BaseModel

class AskPDFRequest(BaseModel):
    question: str

# ✅ **Ask a Question Based on PDF Data**
@app.post("/ask_pdf")
async def ask_pdf(request: AskPDFRequest, request_user: Dict[str, str] = Depends(get_current_user)):
    user_id = request_user["user_id"]
    data = request.dict()
    question = data.get("question", "")

    s3_faiss_key = f"{user_id}/faiss/faiss_store.pkl"
    local_faiss_path = f"/tmp/{user_id}_faiss.pkl"
    try:
        s3_client.download_file(S3_BUCKET_NAME, s3_faiss_key, local_faiss_path)
        with open(local_faiss_path, "rb") as f:
            vectorstore = pickle.load(f)
    except:
        raise HTTPException(status_code=404, detail="FAISS index not found")
    

    if not question:
        raise HTTPException(status_code=400, detail="No question provided.")
    
    retrieved_docs = vectorstore.as_retriever().get_relevant_documents(question)
    for doc in retrieved_docs:
        if "source" not in doc.metadata:
            doc.metadata["source"] = "unknown"

    # Perform retrieval-based Q&A
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    result = chain.invoke({"question": question})

    return {"answer": result["answer"]}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
