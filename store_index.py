from src.helper import (
    load_pdf_data,
    text_split,
    download_hugging_face_embeddings,
    initialize_pinecone
)
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from dotenv import load_dotenv
import os
from tqdm import tqdm
import time
import sys

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
index_name = "medical-chatbot"
PDF_PATH = "data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"

def process_and_store_documents():
    """Process PDF documents, create embeddings, and store them in Pinecone."""
    print("\nStarting document processing...")
    
    try:
        # Initialize Pinecone
        pc = initialize_pinecone(index_name, PINECONE_API_KEY)
        
        # Load and process documents
        print("Loading PDF documents...")
        documents = load_pdf_data(PDF_PATH)
        
        print("Splitting text into chunks...")
        text_chunks = text_split(documents)
        print(f"Created {len(text_chunks)} text chunks")
        
        # Download embeddings
        print("Downloading embeddings model...")
        embeddings = download_hugging_face_embeddings()
        
        # Store embeddings in batches
        print("Storing embeddings in Pinecone...")
        
        # We can use `from_documents` for a more straightforward approach
        # It handles the creation and upserting in batches internally.
        PineconeVectorStore.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            index_name=index_name
        )
        
        print("\nEmbeddings stored successfully")
        index_stats = pc.Index(index_name).describe_index_stats()
        print(f"Total vectors: {index_stats.total_vector_count}")
        
    except Exception as e:
        print(f"[ERROR] An error occurred during document processing: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    process_and_store_documents()