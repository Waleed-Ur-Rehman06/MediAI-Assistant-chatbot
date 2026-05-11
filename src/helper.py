import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from pinecone import Pinecone, ServerlessSpec, PodSpec
from typing import List
from langchain.schema import Document

def load_pdf_data(pdf_path: str) -> List[Document]:
    """Load and extract text from a PDF file."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The PDF file '{pdf_path}' was not found.")
        
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s) from {pdf_path}")
        return documents
    except Exception as e:
        raise Exception(f"Failed to load PDF '{pdf_path}': {str(e)}")

def text_split(extracted_data: List[Document]) -> List[Document]:
    """Split extracted documents into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"Split {len(extracted_data)} page(s) into {len(text_chunks)} chunks")
    return text_chunks

def download_hugging_face_embeddings() -> HuggingFaceInferenceAPIEmbeddings:
    """Download and initialize the HuggingFace sentence-transformer embeddings model."""
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        # Fallback to local if no token provided (useful for local dev with PyTorch installed)
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("Warning: No HF_TOKEN found in environment. Falling back to local HuggingFaceEmbeddings (requires PyTorch, will fail on Vercel).")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

def initialize_pinecone(index_name: str, api_key: str) -> Pinecone:
    """Initialize Pinecone connection and create a serverless index if it doesn't exist."""
    if not api_key:
        raise ValueError("Pinecone API key is not set. Please check your .env file.")
        
    pc = Pinecone(api_key=api_key)
    
    # Check if the index already exists
    if index_name not in pc.list_indexes().names():
        print(f"Pinecone index '{index_name}' not found. Creating a new one...")
        pc.create_index(
            name=index_name,
            dimension=768,  # Dimension of all-mpnet-base-v2
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Successfully created new Pinecone index: '{index_name}'")
    else:
        print(f"Found existing Pinecone index: '{index_name}'")
        
    return pc
