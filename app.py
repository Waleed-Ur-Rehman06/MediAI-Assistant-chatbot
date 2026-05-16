from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from src.config import config
import os
import re
import traceback
from typing import Dict, Any

app = Flask(__name__)
load_dotenv()

# ===== Configuration =====
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
index_name = "medical-chatbot"
MODEL_PATH = "model/llama-2-7b-chat.ggmlv3.q4_0.bin"


# ===== Enhanced Initialization =====
def initialize_components() -> Dict[str, Any]:
    """
    Validate and initialize all core components of the application.
    This includes Pinecone, embeddings, and the LLM.
    """
    print("\n" + "="*50)
    print("Initializing MediAI Medical Chatbot")
    print("="*50)
    
    components = {}
    
    try:
        from src.helper import download_hugging_face_embeddings, initialize_pinecone
        from langchain_groq import ChatGroq

        if not PINECONE_API_KEY:
            raise RuntimeError("PINECONE_API_KEY is missing from the environment.")

        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is missing from the environment.")

        # 1. Initialize Pinecone
        pc = initialize_pinecone(index_name, PINECONE_API_KEY)
        index = pc.Index(index_name)
        components['pinecone'] = pc
        
        # Validate index readiness
        index_stats = index.describe_index_stats()
        print(f"Pinecone index '{index_name}' ready (Dimension: {index_stats.dimension}, Vectors: {index_stats.total_vector_count})")
        
        # 2. Initialize Embeddings
        components['embeddings'] = download_hugging_face_embeddings()
        print("Embeddings model loaded")
        
        # 3. Initialize High-Speed Cloud LLM via Groq (Sub-second responses)
        components['llm'] = ChatGroq(
            temperature=config.TEMPERATURE,
            groq_api_key=config.GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",   # Using the active supported Groq model
            max_tokens=config.MAX_NEW_TOKENS
        )
        
        print("Groq Cloud LLM loaded successfully.")
        
        return components
        
    except Exception as e:
        print(f"[Initialization Failed] {str(e)}")
        # Remove sys.exit(1) so Vercel doesn't crash on cold start
        raise e

# Lazy initialization globals
qa_chain = None

def get_qa_chain():
    from langchain_community.vectorstores import Pinecone as PineconeVectorStore
    from langchain.chains import RetrievalQA
    from src.prompt import get_prompt_template
    import importlib

    global qa_chain
    if qa_chain is not None:
        return qa_chain
    components = initialize_components()

    # Compatibility shim: some versions of the `pinecone` package expose Index
    # differently. Ensure the module has an `Index` attribute so
    # langchain_community.vectorstores.Pinecone can perform isinstance checks.
    try:
        pc = components.get('pinecone')
        if pc is not None:
            index_obj = pc.Index(index_name)
            pinecone_module = importlib.import_module('pinecone')
            if not hasattr(pinecone_module, 'Index'):
                setattr(pinecone_module, 'Index', type(index_obj))
    except Exception:
        # Best-effort shim; if it fails we continue and let the higher-level
        # error handling report a useful message.
        pass

    # ===== Enhanced QA Pipeline =====
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=components['embeddings']
    )

    # get_prompt_template() now directly returns a PromptTemplate object
    PROMPT = get_prompt_template()

    retriever = docsearch.as_retriever(
        search_type="similarity", # Changed to standard similarity for fast execution
        search_kwargs={
            'k': 1 # Limit down to 1 chunk. Less text for CPU to read = astronomically faster!
        }
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=components['llm'],
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# ===== Advanced Medical Filter =====
MEDICAL_KEYWORDS = [
    'pain', 'fever', 'headache', 'nausea', 'vomit', 'rash', 'swelling', 'bleeding',
    'dizziness', 'fatigue', 'cough', 'sneeze', 'sore throat', 'shortness of breath',
    'diabetes', 'cancer', 'hypertension', 'asthma', 'arthritis', 'alzheimer', 
    'parkinson', 'stroke', 'heart attack', 'pneumonia', 'bronchitis',
    'medicine', 'medication', 'treatment', 'therapy', 'surgery', 'vaccine',
    'antibiotic', 'antiviral', 'chemotherapy', 'radiation',
    'heart', 'lung', 'liver', 'kidney', 'brain', 'stomach', 'intestine',
    'health', 'medical', 'diagnosis', 'symptom', 'patient', 'doctor', 'hospital',
    'clinic', 'pharmacy', 'prescription', 'dosage', 'side effect'
]

MEDICAL_PREFIXES = [
    'what is', 'how to treat', 'symptoms of', 'causes of', 'diagnosis of',
    'treatment for', 'medication for', 'side effects of', 'recovery from',
    'prevention of', 'risk factors for', 'prognosis of', 'complications of'
]

def is_medical_query(query: str) -> bool:
    """Advanced medical query detection with regex patterns."""
    query = query.lower().strip()
    if any(re.search(rf'\b{keyword}\b', query) for keyword in MEDICAL_KEYWORDS):
        return True
    if any(query.startswith(prefix) for prefix in MEDICAL_PREFIXES):
        return True
    if re.match(r'what (causes|triggers) .+', query) or re.match(r'how to (prevent|avoid) .+', query):
        return True
    return False

def format_response(response: str) -> str:
    """Format the LLM response for better readability and enforce disclaimer at the absolute end."""
    response = "" if response is None else str(response)
    response = response.replace('Medical Answer:', '').replace('Safe Medical Answer:', '').strip()
    
    # Strip any disclaimers the LLM might have hallucinated in the middle
    response = re.sub(r'(?i)\*?\*?disclaimer:\*?\*?.*', '', response, flags=re.DOTALL).strip()
    
    disclaimer = "\n\n**Disclaimer:** This information is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a licensed doctor."
    
    return response + disclaimer

# ===== Routes =====
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    try:
        user_input = request.form.get("msg", "").strip()
        if not user_input:
            return jsonify({"error": "Empty input"}), 400
            
        print(f"\n[User Query]: '{user_input}'")
        
        if not is_medical_query(user_input):
            print("[Filtered] Non-medical question")
            return "I specialize in medical topics. Please ask health-related questions about symptoms, conditions, or treatments."
        
        # Initialize lazily to prevent Vercel 10s cold-start crash
        chain = get_qa_chain()
        result = chain.invoke({"query": user_input})
        
        if not result.get("result"):
            print("[Error] Empty response from QA chain")
            return "I couldn't generate a response. Please try rephrasing your medical question."
            
        formatted_response = format_response(result["result"])
        print(f"[Response Sent] {formatted_response[:200].strip()}...")
        
        return formatted_response
            
    except Exception as e:
        print(f"[Critical Error] {traceback.format_exc()}")
        return "I am experiencing technical difficulties. Please try again later.", 503

if __name__ == '__main__':
    print("\nStarting MediAI server...")
    app.run(host="0.0.0.0", port=8080, debug=False)