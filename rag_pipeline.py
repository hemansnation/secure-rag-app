import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL 1.1.1+")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from utils import load_document, decrypt_text, chunk_text, logging
from dotenv import load_dotenv
import os

load_dotenv()

# Gemini LLM (use gemini-1.5-flash or available free model)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Or "gemini-1.5-pro" if available in free tier
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.3
)

# Embeddings: Local HuggingFace model (no API limits)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

VECTOR_STORE_DIR = "./chroma_db"

def ingest_document(file_path):
    encrypted_text = load_document(file_path)
    text = decrypt_text(encrypted_text)
    docs = chunk_text(text)  # Now returns list of Document
    
    vector_store = Chroma.from_documents(docs, embeddings, persist_directory=VECTOR_STORE_DIR)
    vector_store.persist()
    logging.info(f"Ingested {file_path} with {len(docs)} chunks")
    return vector_store

# Proper LangChain RAG chain for citations
retriever = None  # Will set globally or pass

prompt = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context.
    Cite sources strictly using the chunk IDs provided in the context (e.g., [Chunk 1], [Chunk 2]) at the end of relevant sentences.

    Context:
    {context}

    Question: {question}
    """
)

def format_docs(docs):
    return "\n\n".join(
        f"Chunk {doc.metadata['chunk_id']}: {doc.page_content}"
        for doc in docs
    )

def query_rag(query):
    global retriever
    if retriever is None:
        vector_store = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})  # Top 4 chunks

    # Create chain dynamically when retriever is available
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(query)
    logging.info(f"Query: {query} | Response: {response}")
    return response


