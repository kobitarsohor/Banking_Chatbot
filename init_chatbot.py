import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine

# Load env vars
load_dotenv()

# Setup Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "banking-rag"
pinecone_index = pc.Index(index_name)

# Setup LLM & embeddings
Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# Connect Pinecone
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# Setup retriever & query engine
retriever = index.as_retriever(similarity_top_k=5)
query_engine = RetrieverQueryEngine(retriever=retriever)

def get_bot_response(query: str) -> str:
    response = query_engine.query(query)
    return str(response)
