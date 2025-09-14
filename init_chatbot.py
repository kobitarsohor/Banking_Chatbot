import os
import pandas as pd
from llama_index.core.schema import Document
from pinecone import Pinecone, ServerlessSpec
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.groq import Groq
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Load dataset
df = pd.read_csv("Dataset_Banking_chatbot.csv", encoding="cp1252")

documents = [Document(text=f"Query: {row['Query']}\nResponse: {row['Response']}") for _, row in df.iterrows()]

# Setup Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "banking-rag"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

pinecone_index = pc.Index(index_name)

# Setup LLM & embeddings
Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
Settings.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

retriever = index.as_retriever(similarity_top_k=5)
query_engine = RetrieverQueryEngine(retriever=retriever)

def get_bot_response(query: str) -> str:
    response = query_engine.query(query)
    return str(response)