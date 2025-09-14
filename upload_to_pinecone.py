import os
import pandas as pd
from dotenv import load_dotenv
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, Settings
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Load dataset
df = pd.read_csv("Dataset_Banking_chatbot.csv", encoding="cp1252")

documents = [
    Document(text=f"Query: {row['Query']}\nResponse: {row['Response']}")
    for _, row in df.iterrows()
]

# Setup Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "banking-rag"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,   # ✅ better match for MiniLM embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(index_name)

# Setup embeddings (lighter to fit memory)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)

# Connect Pinecone to LlamaIndex
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# Upload all docs into Pinecone
index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

print("✅ All documents uploaded into Pinecone successfully!")
