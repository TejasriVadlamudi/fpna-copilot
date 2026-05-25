import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PERSIST_DIR = "chroma_db"

def build_vector_store(pdf_path: str):
    """Load PDF, chunk it, embed it, store it in ChromaDB."""
    print(f"📄 Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"   Loaded {len(pages)} pages")

    print("✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(pages)
    print(f"   Created {len(chunks)} chunks")

    print("🧠 Embedding chunks and saving to ChromaDB (this may take 30-60s)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"✅ Vector store built and saved to {PERSIST_DIR}/")
    return vectorstore

def load_vector_store():
    """Load existing vector store from disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

if __name__ == "__main__":
    build_vector_store("data/apple_10k.pdf")
