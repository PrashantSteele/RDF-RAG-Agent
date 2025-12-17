import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "libs"))

print("Importing time...")
import time
print("Importing dotenv...")
from dotenv import load_dotenv
print("Importing PyPDFLoader...")
from langchain_community.document_loaders import PyPDFLoader
print("Importing RecursiveCharacterTextSplitter...")
from langchain_text_splitters import RecursiveCharacterTextSplitter
print("Importing GoogleGenerativeAIEmbeddings...")
from langchain_huggingface import HuggingFaceEmbeddings
print("Importing PineconeVectorStore...")
from langchain_pinecone import PineconeVectorStore
print("Importing Pinecone...")
from pinecone import Pinecone, ServerlessSpec
print("Imports done.")

# Load environment variables
load_dotenv()

def create_vector_db():
    # Configuration
    pdf_path = "RDF_RIKSDF_Guidelines_2022_23.pdf"
    index_name = "ordinance-rag"
    
    # Check for API keys
    if not os.getenv("PINECONE_API_KEY"):
        raise ValueError("PINECONE_API_KEY not found in .env")
    # GOOGLE_API_KEY is no longer strictly needed for embeddings, but might be for other parts if used later.

    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} chunks.")

    print("Initializing Embeddings...")
    # Using HuggingFace Embeddings (local)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Initializing Pinecone...")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Check if index exists
    existing_indexes = [i.name for i in pc.list_indexes()]
    if index_name in existing_indexes:
        index_info = pc.describe_index(index_name)
        if index_info.dimension != 384:
            print(f"Index {index_name} exists but has dimension {index_info.dimension} (expected 384). Deleting and recreating...")
            pc.delete_index(index_name)
            # Wait for deletion to propagate
            time.sleep(5)
            existing_indexes.remove(index_name)
    
    if index_name not in existing_indexes:
        print(f"Creating index {index_name}...")
        pc.create_index(
            name=index_name,
            dimension=384, # Dimension for all-MiniLM-L6-v2 is 384
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        # Wait for index to be ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    
    print("Indexing documents...")
    # Using from_documents to ingest
    # max_concurrency is handled by the underlying client or can be tuned if exposed.
    # Langchain's Pinecone wrapper handles batching.
    # We pass the index_name to connect to the existing (or just created) index.
    
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=index_name,
        pool_threads=5 # Setting concurrency to 5 as requested
    )
    
    print("Vector Database created and populated successfully!")
    
    # Verification query
    print("Running verification query...")
    results = vectorstore.similarity_search("What is this document about?", k=1)
    if results:
        print("Verification successful. Top result snippet:")
        print(results[0].page_content[:200])
    else:
        print("Verification warning: No results found.")

if __name__ == "__main__":
    try:
        create_vector_db()
    except Exception as e:
        import traceback
        traceback.print_exc()
