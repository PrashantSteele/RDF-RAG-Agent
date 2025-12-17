import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "libs"))

from dotenv import load_dotenv

load_dotenv()

def get_rag_chain():
    """
    Initializes and returns the RAG chain with history support.
    """
    # Import dependencies locally to avoid PyO3/NumPy re-initialization issues on Streamlit reload
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_groq import ChatGroq

    print("Initializing Embeddings (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Connecting to Pinecone Vector Store...")
    vectorstore = PineconeVectorStore(
        index_name="ordinance-rag",
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    print("Initializing LLM (Groq API)...")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    print("LLM initialized successfully!")

    # Contextualize question prompt
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Answer prompt
    qa_system_prompt = """You are a helpful assistant for answering questions about the Ordinance 11. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.

    Context:
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    # Test the chain locally
    chain = get_rag_chain()
    chat_history = []
    
    query1 = "What is PSDC?"
    print(f"Query: {query1}")
    response1 = chain.invoke({"input": query1, "chat_history": chat_history})
    print(f"Answer: {response1['answer']}")
    
    chat_history.extend([
        ("human", query1),
        ("ai", response1['answer'])
    ])
    
    query2 = "How is it constituted?"
    print(f"\nQuery: {query2}")
    response2 = chain.invoke({"input": query2, "chat_history": chat_history})
    print(f"Answer: {response2['answer']}")
