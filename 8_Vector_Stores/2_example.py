from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

# 1. Create Document objects
docs = [
    Document(
        page_content="Python is a programming language.",
        metadata={"topic": "Python"}
    ),
    Document(
        page_content="ChromaDB is a vector database.",
        metadata={"topic": "Vector Database"}
    ),
    Document(
        page_content="LangChain helps build LLM applications.",
        metadata={"topic": "LangChain"}
    )
]

# 2. Create embeddings using Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. Create Chroma vector store
vector_store = Chroma(
    collection_name="my_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# 4. Add documents
vector_store.add_documents(docs)

# 5. Search similar documents
results = vector_store.similarity_search(
    "What is a vector database?",
    k=2
)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)

# 6. Create Llama model
llm = ChatOllama(model="llama3.2")

# 7. Give retrieved documents to Llama
context = "\n".join(doc.page_content for doc in results)

response = llm.invoke(
    f"""Answer the question using the context below.

Context:
{context}

Question:
What is a vector database?
"""
)

print(response.content)