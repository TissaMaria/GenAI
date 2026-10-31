# How It Works

# 1. You store your documents in a vector store (like FAISS, Chroma, Weaviate)
# 2. Each document is converted into a dense vector using an embedding model

# 3. When the user enters a query:

# . It's also turned into a vector
# . The retriever compares the query vector with the stored vectors

# . It retrieves the top-k most similar ones
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document

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
#Convert vector store into a retriever
retriever=vector_store.as_retriever(search_kwargs={"k":2})
# Why convert a vector store into a retriever?
# A vector store is essentially a database that holds embeddings (numerical representations of text, images, etc.). On its own, it’s just storage. To actually use it for search or question answering, you need a retriever.

# Retriever role: A retriever is an interface that knows how to query the vector store. It takes a user’s query, converts it into an embedding, and finds the most relevant stored items.

# Why as_retriever: This method wraps the vector store in a retriever object so you can plug it directly into higher-level pipelines (like LangChain’s RetrievalQA or conversational agents). It standardizes how queries are handled.

# What does search_kwargs={"k":2} mean?
# k=2 tells the retriever to return the top 2 most relevant results for each query.

# This is useful when you want concise context (e.g., only the two best matching documents) instead of flooding your model with too many.
query="What is Chroma used for?"

results=retriever.invoke(query)

for i,doc in enumerate(results):
    print("======Result=======")
    print(doc.page_content)



