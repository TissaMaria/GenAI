# ambiguous query ->llm ->multiple queries that cover the input -> retriever _>each retriever search docs based on query
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

# 1. Create Document objects
docs = [
    Document(
        page_content="Python is a programming language.",
        metadata={"topic": "Python"}
    ),
    Document(
        page_content="ChromaDB is a vector database used to store embeddings.",
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

# 5. Create a normal retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# 6. Create LLM
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

# 7. Convert normal retriever into MultiQuery Retriever
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

# 8. User query
query = "What is Chroma used for?"

# 9. Retrieve documents
results = multi_query_retriever.invoke(query)

# 10. Print results
for i, doc in enumerate(results):
    print("====== Result ======")
    print(doc.page_content)