from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -------------------------
# 1. Documents
# -------------------------

documents = [
    "Iron Man is Tony Stark, a genius inventor and billionaire who built the Iron Man suit.",
    "Captain America is Steve Rogers. He received the Super Soldier Serum and carries a shield.",
    "Thor is the God of Thunder and uses the magical hammer Mjolnir.",
    "The Hulk is Bruce Banner, a scientist who transforms into a powerful green creature.",
    "Black Widow is Natasha Romanoff, a highly skilled spy and member of the Avengers.",
    "The Battle of New York was fought by the Avengers against Loki and his invading army."
]


# -------------------------
# 2. Split documents
# -------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.create_documents(documents)


# -------------------------
# 3. Embeddings + Vector DB
# -------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    collection_name="avengers"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# -------------------------
# 4. Prompt Template
# -------------------------

prompt = PromptTemplate.from_template("""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
""")


# -------------------------
# 5. LLM
# -------------------------

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# -------------------------
# 6. Format retrieved docs
# -------------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -------------------------
# 7. RAG Chain
# -------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


# -------------------------
# 8. Ask question
# -------------------------

question = "Who is Iron Man?"

answer = rag_chain.invoke(question)

print(answer)