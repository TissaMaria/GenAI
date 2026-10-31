# pip install -U langchain langchain-ollama langchain-chroma langchain-text-splitters

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# ============================================================
# 1. DOCUMENTS
# ============================================================

documents = [
    """
    The Avengers are a superhero team formed to protect Earth from
    powerful threats. The team has included heroes such as Iron Man,
    Captain America, Thor, Hulk, Black Widow and Hawkeye.
    """,

    """
    Tony Stark, also known as Iron Man, is a brilliant inventor and
    engineer. He created the Iron Man suit using advanced technology.
    His main strengths are intelligence, engineering and technological
    innovation.
    """,

    """
    Steve Rogers, also known as Captain America, became a super soldier
    after receiving the Super Soldier Serum. He uses a nearly indestructible
    shield and is known for his leadership and combat abilities.
    """,

    """
    Thor is an Asgardian warrior who possesses incredible strength,
    durability and control over lightning. His weapon Mjolnir can be
    wielded only by someone considered worthy.
    """,

    """
    Bruce Banner is a scientist who transforms into the Hulk when
    exposed to extreme anger or stress. The Hulk possesses enormous
    physical strength and durability.
    """,

    """
    Natasha Romanoff, also known as Black Widow, is a highly trained
    spy and combatant. She does not possess superhuman powers but relies
    on martial arts, intelligence, espionage and advanced equipment.
    """,

    """
    The Avengers have faced many threats, including Loki and the Chitauri
    invasion of Earth. The Battle of New York was one of the team's
    earliest major battles and brought the Avengers together against
    Loki's forces.
    """,

    """
    The Tesseract is a powerful cosmic object that contains the Space
    Stone. It has been associated with several major events involving
    Loki, the Avengers and the larger Marvel universe.
    """
]


# ============================================================
# 2. DOCUMENT CHUNKING
# ============================================================

# Recursive chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
#Try to make a chunk at the largest sensible boundary. If it is still too large, split it again using a smaller boundary.


chunks = text_splitter.create_documents(documents)

print("Number of chunks:", len(chunks))


# ============================================================
# 3. EMBEDDING MODEL
# ============================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ============================================================
# 4. VECTOR DATABASE
# ============================================================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="avengers-rag"
)


# ============================================================
# 5. RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
#When retrieving from the vector database, return the 3 most relevant chunks.


# ============================================================
# 6. LLM
# ============================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ============================================================
# 7. RAG
# ============================================================

def rag(query):

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(query)

    # Combine retrieved chunks
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Create prompt
    prompt = f"""
You are an Avengers knowledge assistant.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{query}

If the answer is not present in the context,
say "I don't know based on the provided context."
"""

    # Generate response
    response = llm.invoke(prompt)

    return response.content


# ============================================================
# 8. RUN
# ============================================================

while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    answer = rag(query)

    print("\nAnswer:")
    print(answer)