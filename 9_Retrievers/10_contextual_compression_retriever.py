# The Contextual Compression Retriever in LangChain is an advanced retriever that improves
# retrieval quality by compressing documents after retrieval - keeping only the relevant
# content based on the user's query.

# ? Query:
# "What is photosynthesis?"

# Retrieved Document (by a traditional retriever):

# "The Grand Canyon is a famous natural site.
# Photosynthesis is how plants convert light into energy.
# Many tourists visit every year."

# X Problem:
# . The retriever returns the entire paragraph

# . Only one sentence is actually relevant to the query
# . The rest is irrelevant noise that wastes context window and may confuse the LLM

# V What Contextual Compression Retriever does:

# Returns only the relevant part, e.g.
# "Photosynthesis is how plants convert light into energy."

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from langchain_classic.retrievers.contextual_compression import (
    ContextualCompressionRetriever
)
from langchain_classic.retrievers.document_compressors import (
    EmbeddingsFilter
)


# 1. Documents
docs = [
    Document(
        page_content="""
        Netflix uses recommendation algorithms to personalize content
        for each user. It analyzes viewing history, ratings, searches,
        and the behavior of similar users to predict which shows a person
        may want to watch.

        Netflix was originally a DVD rental company founded in 1997.
        Netflix produces many original movies and television shows.
        Stranger Things is one of its popular original series.
        Netflix operates in many countries.
        """,
        metadata={"topic": "Netflix"}
    ),

    Document(
        page_content="""
        Spotify uses machine learning to recommend music to users.
        It considers listening history, playlists, skipped songs,
        liked songs, and artists that a user follows.

        Spotify also provides podcasts and audiobooks.
        """,
        metadata={"topic": "Spotify"}
    ),

    Document(
        page_content="""
        Amazon uses recommendation systems on its shopping platform.
        These systems analyze products that customers view, purchase,
        or add to their cart.

        Amazon also operates AWS and Prime Video.
        """,
        metadata={"topic": "Amazon"}
    )
]


# 2. Embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# 3. Chroma
vector_store = Chroma(
    collection_name="compression_demo_2",
    embedding_function=embeddings,
    persist_directory="./compression_db_2"
)

vector_store.add_documents(docs)


# 4. Normal retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# 5. Create embedding-based compressor
compressor = EmbeddingsFilter(
    embeddings=embeddings,
    similarity_threshold=0.5
)


# 6. Contextual Compression Retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)


# 7. Query
query = "How does Netflix personalize recommendations?"


# 8. Normal retrieval
print("\n========== NORMAL RETRIEVER ==========\n")

results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"--- Document {i + 1} ---")
    print(doc.page_content)


# 9. Compressed retrieval
print("\n========== COMPRESSED RETRIEVER ==========\n")

results = compression_retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"--- Document {i + 1} ---")
    print(doc.page_content)