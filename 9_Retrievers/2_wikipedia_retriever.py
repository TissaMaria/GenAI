from langchain_community.retrievers import WikipediaRetriever
#Initialize Retriver
retriever=WikipediaRetriever(top_k_results=2,lang='en')
query="What came first chicken or egg?"
docs=retriever.invoke(query)
for i,doc in enumerate(docs):
    print(f"--------------------{i+1}------------------")
    print(f"Content:\n{doc.page_content}...")
    
