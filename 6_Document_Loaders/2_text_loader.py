from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
model = ChatOllama(model="llama3.2")
prompt=PromptTemplate(
    template="Write a summary about the text \n - {text}",
    input_variables=["text"]

)
parser=StrOutputParser()
chain = prompt|model|parser
loader = TextLoader("6_Document_Loaders\\doomsday.txt")
documents = loader.load()
#load() vs lazy_load() ->generator of Document Objects, documents are not all loaded at once, when you are working with large documents

#print(documents)
print(chain.invoke({'text':documents[0].page_content}))