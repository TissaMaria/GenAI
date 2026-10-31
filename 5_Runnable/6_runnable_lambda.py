#returns input as output no modifications
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

model = ChatOllama(model="llama3.2")
prompt=PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]

)

parser=StrOutputParser()


joke_gen_chain=RunnableSequence(prompt,model,parser)

parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(lambda x:len(x.split()))
})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)

print(final_chain.invoke({'topic':'AI'}))

