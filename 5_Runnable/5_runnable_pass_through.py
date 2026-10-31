#returns input as output no modifications
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

model = ChatOllama(model="llama3.2")
prompt1=PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]

)
prompt2=PromptTemplate(template="Explain the following joke - {text}",
                       input_variables=['text'])
parser=StrOutputParser()


joke_gen_chain=RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough,
    'explanation':RunnableSequence(prompt2,model,parser)

})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)

print(final_chain.invoke({'topic':'AI'}))

