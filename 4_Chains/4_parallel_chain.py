from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel
load_dotenv()

# LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
template='Generate short and simple notes from the following text \n {text}',
input_variables=['text']
)
prompt2 = PromptTemplate(
template='Generate 5 short question answers from the following text \n {text}',
input_variables=['text']
)
prompt3 = PromptTemplate(
template='Merge the provided notes and quiz into a single document \n notes -> {notes} and {quiz}',
input_variables=['notes', 'quiz']
)

parallel_chain=RunnableParallel({
    'notes':prompt1|model|parser,
    'quiz':prompt2|model|parser
})

merge_chain=prompt3|model|parser

chain=parallel_chain|merge_chain

text="""The Odyssey is an ancient epic poem**: The Odyssey is an ancient Greek epic poem attributed to the poet Homer, and it is estimated to have been composed between 800 and 700 BCE. It is one of the oldest surviving works of literature in the Western canon.

2. **The Odyssey was not a single work**: Until the 20th century, the Odyssey was thought to be a single, unified work. However, scholars now believe that the poem was composed in stages, with different authors contributing to it over time. The oldest parts of the poem date back to around 750 BCE, while the final version was likely compiled in the 7th century BCE.

3. **The Odyssey was a journey in more ways than one**: The Odyssey is not just a story about Odysseus' journey home from the Trojan War, but also a metaphor for the journey of life. The poem explores themes of identity, morality, and the human condition, making it a timeless classic that continues to resonate with readers today.

4. **The Sirens were not just mythical creatures**: The Sirens, creatures that lure sailors to their deaths with their enchanting voices, were likely inspired by real-life sea creatures. Some scholars believe that the Sirens were based on the songs of the Myna bird, a species of bird that was known to inhabit the Mediterranean region and was said to have a hauntingly beautiful song.
"""

result=chain.invoke({'text':text})

print(result)

