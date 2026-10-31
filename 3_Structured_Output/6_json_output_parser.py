from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="""
Return ONLY a valid JSON object.

Format:
{{
    "topic": "...",
    "definition": "...",
    "fact": "..."
}}

Topic: {topic}
""",
    input_variables=["topic"]
)

chain = template | model | parser

result = chain.invoke({"topic": "Black Hole"})

print(result)