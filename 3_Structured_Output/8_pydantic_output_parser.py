from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Pydantic model
class Student(BaseModel):
    name: str = Field(description="Student's name")
    age: int = Field(description="Student's age")

# Parser
parser = PydanticOutputParser(pydantic_object=Student)

# Prompt
template = PromptTemplate(
    template="""
Extract the information.

{format_instructions}

Text:
{text}
""",
    input_variables=["text"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

chain = template | model | parser

result = chain.invoke({
    "text": "Rahul is 22 years old."
})

print(result)