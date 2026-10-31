from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser
from langchain_core.output_parsers import ResponseSchema
from dotenv import load_dotenv

load_dotenv()
#In LangChain 1.x, StructuredOutputParser and ResponseSchema are no longer part of langchain-core.
# LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Define the fields you want
language = ResponseSchema(
    name="language",
    description="Programming language name"
)

creator = ResponseSchema(
    name="creator",
    description="Creator of the language"
)

# Create parser
parser = StructuredOutputParser.from_response_schemas(
    [language, creator]
)

# Prompt
template = PromptTemplate(
    template="""
Answer the question.

{format_instructions}

Question: {question}
""",
    input_variables=["question"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# Chain
chain = template | model | parser

result = chain.invoke({
    "question": "Tell me about Python."
})

print(result)