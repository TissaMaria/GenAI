from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# ---------------- First Chain ----------------

report_prompt = PromptTemplate(
    template="Write a detailed report about {topic}.",
    input_variables=["topic"]
)

report_chain = report_prompt | model | parser

# ---------------- Second Chain ----------------

summary_prompt = PromptTemplate(
    template="Summarize the following report in 5 lines.\n\n{report}",
    input_variables=["report"]
)

summary_chain = summary_prompt | model | parser

# ---------------- Sequential Chain ----------------

chain = report_chain | summary_chain

result = chain.invoke({"topic": "Artificial Intelligence"})

print(result)

chain.get_graph().print_ascii()