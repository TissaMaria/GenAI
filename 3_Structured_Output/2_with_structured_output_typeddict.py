from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal

load_dotenv()
"""Note: Llama 3.1 Instruct does not natively guarantee structured outputs like OpenAI's GPT models.
        This is just for referrence"""
llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation"
    )
model = ChatHuggingFace(llm=llm)

#schema
class Review(TypedDict):
    key_themes:Annotated[list[str],"Write down all the themes in the review"]
    summary:Annotated[str,"A brief summary"]
    sentiment:Annotated[str,"Return positive negative or neutral"]
    pros:Annotated[Optional[list[str]],"Any pros of the product in the review"]
#Not the best for data validation as it may still return things that isnt valid wrt to the label given
#It ensures that the output is structured
structured_model=model.with_structured_output(Review)
result=structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many
pre-installed apps that I can't remove. Also, the UI looks outdated compared to other
brands. Hoping for a software update to fix this.""")

print(result)