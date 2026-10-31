from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

"""Note: Llama 3.1 Instruct does not natively guarantee structured outputs like OpenAI's GPT models.
        This is just for reference.
"""

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


# Schema
class Review(BaseModel):
    key_themes: list[str] = Field(
        description="Write down all the themes in the review"
    )
    summary: str = Field(
        description="A brief summary"
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Return positive, negative, or neutral"
    )
    pros: Optional[list[str]] = Field(
        default=None,
        description="Any pros of the product in the review"
    )


# Better data validation than TypedDict
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
The hardware is great, but the software feels bloated. There are too many
pre-installed apps that I can't remove. Also, the UI looks outdated compared to other
brands. Hoping for a software update to fix this.
""")

print(result)