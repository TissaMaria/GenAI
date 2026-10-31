from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

"""Note: Llama 3.1 Instruct does not natively guarantee structured outputs like OpenAI's GPT models.
        This is just for reference.
"""

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


# JSON Schema
review_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Write down all the key themes discussed in the review"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "Overall sentiment of the review"
        },
        "pros": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Any positive aspects mentioned in the review"
        }
    },
    "required": ["key_themes", "summary", "sentiment"]
}

structured_model = model.with_structured_output(review_schema)

result = structured_model.invoke("""
The hardware is great, but the software feels bloated. There are too many
pre-installed apps that I can't remove. Also, the UI looks outdated compared to other
brands. Hoping for a software update to fix this.
""")

print(result)

"""Output
{'key_themes': ['hardware', 'software', 'UI'], 'summary': 'Overall, the device has some great hardware but is let down by its software and outdated UI.', 'sentiment': 'negative', 'pros': ['hardware'], 'cons': ['software', 'UI']}
"""