from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Step 1: Analyze sentiment
sentiment_prompt = PromptTemplate(
    template="Classify the following feedback as Positive or Negative.\n\nFeedback: {feedback}",
    input_variables=["feedback"]
)

sentiment_chain = sentiment_prompt | model | parser

# Step 2: Positive response
positive_prompt = PromptTemplate(
    template="Write a thank-you message for this customer.\n\n{feedback}",
    input_variables=["feedback"]
)

positive_chain = positive_prompt | model | parser

# Step 3: Negative response
negative_prompt = PromptTemplate(
    template="Write an apology message for this customer.\n\n{feedback}",
    input_variables=["feedback"]
)

negative_chain = negative_prompt | model | parser

# -------------------------

feedback = "The product quality is amazing. I loved it!"

# First chain
sentiment = sentiment_chain.invoke({"feedback": feedback})

print("Sentiment:", sentiment)

# Conditional routing
if "positive" in sentiment.lower():
    result = positive_chain.invoke({"feedback": feedback})
else:
    result = negative_chain.invoke({"feedback": feedback})

print("\nResponse:\n")
print(result)