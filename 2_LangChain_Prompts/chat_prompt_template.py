from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
#Cannot be used as the it does not fill the templates properly

load_dotenv()

chat_template = ChatPromptTemplate([
('system', 'You are a helpful {domain} expert'),
('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)

#Message Placeholder: used inside ChatPromptTempate to dynamically insert chat history to a lost of messages at runtime

