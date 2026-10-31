import langchain

print(langchain.__version__)

#model = ChatOpenAI(model='gpt-4', temperature=1.5, max_completion_tokens=10)
#keep temp 0 for deterministic chats (Same input same output) for creative tasks keep it greater than zero (Same input different output)