import random
from abc import ABC, abstractmethod


# --------------------------------------------------
# 1. Runnable = common interface
# --------------------------------------------------

class Runnable(ABC):

    @abstractmethod
    def invoke(self, input_data):
        pass


# --------------------------------------------------
# 2. Dummy Prompt Template
# --------------------------------------------------

class DummyPromptTemplate(Runnable):

    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        return self.template.format(**input_dict)


# --------------------------------------------------
# 3. Dummy LLM
# --------------------------------------------------

class DummyLLM(Runnable):

    def __init__(self):
        print("LLM Created")

    def invoke(self, prompt):
        response_list = [
            "Delhi is the capital of India",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence"
        ]

        return random.choice(response_list)

    def predict(self, prompt):
        response_list = [
            "Delhi is the capital of India",
            "IPL is a cricket league",
            "AI stands for Artificial Intelligence"
        ]

        return {
            "response": random.choice(response_list)
        }


# --------------------------------------------------
# 4. Runnable Connector
# --------------------------------------------------

class RunnableConnector(Runnable):

    def __init__(self, runnable_list):
        self.runnable_list = runnable_list

    def invoke(self, input_data):

        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)

        return input_data


# --------------------------------------------------
# 5. Create Prompt
# --------------------------------------------------

prompt = DummyPromptTemplate(
    template="Answer the following question: {question}",
    input_variables=["question"]
)


# --------------------------------------------------
# 6. Create LLM
# --------------------------------------------------

llm = DummyLLM()


# --------------------------------------------------
# 7. Connect Prompt + LLM
# --------------------------------------------------

chain = RunnableConnector([
    prompt,
    llm
])


# --------------------------------------------------
# 8. Invoke the chain
# --------------------------------------------------

result = chain.invoke({
    "question": "What is AI?"
})

print(result)