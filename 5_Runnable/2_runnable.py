#This method is not flexible. This chain is not flexible for any workflows
import random
class DummyLLM:
    def __init__(self):
        print("LLM Created")
    def predict(self,prompt):
        response_list=[
            "Delhi is the capital of India",
            "AI stands for Artificial Intelligence"
        ]
        return {"response":random.choice(response_list)}
llm=DummyLLM()

class DummyPromptTemplate:
    def _init_(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format( ** input_dict)

class DummyLLMChain:
    def _init_(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt

    def run(self, input_dict):

        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)

        return result['response']