from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.set_page_config(page_title="Research Assistant")

st.header("Research Assistant")

paper = st.text_input("Research Paper / Topic")

style = st.selectbox(
    "Explanation Style",
    ["Beginner", "Technical", "Mathematical", "Bullet Points"]
)

length = st.selectbox(
    "Summary Length",
    ["Short", "Medium", "Detailed"]
)

if st.button("Summarize"):

    prompt = PromptTemplate(
        input_variables=["paper_input", "style_input", "length_input"],
        template="""
Please summarize the research paper titled "{paper_input}" with the following specifications:

Explanation Style: {style_input}
Explanation Length: {length_input}

1. Mathematical Details:
- Include relevant mathematical equations if present.
- Explain mathematical concepts using simple, intuitive code snippets where applicable.

2. Analogies:
- Use relatable analogies to simplify complex ideas.

If certain information is not available in the paper, respond with:
"Insufficient information available."

Ensure the summary is clear, accurate, and aligned with the provided style and length.
"""
    )

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation"
    )

    model = ChatHuggingFace(llm=llm)

    final_prompt = prompt.invoke(
        {
            "paper_input": paper,
            "style_input": style,
            "length_input": length,
        }
    )

    response = model.invoke(final_prompt)

    st.subheader("Summary")
    st.write(response.content)