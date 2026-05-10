from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st

#model = ChatHuggingFace()

st.header('Research Tool')
user_input = st.text_input('Enter Your prompt')

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
       temperature= 0.5,
       max_new_tokens = 100
)
)
model = ChatHuggingFace(llm=llm)

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content)
