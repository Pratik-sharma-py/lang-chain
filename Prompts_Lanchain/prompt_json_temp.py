from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

# header
st.header('Research Tool')

paper_input = st.selectbox("Select Research Paper Name", ["Attention Is All YOU Need",
"BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners",
"Diffusion Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Begineer-Friendly", 
"Technical","Code-Oriented","Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)","Medium (3-5 paragraphs)", "Long (detailed explanation)"])


template = load_prompt('template.json')


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation")

model = ChatHuggingFace(llm = llm)

if st.button('Summarize'):
    chain = template | model # create a chain to use only one time invoke
    result = chain.invoke({
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    })
    st.write(result.content)