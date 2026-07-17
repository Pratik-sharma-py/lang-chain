from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    temperature=0.1
)

model1 = ChatHuggingFace(llm=llm)

model2 = ChatCohere(model="command-r-plus-08-2024")

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the followin text \n {text}",
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n {notes} and {quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        'quiz': prompt2 | model2 | parser
    }
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Hallucination in Large Language Models (LLMs) refers to the situation where a model generates information that sounds convincing and 
fluent but is factually incorrect, fabricated, or unsupported by the given context. 
This happens because LLMs are designed to predict the most probable next word based on patterns learned during training rather than verify facts like a search engine or database. 
Hallucinations can occur due to ambiguous prompts, insufficient context, outdated training data, smaller or less capable models, or high creativity settings (temperature). 
While hallucinations can sometimes be useful in creative writing, they are a major challenge in applications such as healthcare, law, finance, and education, where factual accuracy is critical. 
Techniques such as prompt engineering, Retrieval-Augmented Generation (RAG), grounding responses in trusted documents, and using more capable models help reduce hallucinations and improve the reliability of LLM-generated responses.
"""

result = chain.invoke({'text':text})
print(result)