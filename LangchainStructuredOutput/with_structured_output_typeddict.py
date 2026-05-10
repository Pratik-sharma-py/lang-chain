from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

#schema 

class Review(TypedDict):
    summary : str
    sentiment : str
structured_model = model.with_structured_output(Review)

result = structured_model.invoke(""" rain started quietly in the evening.
People rushed through the streets with colorful umbrellas.
By night, the city felt calm and peaceful again.""")

print(result)