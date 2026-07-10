from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"), 
   # provider="hf-inference",                                          
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)
result = model.invoke("What is life?")
print(result.content)