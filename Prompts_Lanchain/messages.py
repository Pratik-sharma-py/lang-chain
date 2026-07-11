from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation")


model = ChatHuggingFace(llm = llm)

message = [
    SystemMessage(content = 'You are a top engineer'),
    HumanMessage(content = 'Tell me about AI/ML')
]

result = model.invoke(message)
message.append(AIMessage(content = result.content))
print(message)