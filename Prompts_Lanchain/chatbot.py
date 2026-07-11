from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation")


model = ChatHuggingFace(llm = llm)

# to store chat history
store_chat = [
    SystemMessage(content="You are very HelpFul and powerful AI Assistant")
]


while True:
    user_input = input("How can I help you Pratik : ")
    store_chat.append(HumanMessage(content = user_input)) # to store the user input and convert user_input as humanmessage
    if user_input == "exit":
        break
    result = model.invoke(user_input)
    store_chat.append(AIMessage(content = result.content)) # to store the result  and convert result as AIMessage
    print("AI :", result.content)

print(store_chat)    