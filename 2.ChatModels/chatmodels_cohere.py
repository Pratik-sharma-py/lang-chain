import os
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, AIMessage

# 1. Load environment variables from the .env file
load_dotenv()

# Verify the key loaded successfully before proceeding
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("Error: COHERE_API_KEY not found. Check your .env file format.")

# 2. Initialize the Cohere Chat Model
chat_model = ChatCohere(model="command-r-plus-08-2024")

# 3. Initialize chat history list
chat_history = []

print("Chatbot initialized! Type 'exit' or 'quit' to end the conversation.\n")

# 4. Start the interactive chat loop
while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
        
    if not user_input.strip():
        continue

    chat_history.append(HumanMessage(content=user_input))

    try:
        response = chat_model.invoke(chat_history)
        print(f"Bot: {response.content}\n")
        chat_history.append(AIMessage(content=response.content))
        
    except Exception as e:
        print(f"An error occurred: {e}\n")
