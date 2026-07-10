from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

models_to_test = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
    "Qwen/Qwen2.5-72B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "tiiuae/falcon-7b-instruct",
]

for m in models_to_test:
    try:
        client.chat_completion(
            model=m,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10
        )
        print(f"WORKS: {m}")
    except Exception as e:
        print(f"FAILED: {m} → {str(e)[:60]}")