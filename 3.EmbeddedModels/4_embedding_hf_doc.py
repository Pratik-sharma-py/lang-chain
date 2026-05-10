from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

document = [
    'My name is Pratik Sharma',
    "I am from Nepal",
    ' i AM 18 YEARS old'
]

vector = embedding.embed_documents(document)

print(str(vector))