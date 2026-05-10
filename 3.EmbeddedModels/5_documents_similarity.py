from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

document = [
    """Aarav, 19, lives in a small town and spends most evenings repairing old electronics he finds at local markets. "
       He dreams of becoming a robotics engineer but currently teaches younger students basic coding to save money for college. "
       Quiet but observant, he enjoys documenting strange mechanical failures in a notebook. Friends often rely on him when devices stop working because he rarely gives up on a problem.
       Although he appears serious, he secretly enjoys sketching comic characters during class breaks. His biggest goal is to build affordable educational robots for rural schools where access to modern technology is still extremely limited.""",
    """Mina, 27, works as a wildlife photographer and travels across remote forests capturing endangered animals through her camera lens. 
       She prefers silence over crowded cities and can spend hours waiting patiently for the perfect photograph. Her work has appeared in environmental magazines, helping raise awareness about habitat destruction. 
       Despite her adventurous career, she is deeply afraid of deep water and avoids boats whenever possible. Mina keeps a journal filled with hand-drawn maps and stories from local villagers she meets during expeditions. 
       She believes storytelling through photography can influence people more powerfully than arguments, statistics, or political debates alone in society.""",
    """Daniel, 34, owns a small bakery famous for unusual bread recipes inspired by different cultures around the world. Every morning, he wakes before sunrise to prepare dough while listening to classical music in the kitchen. 
       Customers admire not only his baking skills but also the handwritten motivational notes he includes with every order. Before becoming a baker, Daniel studied architecture, which explains his artistic approach to pastry design and presentation. 
       He experiments constantly, sometimes failing dozens of times before perfecting a recipe. Outside work, he volunteers at community centers teaching children how cooking can improve creativity, patience, teamwork, and confidence in everyday life.""",
    """Lhamo, 22, is a university student studying environmental science while balancing part-time work at a mountain trekking lodge. Growing up near the Himalayas inspired her passion for climate research and sustainable tourism. 
       She often organizes local cleanup campaigns and encourages travelers to reduce plastic waste during trekking seasons. Although naturally shy, she becomes confident when speaking about environmental issues affecting mountain communities. 
       Lhamo enjoys reading historical travel journals and comparing them with modern environmental conditions. Her long-term ambition is to create eco-friendly tourism programs that protect both local culture and fragile ecosystems while still supporting economic opportunities for families living in remote villages.""",
    """Ethan, 41, was once a professional athlete but later changed careers to become a high school history teacher. After an injury ended his sports career, he struggled with uncertainty before discovering his passion for education. 
       His classes are energetic because he teaches history like storytelling instead of memorization. Students appreciate how he connects ancient events with modern social issues and technology. Outside school, Ethan restores vintage motorcycles in his garage and participates in charity rides across different states. 
       He believes failure can redirect people toward more meaningful paths, and he frequently shares personal experiences to motivate students facing challenges in academics, sports, or personal life."""
]

query = "who is 22 years old Lhamo."

doc_embeddings = embedding.embed_documents(document)
query_embeddings = embedding.embed_query(query)

score = cosine_similarity([query_embeddings],doc_embeddings)[0]

index , score = sorted(list(enumerate(score)),key = lambda x:x[1])[-1]


print(query)
print(document[index])
print("similarity score :", score)