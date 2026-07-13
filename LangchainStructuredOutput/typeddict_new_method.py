from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 200,
        "temperature": 0.1,
        "do_sample": True,
        "return_full_text": False,
    },
)
model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

prompt = PromptTemplate(
    template=(
        "You are a text analyzer. Read the text and produce a JSON object with "
        "exactly two keys: \"summary\" (a one-sentence summary) and "
        "\"sentiment\" (one word: positive, negative, or neutral).\n\n"
        "Example output:\n"
        '{{"summary": "A man walks his dog in the park.", "sentiment": "positive"}}\n\n'
        "Now analyze this text and output ONLY the JSON, nothing else:\n"
        "{text}\n\n"
        "JSON:"
    ),
    input_variables=["text"],
)

chain = prompt | model | parser

result = chain.invoke({"text": """rain started quietly in the evening.
People rushed through the streets with colorful umbrellas.
By night, the city felt calm and peaceful again."""})

print(result)
print("Summary:", result["summary"])
print("Sentiment:", result["sentiment"])