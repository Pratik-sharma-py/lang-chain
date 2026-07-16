import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation',
    temperature=0.1 
)

model = ChatHuggingFace(llm=llm)

# Schema definitions
schema = [
    ResponseSchema(name='fact1', description="Fact 1 about the topic"),
    ResponseSchema(name='fact2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 Facts about {topic}\n{format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.invoke({'topic': 'black hole'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print("\n--- Successfully Parsed JSON Output ---")
print(final_result)
