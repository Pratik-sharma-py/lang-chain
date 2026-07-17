from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

#llm = HuggingFaceEndpoint(
   # repo_id="meta-llama/Llama-3.1-8B-Instruct",
   # task="text-generation",
   # temperature=0.1
#)

#model1 = ChatHuggingFace(llm=llm)

model2 = ChatCohere(model="command-r-plus-08-2024")

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field(description="Give the sentiment of the Feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into Positive or Negative \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifer_chain = prompt1 | model2 | parser2

#result = classifer_chain.invoke({'feedback':"This is okay smartphone"}).sentiment 

#print(result)

# 2nd Part --> Runnable Branch

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive", prompt2 | model2 | parser),
    (lambda x:x.sentiment == "negative", prompt3 | model2 | parser),
    RunnableLambda(lambda x: "Couldn't find Sentiment")
)

chain = classifer_chain | branch_chain

result = chain.invoke({'feedback':'This is excellent phone'})
print(result)