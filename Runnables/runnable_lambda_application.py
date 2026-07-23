from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda , RunnablePassthrough, RunnableSequence, RunnableParallel
from langchain_core.output_parsers  import StrOutputParser


load_dotenv()

def word_counter(text):
    return len(text.split())


prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)



model = ChatCohere(model="command-r-plus-08-2024")

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'Football'})

final_result = """{} \n word-count - {}""".format(result['joke'],result['word_count'])

print(final_result)