from langchain_core.runnables import RunnableLambda

def word_counter(text):
    return len(text.split())


runnable_words_count = RunnableLambda(word_counter)

print(runnable_words_count.invoke('Hi there how are you doing?'))
