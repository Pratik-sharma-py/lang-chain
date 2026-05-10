from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Optional

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task = "text-generation"
)

model = ChatHuggingFace(llm=llm)

#schema

class Review(TypedDict):
    key_themes = Annotated[list[str],'Write down all the key themes discussed in the review in a list']
    summary : Annotated[str,'A brief summary of the review']
    sentiment : Annotated[str,'Return sentiment of the review either negative, positive or neutral or anything you feel']
    pros : Annotated[Optional[list[str]],'Write down all the pros inside a list']
    cons : Annotated[Optional[list[str]],'Write down all the cons inside a list']

structured_model = model.with_structured_output(Review)    

result = structured_model.invoke("""I recently purchased a smartwatch to help track my fitness and daily activities.
The design looks modern and feels very comfortable on the wrist.
Its display is bright and easy to read even under sunlight.
The fitness tracking features, especially step counting and heart rate monitoring, work quite accurately.
Battery life is decent and usually lasts for about two days with regular use.
I also liked the sleep tracking and notification features during work hours.
However, the charging speed is slower than I expected.
Sometimes the mobile app takes a few seconds to sync the data properly.
The price feels slightly expensive compared to similar products in the market.
Overall, it is a reliable and useful gadget for everyday fitness and productivity needs.
""")

print(result['summary'])