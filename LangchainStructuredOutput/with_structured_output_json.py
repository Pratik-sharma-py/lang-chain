from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict, Annotated,Optional,Literal
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import os 

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",
    task = "text-generation",
    provider="auto",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm = llm)

#schema 

json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

tructured_model = model.with_structured_output(json_schema)

result = tructured_model.invoke("""I recently purchased a smartwatch to help track my fitness and daily activities.
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

print(result)