import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("API_KEY", ""), 
    base_url=os.getenv("BASE_URL", "")
)

