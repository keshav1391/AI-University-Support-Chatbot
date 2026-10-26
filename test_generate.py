import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"timeout": 60000}
)

start = time.time()

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="What is ABES Engineering College? Answer in one short sentence."
    )
    print("Response:", response.text)

except Exception as e:
    print("Error:", type(e).__name__)
    print(e)

print("Time taken:", round(time.time() - start, 1), "seconds")