import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"timeout": 20000}
)

models_to_test = [
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]

for name in models_to_test:
    start = time.time()
    try:
        response = client.models.generate_content(
            model=name,
            contents="What is ABES Engineering College? Answer in one short sentence."
        )
        print(f"OK    | {name} | {round(time.time() - start, 1)}s | {response.text[:80]}")
    except Exception as e:
        print(f"FAIL  | {name} | {round(time.time() - start, 1)}s | {type(e).__name__}")