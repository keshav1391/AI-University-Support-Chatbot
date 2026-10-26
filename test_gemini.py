import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("API key nahi mili!")
    exit()

print("API key mil gayi.")
print("Gemini se connection try kar rahe hain...")

client = genai.Client(
    api_key=api_key,
    http_options={"timeout": 30000}
)

try:
    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input="What is ABES Engineering College? Answer in one short sentence."
    )

    print("\nGemini Response:")
    print(interaction.output_text)

except Exception as e:
    print("\nGemini API Error:")
    print(type(e).__name__)
    print(e)