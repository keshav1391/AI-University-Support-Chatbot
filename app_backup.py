from flask import Flask, render_template, request, jsonify
import joblib
import os
from dotenv import load_dotenv
from google import genai

app = Flask(__name__)


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    print("Gemini API key loaded successfully.")
else:
    print("WARNING: Gemini API key not found!")


# ==========================================
# LOAD ML MODEL
# ==========================================

model = joblib.load("university_chatbot_model.pkl")

print("ML Model loaded successfully.")


# ==========================================
# LOAD ABES KNOWLEDGE BASE
# ==========================================

try:

    with open("knowledge_base/abes_info.txt", "r", encoding="utf-8") as file:
        knowledge_base = file.read()

    print("ABES Knowledge Base loaded successfully.")

except Exception as e:

    print("Knowledge Base Error:")
    print(e)

    knowledge_base = ""


# ==========================================
# GEMINI CLIENT
# ==========================================

if api_key:

    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": 60000}
    )

else:

    client = None


# ==========================================
# SIMPLE FIXED RESPONSES
# ==========================================

responses = {

    "greeting":
        "Hello! 👋 How can I help you with university-related information?",

    "goodbye":
        "Goodbye! 👋 Have a great day!"
}


# ==========================================
# SIMPLE INTENT CHECK
# ==========================================

def detect_simple_intent(message):

    message = message.lower().strip()

    # Greeting
    if message in [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good evening"
    ]:
        return "greeting"

    # Goodbye
    if message in [
        "bye",
        "goodbye",
        "see you"
    ]:
        return "goodbye"

    return None


# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# CHAT API
# ==========================================

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_message = data.get("message", "").strip()


    # ==========================================
    # EMPTY MESSAGE
    # ==========================================

    if not user_message:

        return jsonify({
            "response": "Please enter a question."
        })


    print("\n==============================")
    print("User Question:", user_message)


    # ==========================================
    # FIRST: SIMPLE GREETING / GOODBYE
    # ==========================================

    simple_intent = detect_simple_intent(user_message)


    if simple_intent:

        response = responses[simple_intent]

        print("Simple Intent:", simple_intent)
        print("Response Source: Fixed Response")
        print("==============================")

        return jsonify({

            "response": response,
            "intent": simple_intent,
            "source": "fixed"

        })


    # ==========================================
    # SECOND: GEMINI + ABES KNOWLEDGE BASE
    # ==========================================

    if client:

        prompt = f"""
You are UniAssist AI, a university student support chatbot for ABES Engineering College.

Your task is to answer the student's question accurately using the ABES Knowledge Base provided below.

==============================
ABES KNOWLEDGE BASE
==============================

{knowledge_base}

==============================
END OF KNOWLEDGE BASE
==============================

STUDENT QUESTION:
{user_message}

==============================
IMPORTANT RULES
==============================

1. Give a helpful and concise answer.

2. Use simple, student-friendly language.

3. For ABES-related questions, use ONLY information available in the ABES Knowledge Base.

4. NEVER invent or guess ABES-specific information.

5. If the exact information requested by the student is NOT present in the Knowledge Base, clearly say:

"I don't have enough information about that."

6. For current or time-sensitive information such as:
   - fees
   - hostel fees
   - admission dates
   - examination schedules
   - placement statistics
   - notices
   - current college timings

   tell the student to verify the latest information from the official ABES website or relevant college portal.

7. If the Knowledge Base contains general information about a topic but does not contain the exact information requested, DO NOT create an approximate answer.

8. Do not claim access to private student information such as:
   - personal attendance
   - personal marks
   - personal fee status
   - personal timetable
   - student account information

9. If the question is unrelated to ABES, you may answer it as a normal general-knowledge question.

10. Do not mention Gemini, API, prompt, knowledge base, machine learning, or internal systems.

Answer the student's question now.
"""


        try:

            print("Sending question to Gemini...")

            interaction = client.interactions.create(

                model="gemini-3.8-flash",

                input=prompt

            )


            response = interaction.output_text


            print("Response Source: Gemini + ABES Knowledge Base")
            print("==============================")


            return jsonify({

                "response": response,

                "intent": "knowledge_base",

                "source": "gemini"

            })


        except Exception as e:

            print("Gemini API Error:")
            print(type(e).__name__)
            print(e)


            return jsonify({

                "response":
                    "I could not generate an answer right now. "
                    "Please verify the information from the official ABES website "
                    "or contact the appropriate university department.",

                "source": "fallback"

            })


    # ==========================================
    # FINAL FALLBACK
    # ==========================================

    return jsonify({

        "response":
            "I could not find specific information for this question. "
            "Please contact the university student support department.",

        "source": "fallback"

    })


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)