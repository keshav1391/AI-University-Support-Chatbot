from flask import Flask, render_template, request, jsonify, session
import joblib
import os
import re
import csv
import uuid
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

load_dotenv()

app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")


# ==========================================
# SETTINGS
# ==========================================

GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
GEMINI_TIMEOUT_MS = 15000
MAX_MESSAGE_LENGTH = 500
MEMORY_TURNS = 4
TOP_K_SECTIONS = 3
MIN_RETRIEVAL_SCORE = 0.05
LOG_FILE = "chat_log.csv"


# ==========================================
# LOAD API KEY
# ==========================================

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
# LOAD KNOWLEDGE BASE AND SPLIT INTO SECTIONS
# ==========================================

def load_sections(path):

    """Split the knowledge base into sections using the ALL-CAPS headings."""

    sections = []

    try:
        with open(path, "r", encoding="utf-8") as file:
            lines = file.read().splitlines()
    except Exception as e:
        print("Knowledge Base Error:", e)
        return sections

    title = None
    body = []

    def save():
        text = "\n".join(body).strip()
        if title and text and not title.startswith("IMPORTANT RULE"):
            sections.append({"title": title, "text": text})

    for line in lines:
        stripped = line.strip()

        if stripped and stripped.isupper() and not stripped.startswith("-"):
            save()
            title = stripped
            body = []
        else:
            body.append(line)

    save()

    return sections


sections = load_sections("knowledge_base/abes_info.txt")

if sections:
    section_texts = [(s["title"] + " ") * 2 + s["text"] for s in sections]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    section_matrix = vectorizer.fit_transform(section_texts)
    print(f"Knowledge Base loaded: {len(sections)} sections.")
else:
    vectorizer = None
    section_matrix = None
    print("WARNING: Knowledge Base is empty.")


def retrieve_sections(question):

    """Return the most relevant sections for the question."""

    if not sections:
        return [], 0.0

    question_vector = vectorizer.transform([question])
    scores = linear_kernel(question_vector, section_matrix).flatten()

    ranked = scores.argsort()[::-1]
    best_score = float(scores[ranked[0]])

    # Nothing matched well: send all sections (the file is small)
    if best_score < MIN_RETRIEVAL_SCORE:
        return sections, best_score

    top = [sections[i] for i in ranked[:TOP_K_SECTIONS] if scores[i] > 0]

    return top, best_score


# ==========================================
# GEMINI CLIENT
# ==========================================

if api_key:
    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": GEMINI_TIMEOUT_MS}
    )
else:
    client = None


# ==========================================
# FIXED RESPONSES (ONLY SMALL TALK)
# ==========================================

responses = {

    "greeting":
        "Hello! 👋 How can I help you with ABES and university-related information?",

    "thanks":
        "You're welcome! 😊 Ask me anything else about ABES.",

    "goodbye":
        "Goodbye! 👋 Have a great day!"
}


def detect_known_intent(message):

    message = message.lower().strip()

    if len(message.split()) <= 4:

        if re.search(r"\b(hello|hi|hey|good morning|good evening)\b", message):
            return "greeting"

        if re.search(r"\b(thanks|thank you|thankyou)\b", message):
            return "thanks"

        if re.search(r"\b(bye|goodbye)\b", message):
            return "goodbye"

    return None


# ==========================================
# CHAT MEMORY (per browser session)
# ==========================================

chat_memory = {}


def get_session_id():

    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())

    return session["sid"]


def get_history(sid):

    return chat_memory.get(sid, [])


def add_to_history(sid, question, answer):

    history = chat_memory.setdefault(sid, [])
    history.append({"q": question, "a": answer})
    chat_memory[sid] = history[-MEMORY_TURNS:]


# ==========================================
# LOGGING (useful as testing evidence)
# ==========================================

def log_chat(question, source, model_name, sections_used):

    try:
        new_file = not os.path.exists(LOG_FILE)

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if new_file:
                writer.writerow(["time", "question", "source", "model", "sections"])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                question,
                source,
                model_name,
                " | ".join(sections_used)
            ])

    except Exception as e:
        print("Log error:", e)


# ==========================================
# PROMPT
# ==========================================

def build_prompt(question, context_sections, history):

    context = "\n\n".join(
        f"[{s['title']}]\n{s['text']}" for s in context_sections
    )

    if history:
        conversation = "\n".join(
            f"Student: {h['q']}\nUniAssist: {h['a']}" for h in history
        )
    else:
        conversation = "(no earlier messages)"

    return f"""
You are UniAssist AI, a student support assistant for ABES Engineering College, Ghaziabad.

ABES INFORMATION:
{context}

RECENT CONVERSATION:
{conversation}

STUDENT QUESTION:
{question}

RULES:
1. Answer in simple, friendly, concise language.
2. For ABES-specific facts, use ONLY the ABES INFORMATION above. Never invent or guess timings, fees, dates, percentages, rules, names, numbers or contact details.
3. If the ABES information does not contain the answer, say you do not have that information and tell the student to check the official ABES website or contact the relevant department. Do not say "knowledge base", "context" or "provided information".
4. For time-sensitive topics (fees, admission dates, exam schedules, placement statistics, notices, timings), advise verifying on the official ABES website or portal.
5. You cannot access private student data (attendance, marks, fee status, timetable). Guide the student to the official student portal.
6. If the question is not related to ABES, college or university student life, politely say you can only help with ABES and university-related questions.
7. Never mention Gemini, AI models, prompts or internal systems.
8. Formatting: plain text only. No markdown, no asterisks, no headings. For lists, put each item on a new line starting with "- ".
9. Do not start the answer with a greeting such as "Hello" or "Hi". Start directly with the answer.
10. Do not end with filler like "Let me know if you need help". End after giving the answer.
Answer now.
"""


def ask_gemini(prompt):

    """Try each model in order. Returns (text, model_name) or (None, None)."""

    for model_name in GEMINI_MODELS:

        try:
            print("Trying model:", model_name)

            reply = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if reply.text and reply.text.strip():
                return reply.text.strip(), model_name

        except Exception as e:
            print("Gemini error with", model_name, ":", type(e).__name__)

    return None, None


# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


@app.route("/reset", methods=["POST"])
def reset():

    chat_memory.pop(get_session_id(), None)

    return jsonify({"status": "cleared"})


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(silent=True) or {}

    user_message = str(data.get("message", "")).strip()

    if not user_message:
        return jsonify({"response": "Please enter a question."})

    user_message = user_message[:MAX_MESSAGE_LENGTH]

    sid = get_session_id()

    print("\n==============================")
    print("User Question:", user_message)


    # 1. Small talk

    known_intent = detect_known_intent(user_message)

    if known_intent:

        response = responses[known_intent]

        print("Response Source: Fixed Response")
        print("==============================")

        log_chat(user_message, "fixed", "-", [])

        return jsonify({
            "response": response,
            "intent": known_intent,
            "source": "fixed"
        })


    # 2. ML model (kept as intent classifier + small-talk filter)

    intent = model.predict([user_message])[0]
    confidence = float(max(model.predict_proba([user_message])[0]))

    print("ML Intent:", intent, "| Confidence:", round(confidence, 2))

    if confidence >= 0.80 and intent in ("greeting", "goodbye"):

        response = responses[intent]

        log_chat(user_message, "ml", "-", [])

        return jsonify({
            "response": response,
            "intent": intent,
            "confidence": confidence,
            "source": "ml"
        })


    # 3. Retrieval + Gemini

    if client:

        context_sections, score = retrieve_sections(user_message)
        used_titles = [s["title"] for s in context_sections]

        print("Retrieved sections:", used_titles, "| Score:", round(score, 2))

        prompt = build_prompt(user_message, context_sections, get_history(sid))

        answer, used_model = ask_gemini(prompt)

        if answer:

            add_to_history(sid, user_message, answer)

            print("Response Source: Retrieval + Gemini (" + used_model + ")")
            print("==============================")

            log_chat(user_message, "gemini", used_model, used_titles)

            return jsonify({
                "response": answer,
                "intent": intent,
                "confidence": confidence,
                "source": "gemini",
                "sections": used_titles
            })

    # 4. Fallback

    print("Response Source: Fallback")
    print("==============================")

    log_chat(user_message, "fallback", "-", [])

    return jsonify({
        "response": (
            "I could not get an answer right now. Please try again in a moment, "
            "or check the official ABES website for the latest information."
        ),
        "intent": intent,
        "confidence": confidence,
        "source": "fallback"
    })


if __name__== "__main__":
    app.run(debug=True)