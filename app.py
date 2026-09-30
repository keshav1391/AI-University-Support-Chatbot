from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("university_chatbot_model.pkl")


responses = {
    "greeting": "Hello! 👋 How can I help you with university-related information?",

    "attendance": "You can check your attendance through the university/student portal. Please contact your department if you need further assistance.",

    "exam": "You can check the latest examination schedule through the university/student portal or examination department.",

    "academic": "For academic information such as syllabus, study material, or faculty-related queries, please contact your department or academic office.",

    "library": "The library provides books and academic resources. Please check the university library portal for current timings and book availability.",

    "fees": "For fee payment, fee status, and deadlines, please check the student portal or contact the accounts department.",

    "placement": "For placement information, registration, eligibility, and company updates, please check the placement cell or student placement portal.",

    "admission": "For admission information, eligibility, documents, and application procedures, please check the official university admission portal.",

    "courses": "The university offers multiple courses and programs. Please check the official university website for the latest course information.",

    "college_timing": "College timings may vary by department and semester. Please check your official timetable or department notice.",

    "contact": "For official assistance, please contact the university administration or student support department.",

    "goodbye": "Goodbye! 👋 Have a great day!"
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({
            "response": "Please enter a question."
        })

    intent = model.predict([user_message])[0]

    response = responses.get(
        intent,
        "Sorry, I could not understand your question. Please try asking in a different way."
    )

    return jsonify({
        "response": response,
        "intent": intent
    })


if __name__ == "__main__":
    app.run(debug=True)