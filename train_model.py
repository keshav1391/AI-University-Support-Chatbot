import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib


data = [
    # GREETING
    ("hello", "greeting"),
    ("hi", "greeting"),
    ("hey", "greeting"),
    ("hello chatbot", "greeting"),
    ("good morning", "greeting"),
    ("good evening", "greeting"),

    # ATTENDANCE
    ("how can I check my attendance", "attendance"),
    ("where can I see my attendance", "attendance"),
    ("what is my attendance percentage", "attendance"),
    ("how much attendance is required", "attendance"),
    ("what is the minimum attendance requirement", "attendance"),
    ("can I check attendance online", "attendance"),

    # EXAM
    ("when are the exams", "exam"),
    ("when is my exam", "exam"),
    ("where can I find exam schedule", "exam"),
    ("when will exams start", "exam"),
    ("tell me about examination dates", "exam"),
    ("where is the exam timetable", "exam"),

    # ACADEMIC
    ("where can I find my syllabus", "academic"),
    ("tell me about my syllabus", "academic"),
    ("where can I get study material", "academic"),
    ("how can I contact my faculty", "academic"),
    ("where can I find course material", "academic"),
    ("I need academic information", "academic"),

    # LIBRARY
    ("what are the library timings", "library"),
    ("when does the library open", "library"),
    ("when does the library close", "library"),
    ("how can I issue a book", "library"),
    ("can I borrow books from library", "library"),
    ("where is the college library", "library"),

    # FEES
    ("how can I pay my fees", "fees"),
    ("where can I pay college fees", "fees"),
    ("what is the fee structure", "fees"),
    ("tell me about semester fees", "fees"),
    ("how can I check my fee status", "fees"),
    ("when is the fee payment deadline", "fees"),

    # PLACEMENT
    ("how can I register for placement", "placement"),
    ("tell me about placements", "placement"),
    ("where can I find placement information", "placement"),
    ("which companies visit campus", "placement"),
    ("how do I apply for placement", "placement"),
    ("what are the placement requirements", "placement"),

    # ADMISSION
    ("how can I get admission", "admission"),
    ("what is the admission process", "admission"),
    ("tell me about admission", "admission"),
    ("what documents are required for admission", "admission"),
    ("how can I apply for admission", "admission"),
    ("what are the admission requirements", "admission"),

    # COURSES
    ("what courses are available", "courses"),
    ("which branches are available", "courses"),
    ("what programs does the college offer", "courses"),
    ("tell me about available courses", "courses"),
    ("which engineering branches are available", "courses"),
    ("what degrees are offered", "courses"),

    # COLLEGE TIMING
    ("what are college timings", "college_timing"),
    ("when does college start", "college_timing"),
    ("when does college end", "college_timing"),
    ("what is the college schedule", "college_timing"),
    ("when should students report to college", "college_timing"),
    ("tell me college working hours", "college_timing"),

    # CONTACT
    ("how can I contact the college", "contact"),
    ("what is the college contact number", "contact"),
    ("how can I contact administration", "contact"),
    ("where is the administration office", "contact"),
    ("how can I contact student support", "contact"),
    ("I need help from college support", "contact"),

    # GOODBYE
    ("bye", "goodbye"),
    ("goodbye", "goodbye"),
    ("see you later", "goodbye"),
    ("thanks bye", "goodbye"),
]


df = pd.DataFrame(data, columns=["utterance", "intent"])


X = df["utterance"]
y = df["intent"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("University chatbot model trained successfully!")
print("Accuracy:", accuracy)


joblib.dump(model, "university_chatbot_model.pkl")

print("Model saved as university_chatbot_model.pkl")