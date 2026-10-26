# UniAssist AI - ABES Student Support Chatbot

## Project Overview

UniAssist AI is a web-based chatbot that answers student questions about ABES Engineering College, Ghaziabad. It combines a Machine Learning intent classifier, TF-IDF retrieval over a verified knowledge base, and the Gemini API to write clear, grounded answers.

Built as part of the HCL Digital AI project.

## Features

- Retrieval-augmented answers: the most relevant knowledge-base sections are retrieved and given to Gemini
- Honest fallback: if the information is not in the knowledge base, the bot says so instead of guessing
- Small-talk handling (greeting, thanks, goodbye)
- Short chat memory for follow-up questions
- Automatic backup model if the primary Gemini model fails
- Light and dark theme with saved preference
- Request logging to a local CSV file for testing

## How It Works

```text
Student question
      |
      v
Small-talk filter (regex)  ->  fixed reply
      |
      v
ML intent classifier (TF-IDF + Logistic Regression)  ->  small-talk fallback and intent logging
      |
      v
TF-IDF retrieval over knowledge-base sections
      |
      v
Gemini (primary model, then backup model) with retrieved context and chat memory
      |
      v
Answer, or safe fallback message
```

## Technologies Used

- Python, Flask
- Scikit-learn, Pandas, NumPy, Joblib
- Google Gemini API (google-genai)
- HTML, CSS, JavaScript

## Machine Learning Approach

1. Text preprocessing
2. TF-IDF vectorization
3. Logistic Regression for intent classification, trained on a public customer-service intent dataset (Bitext, from Kaggle)

The classifier is used as a lightweight first filter and for intent logging. Factual answers about ABES come from the knowledge base through retrieval and Gemini, not from the classifier.

## Project Structure

```text
AI-UNIVERSITY CHATBOT
│
├── app.py                    Flask app, retrieval, Gemini calls
├── train_model.py            Trains the intent classifier
├── check_dataset.py          Dataset inspection
├── view_data.py              Dataset viewing
├── university_chatbot_model.pkl
├── requirements.txt
│
├── data/                     Training dataset
├── knowledge_base/
│   └── abes_info.txt         ABES information used for retrieval
├── templates/
│   └── index.html            Chat interface
└── static/
    └── style.css             Styling and themes
```

## Setup

1. Clone the repository and open the folder.
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_key_here
   FLASK_SECRET_KEY=any_random_string
   ```
5. Run the app:
   ```
   python app.py
   ```
6. Open http://127.0.0.1:5000

## Knowledge Base

`knowledge_base/abes_info.txt` contains public information from the official ABES website. Each section starts with an ALL-CAPS heading. The chatbot is instructed not to invent fees, timings, dates, rules or contact details.

## Limitations

- Attendance rules, examination rules and library timings are not yet in the knowledge base
- The bot cannot access private student data such as marks or attendance records
- The knowledge base is a snapshot; students should verify important details on the official ABES website

## Future Work

- Add attendance, examination and library details from official ABES documents
- Embedding-based semantic retrieval
- Admin page to update the knowledge base
- Cloud deployment