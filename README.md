# AI-Powered Chatbot for University Student Support

## Project Overview

The AI-Powered Chatbot for University Student Support is a web-based chatbot designed to provide quick assistance to university students.

The chatbot uses Natural Language Processing (NLP) and Machine Learning to understand student queries and identify their intent. Based on the identified intent, the system provides a relevant response.

## Features

- Student-friendly chatbot interface
- Natural Language Processing based query classification
- Machine Learning based intent prediction
- Attendance-related queries
- Examination-related queries
- Academic information
- Library information
- Fee-related queries
- Placement-related queries
- Admission-related queries
- Course information
- College timing information
- University contact/support queries
- Greeting and goodbye handling

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- Joblib
- HTML
- CSS
- JavaScript

## Machine Learning Approach

The chatbot uses the following NLP and Machine Learning techniques:

1. Text preprocessing
2. TF-IDF Vectorization
3. Logistic Regression
4. Intent Classification

The user query is converted into numerical features using TF-IDF. The trained Logistic Regression model then predicts the intent of the query. Based on the predicted intent, the chatbot returns an appropriate response.

## Project Structure

```text
AI-UNIVERSITY CHATBOT
│
├── app.py
├── train_model.py
├── check_dataset.py
├── view_data.py
├── university_chatbot_model.pkl
├── requirements.txt
│
├── data
│   └── Bitext_Sample_Customer_Service_Training_Dataset.csv
│
├── templates
│   └── index.html
│
├── static
│   └── style.css
│
└── venv