# Intent-Based Chatbot

Overview
This project is an intent-based chatbot that uses Natural Language Processing (NLP) techniques to understand user inputs and generate appropriate responses. The chatbot is built using the nltk library for text preprocessing, scikit-learn for machine learning (specifically Logistic Regression), and streamlit for deployment as a web application.

The chatbot is trained on a set of predefined intents, each containing patterns and corresponding responses. It uses a TF-IDF Vectorizer to convert text into numerical features and a Logistic Regression model to predict the intent of the user's input. Based on the predicted intent, the chatbot selects a random response from the available options.

# Features
Intent Recognition: The chatbot can recognize user intents based on predefined patterns.

Dynamic Responses: For each intent, the chatbot provides a random response from a list of possible responses.

Streamlit Interface: The chatbot is deployed as a user-friendly web application using Streamlit.

Customizable Intents: You can easily add, modify, or remove intents in the intents list to customize the chatbot's behavior.

# Technologies Used
Python: The programming language used for development.

NLTK (Natural Language Toolkit): Used for tokenization and text preprocessing.

Scikit-learn: Used for TF-IDF vectorization and Logistic Regression.

Streamlit: Used for deploying the chatbot as a web application.

# How It Works
Training the Model:

The chatbot is trained on a set of intents, where each intent contains patterns (user inputs) and responses.

The patterns are converted into numerical features using a TF-IDF Vectorizer.

A Logistic Regression model is trained on these features to predict the intent of new user inputs.

# Chatbot Interaction:

When a user inputs a message, the chatbot uses the trained model to predict the intent.

Based on the predicted intent, a random response is selected from the list of responses for that intent.

The response is displayed to the user in the Streamlit interface.

# Installation
To run this chatbot locally, follow these steps:

Clone the repository:

git clone https://github.com/Hafsana55/intentschatbot.git

cd intent-based-chatbot

Install the required dependencies:

pip install -r requirements.txt

Download NLTK data:

import nltk
nltk.download('punkt')

Run the Streamlit application:

streamlit run app.py
