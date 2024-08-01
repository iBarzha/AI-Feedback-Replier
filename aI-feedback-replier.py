import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from datasets import load_dataset
import random

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

dataset = load_dataset('imdb')
df_train = pd.DataFrame(dataset['train'])
df_test = pd.DataFrame(dataset['test'])

df = pd.concat([df_train, df_test])

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['text'] = df['text'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)

vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=300)
model.fit(X_train_vec, y_train)

# Оценка точности модели
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

positive_responses = [
    "Thank you for your positive feedback! We are glad you liked it.",
    "We're thrilled that you enjoyed it! Your positive review means a lot to us.",
    "Thank you for the kind words! We're happy to hear that you had a good experience.",
    "We appreciate your positive feedback and are glad you enjoyed it!",
    "It's great to hear you had a good time! Thanks for the positive review."
]

negative_responses = [
    "We are sorry that you did not enjoy it. We will try to improve our product.",
    "We apologize for the inconvenience. Your feedback is important and we will work on it.",
    "We're sorry to hear that your experience was not satisfactory. We are striving to do better.",
    "Thank you for your feedback. We're sorry for the disappointment and will work to improve.",
    "We regret that we did not meet your expectations. Your feedback helps us improve."
]

def generate_response(text):
    text_preprocessed = preprocess_text(text)
    text_vec = vectorizer.transform([text_preprocessed])
    prediction = model.predict(text_vec)[0]
    if prediction == 1:
        return random.choice(positive_responses)
    else:
        return random.choice(negative_responses)

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chat ended.")
        break

    response = generate_response(user_input)
    print(f"AI: {response}")
