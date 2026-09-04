# events_chatbot.py
# Chatbot for Events platform: Kaggle dataset + keyword matching

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# STEP 1: Load Kaggle intent dataset
with open("Intent.json", "r") as f:
    data = json.load(f)

# Prepare ML training data from dataset
sentences = []
labels = []

for intent in data["intents"]:
    tag = intent["intent"]  # <-- correct key
    for pattern in intent["text"]:
        sentences.append(pattern.lower())
        labels.append(tag)

print("Training samples:", len(sentences))

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

#Train ML model
model = MultinomialNB()
model.fit(X, labels)

# Keyword-based mapping (action intents)
keywords = {
    "book_event": ["book", "reserve", "buy", "create booking", "attend"],
    "cancel_booking": ["cancel", "refund", "delete booking"],
    "contact_info": ["contact", "support", "reach", "get in touch"]
}

#  Knowledge base / Responses
responses = {
    "Greeting": "Hey! Welcome to the Events platform. How can I help?",
    "Goodbye": "Catch you later! Have a great day!",
    "event_info": (
        "We host concerts, nightlife events, tech meetups, "
        "and cultural experiences."
    ),
    "event_categories": (
        "Our events include concerts, parties, festivals, tech events, "
        "and creative meetups."
    ),
    "ticket_info": (
        "Ticket prices vary depending on the event and ticket category. "
        "Each event page shows available ticket options."
    ),
    "book_event": (
        "To book an event, select an event, choose your ticket type, "
        "and proceed to checkout."
    ),
    "cancel_booking": (
        "Ticket cancellations and refunds depend on the event policy. "
        "Please check the event page for refund terms."
    ),
    "contact_info": (
        "You can contact us through the contact section on the website "
        "or via our official social media pages."
    )
}


# STEP 7: Hybrid intent prediction
def predict_intent(user_input):
    user_input = user_input.lower()

    # Keyword-based intents first
    for intent, words in keywords.items():
        if any(word in user_input for word in words):
            return intent

    #ML fallback
    vector = vectorizer.transform([user_input])
    return model.predict(vector)[0]

#Chat loop
print("Events Bot is running. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").lower()
    if user_input == "exit":
        print("Events Bot: Goodbye! See you at the next event 🎶")
        break

    intent = predict_intent(user_input)
    print("Events Bot:", responses.get(intent, "Sorry, I don’t have information on that yet."))
