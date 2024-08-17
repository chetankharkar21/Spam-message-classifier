import joblib

# Load the trained model and vectorizer
best_model = joblib.load('final_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Sample ham messages
ham_messages = [
    "Let's meet for lunch tomorrow.",
    "Don't forget to send the report by 5 PM.",
    "The project deadline is next week.",
    "See you at the team meeting.",
    "Call me when you are free."
]

# Transform the messages using the same vectorizer
features_ham_transformed = vectorizer.transform(ham_messages)

# Predict using the trained model
predictions = best_model.predict(features_ham_transformed)

# Print the results
for message, prediction in zip(ham_messages, predictions):
    print(f"Message: {message}")
    print(f"Prediction: {'Spam' if prediction == 1 else 'Ham'}")
    print()
