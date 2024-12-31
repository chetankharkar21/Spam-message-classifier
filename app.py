from flask import Flask, render_template, request
import joblib

# Load the pre-trained model and vectorizer
model = joblib.load('final_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    # Transform the message using the vectorizer
    message_vectorized = vectorizer.transform([message])
    
    # Predict using the model
    prediction = model.predict(message_vectorized)[0]
    
    # If prediction is 1, it's spam; if 0, it's ham
    result = "Spam" if prediction == 1 else "Ham"
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
