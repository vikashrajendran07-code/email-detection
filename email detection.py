import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- STEP 1: LOAD DATASET ---
# In a real scenario, use: df = pd.read_csv('your_dataset.csv')
data = {
    'text': [
        "Your account is locked! Click here to reset: http://bit.ly/secure-login",
        "Congratulations! You've won a $500 gift card. Claim now.",
        "Meeting reminder: Project sync tomorrow at 10 AM in Room 4.",
        "URGENT: Verify your bank details immediately or your card will be blocked.",
        "Hey, are we still going to the gym tonight?",
        "Final notice: Your invoice #402 is past due. Pay at http://malicious-link.net",
        "The updated schedule for the conference is attached below.",
        "Win a free iPhone 15 by clicking this link now!"
    ],
    'label': ['Phishing', 'Phishing', 'Safe', 'Phishing', 'Safe', 'Phishing', 'Safe', 'Phishing']
}

df = pd.DataFrame(data)

# --- STEP 2: BUILD THE PIPELINE ---
# The Pipeline combines the Vectorizer and the Classifier into one object
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', lowercase=True)),
    ('nb', MultinomialNB())
])

# --- STEP 3: TRAIN & EVALUATE ---
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

# Show Metrics
predictions = model_pipeline.predict(X_test)
print("--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("-" * 30)

# --- STEP 4: INTERACTIVE USER INPUT ---
def predict_email():
    print("\n--- Phishing Detector ---")
    print("Enter the email content to check (or type 'quit' to exit):")
    
    while True:
        user_input = input("\nEmail Content: ")
        if user_input.lower() == 'quit':
            break
        
        # Make a prediction
        result = model_pipeline.predict([user_input])
        probability = model_pipeline.predict_proba([user_input])
        
        print(f"Result: This email is classified as **{result[0]}**")
        print(f"Confidence: {max(probability[0]) * 100:.2f}%")

if __name__ == "__main__":
    predict_email()