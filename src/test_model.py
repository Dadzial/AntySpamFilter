from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

model = load_model('../model/spam_filter_model.keras')

with open("../model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len=100

def predict_spam(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(pad)[0][0]
    return "Spam" if pred > 0.5 else "Ham", pred

sms = "Congratulations! You won a free ticket."
label, confidence = predict_spam(sms)
print(f"Message: {sms}")
print(f"Prediction: {label} (confidence: {confidence:.2f})")