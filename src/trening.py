import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import pickle


# Load dataset
df = pd.read_csv("../data/SMSSpamCollection.csv", sep='\t', names=["label", "message"])
# Convert labels to binary values
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
# Print first 5 rows
print(df.head())

#Preparing data
X_train , X_test , Y_train , Y_test = train_test_split(df['message'] , df['label_num'] , test_size=0.2 , random_state=42)

tokenizer = Tokenizer(num_words=5000,oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 150
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

#Building model
model = Sequential([
    Embedding(input_dim=30000, output_dim=16, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#Training model
history = model.fit(
    X_train_pad,
    Y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_test_pad, Y_test)
)

loss, accuracy = model.evaluate(X_test_pad, Y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

#Saving model
model.save('spam_filter_model.keras')

#Saving tokenizer
with open('../model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

